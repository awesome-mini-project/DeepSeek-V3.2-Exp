# MODEL_MOE.md - 混合专家系统详解 (ASCII 版本)

## 目录

- [1. 概述](#1-概述)
- [2. Gate 门控模块](#2-gate-门控模块)
- [3. Expert 专家模块](#3-expert-专家模块)
- [4. MoE 混合专家模块](#4-moe-混合专家模块)
- [5. 完整数据流](#5-完整数据流)

## 1. 概述

DeepSeek-V3.2-Exp 使用 **MoE (Mixture of Experts)** 架构，允许模型根据输入动态选择激活不同的专家子集。

```
输入 x
      │
      ▼
Gate (路由选择)
      │
      ├───► Expert 1 ────┐
      ├───► Expert 2 ────┤
      ├───► ...      ────┤
      └───► Expert N ────┤
                      │
                      ▼
                 加权求和
                      │
                      ▼
              + Shared Experts (共享专家)
                      │
                      ▼
                   输出
```

## 2. Gate 门控模块

### 2.1 类定义

**位置**: `model.py:L699-L762`

```python
class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts      # 6
        self.n_groups = args.n_expert_groups       # 1
        self.topk_groups = args.n_limited_groups   # 1
        self.score_func = args.score_func          # "softmax"
        self.route_scale = args.route_scale        # 1.0
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))
```

### 2.2 参数

| 参数 | 形状 | 说明 |
|------|------|------|
| `weight` | (n_exp, d) | 专家权重矩阵 |
| `bias` | (n_exp,) | 专家偏置 |

### 2.3 forward 方法

**位置**: `model.py:L730-L762`

```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = linear(x.float(), self.weight.float())
    if self.score_func == "softmax":
        scores = scores.softmax(dim=-1)
    else:
        scores = scores.sigmoid()
    original_scores = scores
    if self.bias is not None:
        scores = scores + self.bias
    # ... 分组路由逻辑 ...
    indices = scores.topk(self.topk, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    if self.score_func == "sigmoid":
        weights /= weights.sum(dim=-1, keepdim=True)
    weights *= self.route_scale
    return weights, indices
```

### 2.4 计算流程

```
输入 x (M, d)
      │
      ▼
线性投影: scores = x @ W^T
      │
      ▼
score_func?
      │
      ├─── softmax ──► Softmax (∑=1)
      │
      └─── sigmoid ──► Sigmoid (0,1)
             │
             ▼
       有 bias?
             │
        ├─── 是 ──► scores += bias
        │    │
        └─── 否 ──► Top-K 选择
                  │
                  ▼
         indices: Top-K 索引
         weights: 对应权重
                  │
                  ▼
            sigmoid?
                  │
           ├─── 是 ──► 归一化 (weights /= sum)
           │    │
           └─── 否 ──► × route_scale
                      │
                      ▼
              输出 weights, indices
```

### 2.5 输出形状

| 变量 | 形状 | 说明 |
|------|------|------|
| 输入 `x` | (M, d) | 展平后的输入 |
| `scores` | (M, n_exp) | 每个专家的分数 |
| `indices` | (M, k) | Top-K 专家索引，k=6 |
| `weights` | (M, k) | Top-K 专家权重 |

## 3. Expert 专家模块

### 3.1 类定义

**位置**: `model.py:L765-L797`

```python
class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)
```

### 3.2 SwiGLU 激活函数

**公式**：
SwiGLU(x) = SiLU(xW_1) · (xW_3)

其中：
- SiLU(x) = x / (1 + e^(-x)) = x · σ(x)
- · 是逐元素乘法

### 3.3 forward 方法

**位置**: `model.py:L787-L797`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))
```

### 3.4 计算流程

```
输入 x (M, d)
      │
      ├───► w1: d→inter
      │     │
      │     ▼
      │  SiLU 激活
      │
      └───► w3: d→inter
            │
            ▼
      逐元素乘: SiLU(w1x) × w3x
            │
            ▼
         w2: inter→d
            │
            ▼
      输出 y (M, d)
```

### 3.5 张量形状

| 阶段 | 形状 | 说明 |
|------|------|------|
| 输入 `x` | (M, d) | d=2048 |
| `w1(x)` | (M, inter) | inter=1408 |
| `w3(x)` | (M, inter) | inter=1408 |
| `SiLU(w1x) × w3x` | (M, inter) | SwiGLU 激活 |
| `w2(...)` | (M, d) | 投影回原维度 |

## 4. MoE 混合专家模块

### 4.1 类定义

**位置**: `model.py:L800-L857`

```python
class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts          # 64
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts    # 6
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(...) if ... else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim,
                                   reduce_output=False)
```

### 4.2 分布式配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_routed_experts` | 64 | 总路由专家数 |
| `n_local_experts` | 64/world_size | 每个卡的专家数 |
| `n_activated_experts` | 6 | 每次激活的专家数 |
| `n_shared_experts` | 2 | 共享专家数 |

### 4.3 forward 方法

**位置**: `model.py:L831-L857`

```python
def forward(self, x: torch.Tensor):
    bsz, seqlen, dim = x.size()
    x = x.view(-1, dim)
    weights, indices = self.gate(x)

    y = torch.zeros_like(x, dtype=torch.float32)

    # 处理路由专家
    for i in range(self.n_local_experts):
        expert_idx = self.experts_start_idx + i
        mask = (indices == expert_idx).any(dim=-1)
        if mask.any():
            expert_input = x[mask]
            expert_output = self.experts[i](expert_input)
            expert_weights = weights[mask].unsqueeze(-1)
            for j in range(self.topk):
                expert_mask = (indices[mask][:, j] == expert_idx)
                if expert_mask.any():
                    y[mask] += expert_output * expert_weights[:, j:j+1] * expert_mask.unsqueeze(-1)

    # 共享专家
    shared_output = self.shared_experts(x)

    return (y + shared_output).view(bsz, seqlen, dim)
```

### 4.4 计算流程

```
输入 x (B, S, d)
      │
      ▼
flatten: (M, d)
      │
      ▼
┌─────────────────────────────────────┐
│ Gate                                │
│   输入 x → scores → topk            │
│   输出: weights (M, k), indices (M, k) │
└─────────────────────────────────────┘
      │
      ├───────────────────────────────────────────────────┐
      │                                                   │
      ▼                                                   ▼
┌─────────────────────┐                         ┌─────────────────┐
│ 路由专家处理         │                         │ 共享专家        │
├─────────────────────┤                         ├─────────────────┤
│ for i in local_experts:                      │                 │
│   mask = indices==i   │                         │ shared_experts │
│   expert_input = x[mask]                       │    (x)          │
│   expert_output = experts[i](expert_input)     │                 │
│   y += output × weights × mask                  │                 │
└─────────────────────┘                         └─────────────────┘
      │                                                   │
      └───────────────────────┬───────────────────────────┘
                              ▼
                        y = y + shared_output
                              │
                              ▼
                        reshape: (B, S, d)
                              │
                              ▼
                        输出
```

## 5. 完整数据流

```
输入 x (B, S, d)
      │
      ▼
flatten (M = B × S)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Gate 路由                                                   │
├─────────────────────────────────────────────────────────────┤
│ scores = x @ W^T                                           │
│ scores = softmax(scores)                                   │
│ weights, indices = topk(scores, k=6)                       │
└─────────────────────────────────────────────────────────────┘
      │
      ├───────────────────────────────┬───────────────────────┤
      │                               │
      ▼                               ▼
┌─────────────────────┐       ┌─────────────────┐
│ 路由专家 (64 个)     │       │ 共享专家 (2 个)  │
├─────────────────────┤       ├─────────────────┤
│ 仅激活 Top-K=6 个    │       │ 总是激活        │
│ 每个专家:          │       │ SwiGLU MLP      │
│   Expert(x)        │       │                 │
│ 按 weights 加权求和  │       │                 │
└─────────────────────┘       └─────────────────┘
      │                               │
      └───────────────┬───────────────┘
                      ▼
                   求和
                      │
                      ▼
              reshape: (B, S, d)
                      │
                      ▼
                   输出
```

## 6. MoE 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_routed_experts` | 64 | 路由专家总数 |
| `n_shared_experts` | 2 | 共享专家数 |
| `n_activated_experts` | 6 | 每次激活的专家数 |
| `moe_inter_dim` | 1408 | 专家中间维度 |
| `score_func` | "softmax" | 路由评分函数 |
| `route_scale` | 1.0 | 路由缩放因子 |

---

**下一步**: 阅读 [MODEL_MLP.md](MODEL_MLP.md) 了解前馈网络的实现。
