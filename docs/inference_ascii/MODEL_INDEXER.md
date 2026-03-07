# MODEL_INDEXER.md - DSA Indexer 模块详解 (ASCII 版本)

## 目录

- [1. 概述](#1-概述)
- [2. DSA (DeepSeek Sparse Attention) 原理](#2-dsa-deepseek-sparse-attention-原理)
- [3. Indexer 类定义](#3-indexer-类定义)
- [4. forward 方法详解](#4-forward-方法详解)
- [5. 完整数据流](#5-完整数据流)

## 1. 概述

**Indexer** 是 DeepSeek Sparse Attention (DSA) 的核心组件，负责从整个 KV Cache 中**选择最相关的 Top-K 个位置**进行稀疏注意力计算。

```
当前 Query
      │
      ▼
Indexer (计算相关性)
      │
      ▼
Top-K 选择 (K=2048)
      │
      ▼
从 KV Cache 中仅取选中位置
      │
      ▼
稀疏注意力计算
```

## 2. DSA (DeepSeek Sparse Attention) 原理

### 2.1 稀疏注意力动机

**标准注意力**：计算 Query 与所有 Key 的相关性
Attention(Q, K) = softmax(QK^T / sqrt(d_k))

**复杂度**：O(S²)，S 是序列长度

**DSA 稀疏注意力**：只计算 Top-K 个最相关的位置
DSA-Attention(Q, K) = softmax(Q · K_topk^T / sqrt(d_k))

**复杂度**：O(S × K)，K=2048

### 2.2 Indexer 的作用

Indexer **不是**计算最终的注意力值，而是：

1. 计算每个位置的 **index score**（相关性分数）
2. 选择 **Top-K** 个位置
3. 返回位置索引，供 MLA 模块使用

```
Indexer 输入: Q, K, 位置编码
      │
      ▼
Index Score 计算 (FP8 高效计算)
      │
      ▼
Top-K 选择 (默认 K=2048)
      │
      ▼
输出: topk_indices
      │
      ▼
MLA 模块使用 (仅计算选中位置的注意力)
```

### 2.3 Index Score 公式

index_score[i, j] = Σ(ReLU(score_h[i, j]) × w_h)

其中：
- score_h[i, j] 是第 h 个 head 对位置 j 的原始分数
- w_h 是该 head 的权重
- H 是 Indexer head 数量（默认 64）

## 3. Indexer 类定义

### 3.1 类结构

**位置**: `model.py:L460-L479`

```python
class Indexer(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim: int = args.dim
        self.n_heads: int = args.index_n_heads          # 64
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim: int = args.index_head_dim        # 128
        self.rope_head_dim: int = args.qk_rope_head_dim # 64
        self.index_topk: int = args.index_topk          # 2048
        self.q_lora_rank: int = args.q_lora_rank        # 0

        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wk = Linear(self.dim, self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)
        self.softmax_scale = self.head_dim ** -0.5
        self.scale_fmt = args.scale_fmt

        self.register_buffer("k_cache", torch.zeros(..., dtype=torch.float8_e4m3fn))
        self.register_buffer("k_scale_cache", torch.zeros(..., dtype=torch.float32))
        self._trace_layer_id: int = -1
```

### 3.2 参数表

| 参数 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `wq_b.weight` | (n_h × d, q_rank) | BF16/FP8 | Q 投影权重 |
| `wk.weight` | (d, d_h) | BF16/FP8 | K 投影权重 |
| `weights_proj.weight` | (d, n_h) | FP32 | Per-head 权重 |
| `k_cache` | (B, S, d_h) | FP8 | K 值缓存 |
| `k_scale_cache` | (B, S, d_h/128) | FP32 | K scale 缓存 |

### 3.3 模块架构

```
输入 x, qr
      │
      ├───► wq_b (Q 投影)
      │         │
      │         ▼
      │    Q 处理 (RoPE + Hadamard + FP8)
      │
      ├───► wk (K 投影)
      │         │
      │         ▼
      │    K 处理 (LayerNorm + RoPE + Hadamard + FP8)
      │
      └───► weights_proj (Head 权重)
                │
                ▼
           权重广播
```

## 4. forward 方法详解

**位置**: `model.py:L483-L538`

### 4.1 函数签名

```python
def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int,
            freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
```

| 参数 | 形状 | 说明 |
|------|------|------|
| `x` | (B, S, d) | 输入隐藏状态 |
| `qr` | (B, S, q_rank) | Q 的 LoRA 投影结果 |
| `start_pos` | int | 当前起始位置 |
| `freqs_cis` | (S, d_rope/2) | 预计算的 RoPE |
| `mask` | (S, S) or None | 注意力掩码 |

### 4.2 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 1: Q 投影                                                  │
├─────────────────────────────────────────────────────────────────┤
│ wq_b 投影: q = wq_b@qr                                          │
│       │                                                         │
│       ▼                                                         │
│ reshape: (B, S, H, D)                                            │
│       │                                                         │
│       ▼                                                         │
│ split: q_pe, q_nope                                            │
│       │                                                         │
│       ▼                                                         │
│ RoPE 应用: q_pe = RoPE                                         │
│       │                                                         │
│       ▼                                                         │
│ 拼接: q = cat                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 2: K 投影                                                  │
├─────────────────────────────────────────────────────────────────┤
│ wk 投影: k = wk@x                                               │
│       │                                                         │
│       ▼                                                         │
│ LayerNorm: k_norm                                              │
│       │                                                         │
│       ▼                                                         │
│ split: k_pe, k_nope                                            │
│       │                                                         │
│       ▼                                                         │
│ RoPE 应用: k_pe = RoPE                                         │
│       │                                                         │
│       ▼                                                         │
│ 拼接: k = cat                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 3: Hadamard + FP8                                          │
├─────────────────────────────────────────────────────────────────┤
│ q: rotate_activation                                           │
│       │                                                         │
│       ▼                                                         │
│ k: rotate_activation                                           │
│       │                                                         │
│       ▼                                                         │
│ q_fp8, q_scale = act_quant                                    │
│       │                                                         │
│       ▼                                                         │
│ k_fp8, k_scale = act_quant                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 4: 更新 Cache                                              │
├─────────────────────────────────────────────────────────────────┤
│ k_cache 写入: k_fp8                                            │
│ k_scale 写入                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 5: 计算 Weights                                            │
├─────────────────────────────────────────────────────────────────┤
│ weights_proj: w = weights_proj@x                               │
│       │                                                         │
│       ▼                                                         │
│ 归一化: × H^(-0.5) × softmax_scale                            │
│       │                                                         │
│       ▼                                                         │
│ 乘以 q_scale: × q_scale                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 6: Index Score                                             │
├─────────────────────────────────────────────────────────────────┤
│ 从 Cache 读取 k (end_pos 位置)                                  │
│       │                                                         │
│       ▼                                                         │
│ reshape: 连续化                                                 │
│       │                                                         │
│       ▼                                                         │
│ 读取 k_scale                                                   │
│       │                                                         │
│       ▼                                                         │
│ reshape: 连续化                                                 │
│       │                                                         │
       ▼                                                         │
│ fp8_index: 计算 score                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 7: Top-K 选择                                              │
├─────────────────────────────────────────────────────────────────┤
│ index_score                                                    │
│       │                                                         │
│       ▼                                                         │
│ topk (k=2048)                                                  │
│       │                                                         │
│       ▼                                                         │
│ Broadcast: 多卡一致性                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 8: Trace                                                   │
├─────────────────────────────────────────────────────────────────┤
│ tracer enabled?                                                │
│       │                                                         │
│   是 ──► record_dsa_topk                                        │
│       │         │                                               │
│   否 ──► 输出                                                   │
│       │                                                         │
│       ▼                                                         │
│ 输出 topk_indices                                              │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 关键代码段解读

#### 4.3.1 Q 投影与处理

```python
# model.py:L486-L491
q = self.wq_b(qr)
q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
q_pe = apply_rotary_emb(q_pe, freqs_cis, False)
q = torch.cat([q_pe, q_nope], dim=-1)
```

**形状变化**：

| 阶段 | 形状 | 说明 |
|------|------|------|
| `qr` | (B, S, q_rank) | 输入（q_rank=0 时为空） |
| `q` | (B, S, H × D) | 投影后，H=64, D=128 |
| `q` view 后 | (B, S, H, D) | 分离 head |
| `q_pe` | (B, S, H, 64) | RoPE 部分 |
| `q_nope` | (B, S, H, 64) | 非 RoPE 部分 |
| `q` 拼接后 | (B, S, H, 128) | 完整 Q |

#### 4.3.2 K 投影与处理

```python
# model.py:L492-L497
k = self.wk(x)
k = self.k_norm(k)
k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
k = torch.cat([k_pe, k_nope], dim=-1)
```

**形状变化**：

| 阶段 | 形状 | 说明 |
|------|------|------|
| `x` | (B, S, d) | 输入，d=2048 |
| `k` | (B, S, D) | 投影后，D=128 |
| `k` normalize | (B, S, D) | LayerNorm 后 |
| `k_pe` | (B, S, 64) | RoPE 部分 |
| `k_nope` | (B, S, 64) | 非 RoPE 部分 |
| `k` 拼接后 | (B, S, 128) | 完整 K |

#### 4.3.3 Hadamard 变换与 FP8 量化

```python
# model.py:L498-L501
q = rotate_activation(q)
k = rotate_activation(k)
q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
```

#### 4.3.4 计算 Weights

```python
# model.py:L503-L507
weights = self.weights_proj(x)
weights = weights.view(bsz, seqlen, self.n_heads)
weights = (weights * self.softmax_scale * self.head_dim**-0.5).float()
weights = weights * q_scale.transpose(-1, -2)
```

#### 4.3.5 Index Score 计算

```python
# model.py:L509-L516
index_score = fp8_index(
    q_fp8.contiguous(), weights.contiguous(),
    self.k_cache[:bsz, :end_pos].contiguous(),
    self.k_scale_cache[:bsz, :end_pos].contiguous(),
)
```

#### 4.3.6 Top-K 选择

```python
# model.py:L517-L525
topk_weights, topk_indices = torch.topk(index_score, self.index_topk, dim=-1)
if world_size > 1:
    dist.broadcast(topk_indices, src=0)
```

## 5. 完整数据流

```
输入: x (B, S, d), qr (B, S, q_rank), start_pos, freqs_cis
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Q 路径                                                      │
├─────────────────────────────────────────────────────────────┤
│ wq_b(qr) → view → split(q_pe, q_nope)                      │
│   → RoPE(q_pe) → cat → rotate_activation → act_quant       │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ K 路径                                                      │
├─────────────────────────────────────────────────────────────┤
│ wk(x) → LayerNorm → split(k_pe, k_nope)                     │
│   → RoPE(k_pe) → cat → rotate_activation → act_quant       │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Weights 路径                                                │
├─────────────────────────────────────────────────────────────┤
│ weights_proj(x) → view → 归一化 → × q_scale                │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ fp8_index: 计算 index_score                                │
│   = ReLU(q_fp8 @ k_cache^T) × weights × k_scale           │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ topk(index_score, k=2048) → topk_indices                  │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
输出: topk_indices (B, S, 2048)
```

---

**下一步**: 阅读 [MODEL_MLA.md](MODEL_MLA.md) 了解 MLA Attention 模块的实现。
