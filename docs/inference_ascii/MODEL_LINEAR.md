# MODEL_LINEAR.md - 线性层与嵌入层详解 (ASCII 版本)

## 目录

- [1. 概述](#1-概述)
- [2. ParallelEmbedding](#2-parallelembedding)
- [3. Linear](#3-linear)
- [4. ColumnParallelLinear](#4-columnparallellinear)
- [5. RowParallelLinear](#5rowparallellinear)
- [6. 数据流图](#6-数据流图)

## 1. 概述

DeepSeek-V3.2-Exp 使用**张量并行 (Tensor Parallelism)** 进行分布式训练和推理。线性层分为列并行和行并行两种：

```
输入 x (M, K)
      │
      ▼
ColumnParallel (按列切分权重)
      │
      ▼
计算 y1 (M, N/world_size)
      │
      ▼
后续计算
      │
      ▼
RowParallel (按行切分权重)
      │
      ▼
计算 y2 (M, N/world_size)
      │
      ▼
AllReduce (求和)
      │
      ▼
输出 y (M, N)
```

## 2. ParallelEmbedding

### 2.1 类定义

**位置**: `model.py:L93-L132`

```python
class ParallelEmbedding(nn.Module):
    """支持并行的嵌入层"""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, dim))
```

### 2.2 词汇表切分

```
完整词汇表 (vocab_size = 102400)
      │
      ├───► GPU 0: token 0-12799
      │
      ├───► GPU 1: token 12800-25599
      │
      ├───► GPU 2: token 25600-38399
      │
      ├───► ...
      │
      └───► GPU 7: token 114560-102399
```

**切分公式**：
- 每个卡负责 vocab_size / world_size 个 token
- GPU r 负责 [r × part, (r+1) × part)

### 2.3 前向传播

**位置**: `model.py:L111-L132`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if world_size > 1:
        mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
        x = x - self.vocab_start_idx
        x[mask] = 0
    y = F.embedding(x, self.weight)
    if world_size > 1:
        y[mask] = 0
        dist.all_reduce(y)
    return y
```

#### 计算流程

```
输入 token_ids (B, S)
      │
      ▼
world_size > 1?
      │
      ├─── 否 ──► 直接 embedding
      │
      └─── 是 ──► 创建 mask (标记非本卡的 token)
                    │
                    ▼
              偏移 token_id (x -= vocab_start_idx)
                    │
                    ▼
              Embedding 查找
                    │
                    ▼
              将非本卡的输出置 0
                    │
                    ▼
              AllReduce Sum (跨卡求和)
                    │
                    ▼
              输出 (B, S, D)
```

#### 张量形状

| 阶段 | 形状 | 说明 |
|------|------|------|
| 输入 x | (B, S) | token IDs |
| mask | (B, S) | 布尔掩码 |
| y (AllReduce 前) | (B, S, D) | 非本卡位置为 0 |
| y (AllReduce 后) | (B, S, D) | 完整嵌入 |

## 3. Linear

### 3.1 类定义

**位置**: `model.py:L167-L206`

```python
class Linear(nn.Module):
    dtype = torch.bfloat16
    scale_fmt: Optional[str] = None

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features,
                                               dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:  # FP8
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        # ... bias 处理
```

### 3.2 FP8 权重存储

**位置**: `model.py:L185-L190`

```python
if self.weight.element_size() == 1:
    # FP8 权重需要额外的 scale 参数
    scale_out_features = (out_features + block_size - 1) // block_size
    scale_in_features = (in_features + block_size - 1) // block_size
    self.weight.scale = nn.Parameter(
        torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
```

#### 块级量化布局

```
权重矩阵 (out × in)
      │
      ▼
分割为 128×128 块
      │
      ├───► 块 0,0: FP8 + scale
      │
      ├───► 块 0,1: FP8 + scale
      │
      ├───► ...
      │
      └───► 块 m,n: FP8 + scale
```

**Scale 张量形状**：
scale.shape = (⌈out_features / 128⌉, ⌈in_features / 128⌉)

### 3.3 前向传播

**位置**: `model.py:L196-L206`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return linear(x, self.weight, self.bias, self.scale_fmt)
```

调用 `MODEL_BASE.md` 中定义的 `linear()` 函数。

## 4. ColumnParallelLinear

### 4.1 类定义

**位置**: `model.py:L209-L235`

```python
class ColumnParallelLinear(Linear):
    """列并行线性层：输出维度被切分"""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)
```

### 4.2 权重切分

```
完整权重矩阵 (out × in)
      │
      ├───► GPU 0: out/8 × in (行 0-(out/8-1))
      │
      ├───► GPU 1: out/8 × in (行 out/8-(2×out/8-1))
      │
      ├───► ...
      │
      └───► GPU 7: out/8 × in (行 7×out/8-out)
```

**切分方式**：按**输出维度**（行）切分。

### 4.3 前向传播

**位置**: `model.py:L224-L235`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = linear(x, self.weight, self.bias, self.scale_fmt)
    return y
```

**特点**：输出**不需要** AllReduce，因为每个卡的输出就是最终输出的一部分。

### 4.4 张量形状

| 变量 | 完整形状 | 单卡形状 |
|------|----------|----------|
| 输入 x | (M, K) | (M, K) - 完整 |
| 权重 W | (N, K) | (N/8, K) - 切分 |
| 输出 y | (M, N) | (M, N/8) - 部分 |

## 5. RowParallelLinear

### 5.1 类定义

**位置**: `model.py:L238-L270`

```python
class RowParallelLinear(Linear):
    """行并行线性层：输入维度被切分"""
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 reduce_output = True, dtype = None):
        assert in_features % world_size == 0
        self.part_in_features = in_features // world_size
        self.reduce_output = reduce_output
        super().__init__(self.part_in_features, out_features, bias, dtype)
```

### 5.2 权重切分

```
完整权重矩阵 (out × in)
      │
      ├───► GPU 0: out × in/8 (列 0-(in/8-1))
      │
      ├───► GPU 1: out × in/8 (列 in/8-(2×in/8-1))
      │
      ├───► ...
      │
      └───► GPU 7: out × in/8 (列 7×in/8-in)
```

**切分方式**：按**输入维度**（列）切分。

### 5.3 前向传播

**位置**: `model.py:L254-L270`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = linear(x, self.weight, None, self.scale_fmt)
    if self.reduce_output and world_size > 1:
        y = y.float()
        dist.all_reduce(y)
    if self.bias is not None:
        y += self.bias
    return y.type_as(x)
```

**关键步骤**：
1. 本地矩阵乘法
2. **AllReduce** 求和（跨卡）
3. 添加 bias
4. 类型转换回原类型

### 5.4 AllReduce 操作

```
GPU 0: y0 (M, N) ──┐
GPU 1: y1 (M, N) ──┤
                   ├──► AllReduce Sum (NCCL)
GPU 7: y7 (M, N) ──┘
                   │
       ┌───────────┴───────────┐
       ▼       ▼       ▼       ▼
   GPU 0:   GPU 1:  ...   GPU 7:
 y0+y1+... y0+y1+...      y0+y1+...
```

### 5.5 张量形状

| 变量 | 完整形状 | 单卡输入 | 单卡输出 |
|------|----------|----------|----------|
| 输入 x | (M, K) | (M, K/8) | - |
| 权重 W | (N, K) | (N, K/8) | - |
| 本地 y | - | (M, N) | (M, N) |
| AllReduce 后 | - | - | (M, N) |

### 5.6 reduce_output 参数

```python
self.reduce_output = reduce_output
```

- `True` (默认): 执行 AllReduce
- `False`: 不执行 AllReduce（用于某些特殊场景）

## 6. 数据流图

### 6.1 完整前向传播（张量并行）

```
输入层
    │
    ▼
ParallelEmbedding (切分词汇表)
    │
    ▼
Transformer Layers
    │
    ├──► Layer 0
    ├──► Layer 1
    └──► ...
    │
    ▼
输出层
    │
    ├──► RMSNorm
    ├──► ColumnParallelLinear (切分输出)
    │
    ▼
AllGather (收集完整输出)
    │
    ▼
Logits (B, vocab_size)
```

### 6.2 单层内部数据流

```
输入 x (M, K)
      │
      ▼
ColumnParallelLinear (wq_a)
      │
      ▼
RMSNorm
      │
      ▼
ColumnParallelLinear (wq_b)
      │
      ▼
MLA Attention (Indexer + Attn)
      │
      ▼
ColumnParallelLinear (w1/w3)
      │
      ▼
MoE (专家计算)
      │
      ▼
RowParallelLinear (w2, 含 AllReduce)
      │
      ▼
输出 (M, K)
```

### 6.3 FP8 量化数据流

```
输入 x (M, K) BF16
      │
      ▼
weight.dtype?
      │
      ├─── BF16 ──► 标准 F.linear ((M, K) @ (N, K)^T)
      │
      └─── FP8 ──► act_quant (量化 x)
                     │
                     ▼
                  x: FP8, scale: FP32
                     │
                     ▼
                  fp8_gemm (FP8 矩阵乘法)
                     │
                     ▼
                  输出 y (M, N) BF16
```

---

**下一步**: 阅读 [MODEL_NORM.md](MODEL_NORM.md) 了解归一化层的实现。
