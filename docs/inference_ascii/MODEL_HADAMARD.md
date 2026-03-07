# MODEL_HADAMARD.md - Hadamard 变换详解 (ASCII 版本)

## 目录

- [1. 概述](#1-概述)
- [2. Hadamard 变换基础](#2-hadamard-变换基础)
- [3. _hadamard_transform_pytorch](#3-_hadamard_transform_pytorch)
- [4. rotate_activation](#4-rotate_activation)
- [5. 在 DSA Indexer 中的作用](#5-在-dsa-indexer-中的作用)

## 1. 概述

DeepSeek-V3.2-Exp 在 DSA Indexer 中使用 **归一化 Hadamard 变换**来混合通道，用于 QuaRot/SpinQuant 风格的量化。

```
激活 x
   │
   ▼
Hadamard 变换 (H @ x)
   │
   ▼
归一化 (× d^(-1/2))
   │
   ▼
FP8 量化
   │
   ▼
Indexer 计算
```

## 2. Hadamard 变换基础

### 2.1 定义

Hadamard 矩阵 H_n 是一个 n × n 的矩阵（n 是 2 的幂）：

H_1 = [1]
H_2n = [H_n  H_n]
       [H_n -H_n]

### 2.2 小规模示例

H_2 = [ 1   1]
      [ 1  -1]

H_4 = [ 1   1   1   1]
      [ 1  -1   1  -1]
      [ 1   1  -1  -1]
      [ 1  -1  -1   1]

### 2.3 性质

| 性质 | 说明 |
|------|------|
| 正交性 | H_n · H_n^T = n · I_n |
| 对称性 | H_n = H_n^T |
| 快速算法 | O(n log n) 而非 O(n²) |
| 元素 | 仅包含 ±1 |

### 2.4 归一化

H~_n = (1/sqrt(n)) × H_n

归一化后：
H~_n · H~_n^T = I_n

## 3. _hadamard_transform_pytorch

### 3.1 函数定义

**位置**: `model.py:L429-L441`

```python
def _hadamard_transform_pytorch(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """纯 PyTorch Walsh-Hadamard 变换（无 CUDA 扩展）
    最后一维必须是 2 的幂。
    """
    n = x.size(-1)
    if n & (n - 1) != 0:
        raise ValueError(f"hadamard_transform requires last dim to be power of 2, got {n}")
    out = x.clone()
    bit = n
    for _ in range(int(math.log2(n))):
        bit >>= 1
        out = out.view(*out.shape[:-1], -1, 2, bit)
        a, b = out[..., 0, :], out[..., 1, :]
        out = torch.stack([a + b, a - b], dim=-2).flatten(-2)
    return out * scale
```

### 3.2 计算流程

```
输入 x (..., n)
      │
      ▼
n 是 2 的幂?
      │
      ├─── 否 ──► 报错
      │
      └─── 是 ──► 初始化 out = x.clone()
                    │
                    ▼
              log₂ n 次迭代
                    │
                    ▼
              每次迭代: bit >>= 1
                    │
                    ▼
              reshape: (..., -1, 2, bit)
                    │
                    ▼
              a = out[..., 0, :]
              b = out[..., 1, :]
                    │
                    ▼
              out = stack([a+b, a-b])
                    │
                    ▼
              flatten(-2)
                    │
                    ▼
              还有迭代?
                    │
         ┌────────┴────────┐
        是                  否
         │                  │
         ▼                  ▼
      继续迭代      返回 out × scale
```

### 3.3 逐步示例

以 n=8 为例：

**初始状态**：
```
out = [x0, x1, x2, x3, x4, x5, x6, x7]
```

**第 1 次迭代** (bit = 8 → 4):
```
reshape: [..., 2, 4]
a = [x0, x1, x2, x3], b = [x4, x5, x6, x7]
a+b = [x0+x4, x1+x5, x2+x6, x3+x7]
a-b = [x0-x4, x1-x5, x2-x6, x3-x7]
out = [x0+x4, x1+x5, x2+x6, x3+x7, x0-x4, x1-x5, x2-x6, x3-x7]
```

**第 2 次迭代** (bit = 4 → 2):
```
reshape: [..., 2, 2]
对每对元素应用 a+b, a-b
```

**第 3 次迭代** (bit = 2 → 1):
```
reshape: [..., 2, 1]
对每对元素应用 a+b, a-b
```

### 3.4 复杂度分析

| 指标 | 值 |
|------|-----|
| 时间复杂度 | O(n log n) |
| 空间复杂度 | O(n) (原地修改) |
| 迭代次数 | log_2 n |

## 4. rotate_activation

### 4.1 函数定义

**位置**: `model.py:L444-L457`

```python
def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    对 x 的最后一维应用（归一化）Hadamard 变换。
    用于 Indexer 中 act_quant 之前混合通道
    （如 QuaRot/SpinQuant 风格的量化）。
    等价于：x @ H.T * (hidden_size ** -0.5)，其中 H 是 Hadamard 矩阵。
    优先使用 fast_hadamard_transform（CUDA）（如果可用）。
    """
    hidden_size = x.size(-1)
    scale = hidden_size**-0.5  # 归一化因子：1/√n
    try:
        from fast_hadamard_transform import hadamard_transform
        return hadamard_transform(x, scale=scale)
    except ImportError:
        return _hadamard_transform_pytorch(x, scale=scale)
```

### 4.2 调用流程

```
调用 rotate_activation
      │
      ▼
fast_hadamard_transform 可用?
      │
      ├─── 是 ──► 使用 CUDA 扩展 (快速实现)
      │
      └─── 否 ──► 使用 PyTorch 实现 (_hadamard_transform_pytorch)
           │              │
           └──────┬───────┘
                  ▼
         输出 H @ x × d^(-1/2)
```

### 4.3 归一化因子

scale = d^(-0.5) = 1/sqrt(d)

对于 d=128 (Indexer head_dim):
scale = 1/sqrt(128) ≈ 0.088

## 5. 在 DSA Indexer 中的作用

### 5.1 使用位置

**位置**: `model.py:L498-L499`

```python
q = rotate_activation(q)
k = rotate_activation(k)
```

### 5.2 完整数据流

```
Indexer.forward()
    │
    ├───► Q 投影: q (B, S, H, D)
    │         │
    │         ▼
    │    Hadamard 变换 (rotate_activation)
    │         │
    │
    ├───► K 投影: k (B, S, D)
    │         │
    │         ▼
    │    Hadamard 变换 (rotate_activation)
    │         │
    └────┬───┘
         ▼
    FP8 量化 (act_quant)
         │
         ▼
    fp8_index (计算 index_score)
```

### 5.3 为什么需要 Hadamard 变换？

#### 5.3.1 量化前的预处理

Hadamard 变换在量化前混合通道，类似于：
1. **随机旋转** (Random Rotation)
2. **正交变换** (Orthogonal Transform)

#### 5.3.2 理论基础

对于量化误差：
Quant(Hx) = H · Quant(x) + error

由于 H 是正交矩阵：
||H · Quant(x)|| = ||Quant(x)||

Hadamard 变换**不改变** L2 范数，但**分散**了量化误差。

#### 5.3.3 实际效果

| 特性 | 无 Hadamard | 有 Hadamard |
|------|------------|-------------|
| 量化误差 | 集中在某些通道 | 分散到所有通道 |
| 后续计算 | 可能放大误差 | 误差更均匀 |
| 计算代价 | - | O(d log d) |

### 5.4 与 RoPE 的关系

```
Q/K 投影
      │
      ▼
RoPE (旋转位置编码)
      │
      ▼
切分 nope/pe 部分
      │
      ▼
Hadamard (混合通道)
      │
      ▼
FP8 量化
```

**执行顺序**：
1. 先应用 RoPE（位置编码）
2. 切分出 nope（非位置）部分
3. 对 nope 部分应用 Hadamard
4. 然后进行 FP8 量化

### 5.5 张量形状变化

| 阶段 | 形状 | 说明 |
|------|------|------|
| q 初始 | (B, S, H, D) | H=64, D=128 |
| q_pe (RoPE 后) | (B, S, H, D) | 含位置信息 |
| q_nope (切分后) | (B, S, H, D-64) | 去掉 RoPE 部分 |
| Hadamard 后 | (B, S, H, D-64) | 通道混合 |
| FP8 后 | (B, S, H, D-64) | FP8 格式 |

## 6. 性能考虑

### 6.1 CUDA 扩展 vs PyTorch 实现

| 实现 | 性能 | 要求 |
|------|------|------|
| `fast_hadamard_transform` | 快（CUDA kernel） | 需要安装扩展 |
| `_hadamard_transform_pytorch` | 慢（纯 PyTorch） | 无额外依赖 |

### 6.2 安装 CUDA 扩展

```bash
pip install --no-build-isolation git+https://github.com/Dao-AILab/fast-hadamard-transform.git
```

### 6.3 性能对比

假设 d=128：

| 实现 | 时间 (相对) |
|------|------------|
| CUDA 扩展 | 1x |
| PyTorch 实现 | ~10x |

---

**下一步**: 阅读 [MODEL_INDEXER.md](MODEL_INDEXER.md) 了解 DSA Indexer 模块的完整实现。
