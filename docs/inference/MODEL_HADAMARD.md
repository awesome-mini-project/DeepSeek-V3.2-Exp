# MODEL_HADAMARD.md - Hadamard 变换详解

## 目录

- [1. 概述](#1-概述)
- [2. Hadamard 变换基础](#2-hadamard-变换基础)
- [3. _hadamard_transform_pytorch](#3-_hadamard_transform_pytorch)
- [4. rotate_activation](#4-rotate_activation)
- [5. 在 DSA Indexer 中的作用](#5-在-dsa-indexer-中的作用)

## 1. 概述

DeepSeek-V3.2-Exp 在 DSA Indexer 中使用 **归一化 Hadamard 变换**来混合通道，用于 QuaRot/SpinQuant 风格的量化。

```mermaid
flowchart LR
    A[激活 x] --> B[Hadamard 变换<br/>H @ x]
    B --> C[归一化<br/>× d^(-1/2)]
    C --> D[FP8 量化]
    D --> E[Indexer 计算]

    style B fill:#e1f5ff
    style D fill:#fff3e0
```

## 2. Hadamard 变换基础

### 2.1 定义

Hadamard 矩阵 $H_n$ 是一个 $n \times n$ 的矩阵（$n$ 是 2 的幂）：

$$ H_1 = \begin{pmatrix} 1 \end{pmatrix}, \quad H_{2n} = \begin{pmatrix} H_n & H_n \\ H_n & -H_n \end{pmatrix} $$

### 2.2 小规模示例

$$ H_2 = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} $$

$$ H_4 = \begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & -1 & 1 & -1 \\ 1 & 1 & -1 & -1 \\ 1 & -1 & -1 & 1 \end{pmatrix} $$

### 2.3 性质

| 性质 | 说明 |
|------|------|
| 正交性 | $H_n H_n^T = n I_n$ |
| 对称性 | $H_n = H_n^T$ |
| 快速算法 | $O(n \log n)$ 而非 $O(n^2)$ |
| 元素 | 仅包含 $\pm 1$ |

### 2.4 归一化

$$ \tilde{H}_n = \frac{1}{\sqrt{n}} H_n $$

归一化后：
$$ \tilde{H}_n \tilde{H}_n^T = I_n $$

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

```mermaid
flowchart TD
    A[输入 x<br/>(..., n)] --> B[n 是 2 的幂?]
    B -->|否| C[报错]
    B -->|是| D[初始化 out = x.clone]

    D --> E[log₂ n 次迭代]
    E --> F[每次迭代:<br/>bit >>= 1]
    F --> G[reshape:<br/>(..., -1, 2, bit)]
    G --> H[a = out[..., 0, :]<br/>b = out[..., 1, :]]
    H --> I[out = stack[a+b, a-b]]
    I --> J[flatten(-2)]
    J --> K{还有迭代?}
    K -->|是| F
    K -->|否| L[返回 out × scale]

    style G fill:#e1f5ff
    style I fill:#e8f5e9
```

### 3.3 逐步示例

以 $n=8$ 为例：

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
| 时间复杂度 | $O(n \log n)$ |
| 空间复杂度 | $O(n)$ (原地修改) |
| 迭代次数 | $\log_2 n$ |

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

```mermaid
flowchart TD
    A[调用 rotate_activation] --> B{fast_hadamard_transform<br/>可用?}
    B -->|是| C[使用 CUDA 扩展<br/>快速实现]
    B -->|否| D[使用 PyTorch 实现<br/>_hadamard_transform_pytorch]
    C --> E[输出 H @ x × d^(-1/2)]
    D --> E

    style C fill:#c8e6c9
    style D fill:#ffccbc
```

### 4.3 归一化因子

$$ \text{scale} = d^{-0.5} = \frac{1}{\sqrt{d}} $$

对于 $d=128$ (Indexer head_dim):
$$ \text{scale} = \frac{1}{\sqrt{128}} \approx 0.088 $$

## 5. 在 DSA Indexer 中的作用

### 5.1 使用位置

**位置**: `model.py:L498-L499`

```python
q = rotate_activation(q)
k = rotate_activation(k)
```

### 5.2 完整数据流

```mermaid
flowchart TD
    subgraph Indexer ["Indexer.forward()"]
        A[Q 投影<br/>q: (B, S, H, D)] --> B[Hadamard 变换<br/>rotate_activation]
        C[K 投影<br/>k: (B, S, D)] --> D[Hadamard 变换<br/>rotate_activation]

        B --> E[FP8 量化<br/>act_quant]
        D --> E

        E --> F[fp8_index<br/>计算 index_score]
    end

    style B fill:#e1f5ff
    style D fill:#e1f5ff
    style E fill:#fff3e0
```

### 5.3 为什么需要 Hadamard 变换？

#### 5.3.1 量化前的预处理

Hadamard 变换在量化前混合通道，类似于：

1. **随机旋转** (Random Rotation)
2. **正交变换** (Orthogonal Transform)

#### 5.3.2 理论基础

对于量化误差：
$$ \text{Quant}(Hx) = H \cdot \text{Quant}(x) + \text{error} $$

由于 $H$ 是正交矩阵：
$$ \|H \cdot \text{Quant}(x)\| = \|\text{Quant}(x)\| $$

Hadamard 变换**不改变** L2 范数，但**分散**了量化误差。

#### 5.3.3 实际效果

| 特性 | 无 Hadamard | 有 Hadamard |
|------|------------|-------------|
| 量化误差 | 集中在某些通道 | 分散到所有通道 |
| 后续计算 | 可能放大误差 | 误差更均匀 |
| 计算代价 | - | $O(d \log d)$ |

### 5.4 与 RoPE 的关系

```mermaid
flowchart LR
    A[Q/K 投影] --> B[RoPE<br/>旋转位置编码]
    B --> C[切分 nope/pe 部分]
    C --> D[Hadamard<br/>混合通道]
    D --> E[FP8 量化]

    style B fill:#fff3e0
    style D fill:#e1f5ff
```

**执行顺序**：
1. 先应用 RoPE（位置编码）
2. 切分出 nope（非位置）部分
3. 对 nope 部分应用 Hadamard
4. 然后进行 FP8 量化

### 5.5 张量形状变化

| 阶段 | 形状 | 说明 |
|------|------|------|
| q 初始 | $(B, S, H, D)$ | $H=64, D=128$ |
| q_pe (RoPE 后) | $(B, S, H, D)$ | 含位置信息 |
| q_nope (切分后) | $(B, S, H, D-64)$ | 去掉 RoPE 部分 |
| Hadamard 后 | $(B, S, H, D-64)$ | 通道混合 |
| FP8 后 | $(B, S, H, D-64)$ | FP8 格式 |

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

假设 $d=128$：

| 实现 | 时间 (相对) |
|------|------------|
| CUDA 扩展 | 1x |
| PyTorch 实现 | ~10x |

---

**下一步**: 阅读 [MODEL_INDEXER.md](MODEL_INDEXER.md) 了解 DSA Indexer 模块的完整实现。
