# Hadamard Transform 实现分析

## 1. 概述

`inference/model.py` 中的 `rotate_activation()` 函数在 **Indexer** 模块的 FP8 量化前
对 Q/K 向量施加归一化 Hadamard 变换（类似 QuaRot / SpinQuant 的通道混合），其作用是
把各通道的动态范围"均匀化"，让后续 per-block FP8 量化误差更小。

本文档分析当前实现与上游
[`fast-hadamard-transform`](https://github.com/Dao-AILab/fast-hadamard-transform)
库的关系，以及纯 PyTorch 回退路径的正确性。

---

## 2. 上游库（Tri Dao）接口一览

上游 `fast_hadamard_transform_interface.py` 提供：

| 函数 | 支持维度 | 说明 |
|---|---|---|
| `hadamard_transform(x, scale)` | dim = 2^k | 标准 Walsh-Hadamard（Sylvester 构造） |
| `hadamard_transform_12N(x, scale)` | dim = 12 × 2^k | 12N 变体 |
| `hadamard_transform_20N(x, scale)` | dim = 20 × 2^k | 20N 变体 |
| `hadamard_transform_28N(x, scale)` | dim = 28 × 2^k | 28N 变体 |
| `hadamard_transform_40N(x, scale)` | dim = 40 × 2^k | 40N 变体 |
| `hadamard_transform_ref(x, scale)` | 任意 dim（scipy） | 参考实现，用于测试 |

每个函数都被包在 `torch.autograd.Function` 中（提供 backward），底层调用
`fast_hadamard_transform_cuda` CUDA 扩展。

---

## 3. 当前 model.py 的实现

```python
def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    scale = hidden_size ** -0.5
    try:
        from fast_hadamard_transform import hadamard_transform
        return hadamard_transform(x, scale=scale)
    except ImportError:
        return _hadamard_transform_pytorch(x, scale=scale)
```

**调用链：**

```
rotate_activation
  ├─ (CUDA 路径) from fast_hadamard_transform import hadamard_transform
  │     → HadamardTransformFn.apply(x, scale)
  │       → fast_hadamard_transform_cuda.fast_hadamard_transform(x, scale)
  │
  └─ (CPU/无扩展 回退) _hadamard_transform_pytorch(x, scale)
        → 纯 PyTorch butterfly 算法
```

**关键点：`from fast_hadamard_transform import hadamard_transform` 导入的就是
上游接口文件中定义的 `hadamard_transform()` 函数**（它内部已经包含了
`HadamardTransformFn` autograd 封装 + CUDA kernel 调用）。所以当前代码
**已经在使用上游库的官方 API**。

---

## 4. 为什么不需要直接使用上游完整接口

### 4.1 仅推理，无需 autograd

本仓库是纯推理路径（`torch.no_grad()`），不需要 backward。上游的
`HadamardTransformFn(torch.autograd.Function)` 封装虽然在 `no_grad` 模式下
开销几乎为零，但我们只需要其 forward 计算，**当前 import 方式完全满足需求**。

### 4.2 维度始终为 2 的幂

`rotate_activation` 唯一的调用点在 `Indexer.forward()`（`model.py:498-499`）：

```python
q = rotate_activation(q)   # q.shape[-1] = index_head_dim = 128
k = rotate_activation(k)   # k.shape[-1] = index_head_dim = 128
```

`index_head_dim = 128 = 2^7`，始终是 2 的幂次方。因此：

- **不需要** `hadamard_transform_12N / 20N / 28N / 40N` 等变体
- **不需要** 隐式 zero-padding 功能
- 标准 `hadamard_transform()` 完全覆盖

### 4.3 优雅的回退机制

`try/except ImportError` 确保在没有安装 CUDA 扩展（如 CPU-only 环境、CI 测试）时
自动降级到纯 PyTorch 实现，这是上游库本身不提供的能力（上游在顶层直接
`import fast_hadamard_transform_cuda`，缺失时直接报错）。

---

## 5. 数学等价性证明

### 5.1 CUDA 路径

上游文档注明：

> Equivalent to `F.linear(x, torch.tensor(scipy.linalg.hadamard(dim))) * scale`

即计算 $H_n \cdot x \cdot \text{scale}$，其中 $H_n$ 是 $n$ 阶 Sylvester
构造的 Hadamard 矩阵。

### 5.2 PyTorch 回退路径（butterfly 算法）

```python
def _hadamard_transform_pytorch(x, scale=1.0):
    n = x.size(-1)
    out = x.clone()
    bit = n
    for _ in range(int(math.log2(n))):
        bit >>= 1
        out = out.view(*out.shape[:-1], -1, 2, bit)
        a, b = out[..., 0, :], out[..., 1, :]
        out = torch.stack([a + b, a - b], dim=-2).flatten(-2)
    return out * scale
```

这是 Walsh-Hadamard Transform 的标准 **in-place butterfly** 分解：

$$H_{2n} = \begin{pmatrix} H_n & H_n \\ H_n & -H_n \end{pmatrix}$$

每轮迭代在当前 stride (`bit`) 下执行 $(a, b) \mapsto (a+b,\; a-b)$，
经 $\log_2 n$ 轮后等价于 $H_n \cdot x$。

### 5.3 手动验证 (n=4)

```
H_4 = [[1,  1,  1,  1],
       [1, -1,  1, -1],
       [1,  1, -1, -1],
       [1, -1, -1,  1]]
```

对 $x = [x_0, x_1, x_2, x_3]$：

- 第 1 轮 (bit=2)：$(x_0+x_2,\; x_1+x_3,\; x_0-x_2,\; x_1-x_3)$
- 第 2 轮 (bit=1)：$(x_0+x_1+x_2+x_3,\; x_0-x_1+x_2-x_3,\; x_0+x_1-x_2-x_3,\; x_0-x_1-x_2+x_3)$

与 $H_4 \cdot x$ 结果一致。

### 5.4 scale 因子

`rotate_activation` 使用 `scale = hidden_size ** -0.5 = 128 ** -0.5`。
这使得变换矩阵变为**正交矩阵**（$\frac{1}{\sqrt{n}} H_n$），
保持向量的 L2 范数不变，保证量化前后数值稳定性。

---

## 6. 结论

| 问题 | 回答 |
|---|---|
| 当前实现是否正确？ | **正确**。CUDA 路径直接使用上游 `hadamard_transform()`，PyTorch 回退路径数学等价。 |
| 为什么不用上游的完整接口？ | 维度固定为 128（2^7），无需 12N/20N/28N/40N 变体；纯推理无需 autograd 类。 |
| 为什么上游不能直接做回退？ | 上游在模块顶层 `import fast_hadamard_transform_cuda`，缺失时直接失败；我们的 lazy import + fallback 更健壮。 |
| scale 为什么是 `dim ** -0.5`？ | 使变换矩阵正交化，保持 L2 范数不变，利于 FP8 量化数值稳定。 |
| 会影响模型精度吗？ | 不会。两条路径在浮点精度内完全一致（CUDA 路径在 GPU 上更快）。 |

---

## 7. 如需修改的场景

如果未来 `index_head_dim` 变为非 2 的幂（例如 192 = 12 × 16），则需要：

1. CUDA 路径改用 `hadamard_transform_12N`（或相应变体）
2. PyTorch 回退路径添加 zero-padding 或切换到 `hadamard_transform_ref` 逻辑

当前配置下（`index_head_dim = 128`）无需任何更改。
