# MODEL_ROPE.md - 旋转位置编码详解 (ASCII 版本)

## 目录

- [1. 概述](#1-概述)
- [2. RoPE 基础原理](#2-rope-基础原理)
- [3. YaRN 扩展机制](#3-yarn-扩展机制)
- [4. precompute_freqs_cis](#4-precompute_freqs_cis)
- [5. apply_rotary_emb](#5-apply_rotary_emb)

## 1. 概述

DeepSeek-V3.2-Exp 使用 **YaRN (Yet another RoPE extensioN)** 扩展 RoPE 以支持长上下文：

```
RoPE (原始 4K 长度)
      │
      ▼
YaRN 扩展 (factor=40)
      │
      ▼
支持 160K+ 长度
```

## 2. RoPE 基础原理

### 2.1 核心思想

RoPE (Rotary Position Embedding) 通过**旋转变换**将位置信息注入 Query 和 Key。

### 2.2 数学公式

对于位置 m 的向量 x_m ∈ R^d，将其分为 d/2 对：

x_m = (x_{m,1}, x_{m,2}, ..., x_{m,d-1}, x_{m,d})

形成 d/2 个复数：
z_m^(i) = x_{m,2i-1} + j · x_{m,2i}, i = 1, ..., d/2

旋转角度：
Θ_i = 10000^(-2(i-1)/d)

旋转变换：
RoPE(z_m^(i)) = z_m^(i) · e^(j · m · Θ_i)

### 2.3 矩阵形式

[x'_m,2i-1]   [cos(m·Θ_i)  -sin(m·Θ_i)] [x_m,2i-1]
[x'_m,2i]   = [sin(m·Θ_i)   cos(m·Θ_i)] [x_m,2i]

## 3. YaRN 扩展机制

### 3.1 问题

标准 RoPE 训练在固定长度（如 4096）上，直接扩展到更长序列性能下降。

### 3.2 YaRN 解决方案

```
计算频率 (freqs = 1/θ^2i/d)
      │
      ▼
seq_len > original?
      │
      ├─── 否 ──► 使用原始频率 (freqs)
      │
      └─── 是 ──► 计算修正范围 (find_correction_range)
                    │
                    ▼
              计算平滑因子 (linear_ramp_factor)
                    │
                    ▼
              混合频率 (freqs / factor × 1-smooth + freqs × smooth)
                    │
                    ▼
              扩展频率 (freqs_extended)
```

### 3.3 修正维度计算

**位置**: `model.py:L342-L345`

```python
def find_correction_dim(num_rotations, dim, base, max_seq_len):
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
```

**公式**：
d_corr = d × ln(L / (2πr)) / (2 × ln(θ))

其中：
- L 是最大序列长度
- r 是旋转次数
- θ 是基频 (10000)

### 3.4 线性平滑因子

**位置**: `model.py:L375-L392`

```python
def linear_ramp_factor(min, max, dim):
    if min == max:
        max += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func
```

**公式**：
ramp(i) = clamp((i - min) / (max - min), 0, 1)

**可视化**：

```
ramp(i)
  1 |           _______
    |          /
    |         /
    |        /
  0 |_______/
    +------------------> i
     min          max
```

## 4. precompute_freqs_cis

### 4.1 函数签名

**位置**: `model.py:L325-L403`

```python
def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """预计算旋转位置的复数指数值"""
    dim = args.qk_rope_head_dim      # RoPE 维度: 64
    seqlen = args.max_seq_len         # 最大序列长度: 16384
    beta_fast = args.beta_fast        # 32
    beta_slow = args.beta_slow        # 1
    base = args.rope_theta            # 10000.0
    factor = args.rope_factor         # 40
```

### 4.2 计算流程

```
计算基础频率 (freqs = 1/base^2i/d)
      │
      ▼
seqlen > original?
      │
      ├─── 否 ──► 直接使用 freqs
      │
      └─── 是 ──► 计算修正范围 (low, high)
                    │
                    ▼
              计算平滑因子 (smooth)
                    │
                    ▼
              频率插值 (混合扩展频率)
                    │
                    ▼
              扩展后频率
                    │
                    ▼
              外积位置 (freqs × t)
                    │
                    ▼
              构造复数 (e^j×freqs)
                    │
                    ▼
              输出 (seqlen, d/2) 复数
```

### 4.3 代码详解

#### 4.3.1 基础频率计算

```python
# model.py:L394
freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
```

**计算**：
Θ_i = 1 / (10000^(2i/d)) = 10000^(-2i/d)

**示例** (d=64):
| i | Θ_i |
|---|-----|
| 0 | 10000^0 = 1.0 |
| 1 | 10000^(-1/32) ≈ 0.79 |
| 2 | 10000^(-2/32) ≈ 0.63 |
| 31 | 10000^(-31/32) ≈ 0.01 |

#### 4.3.2 YaRN 扩展

```python
# model.py:L395-L398
if seqlen > args.original_seq_len:
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
    smooth = 1 - linear_ramp_factor(low, high, dim // 2)
    freqs = freqs / factor * (1 - smooth) + freqs * smooth
```

**混合公式**：
freqs' = freqs_extended × (1 - s) + freqs_original × s

其中：
- freqs_extended = freqs / factor
- s 是平滑因子

**效果**：
- 低频维度（大 i）：使用扩展频率
- 高频维度（小 i）：保留原始频率
- 中间维度：平滑过渡

#### 4.3.3 位置编码

```python
# model.py:L400-L402
t = torch.arange(seqlen)
freqs = torch.outer(t, freqs)
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
```

**计算**：
1. t = [0, 1, 2, ..., L-1]
2. freqs[t, i] = t × Θ_i
3. freqs_cis[t, i] = e^(j · t · Θ_i)

### 4.4 输出形状

| 变量 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| freqs | (L, d/2) | float32 | 角度频率 |
| freqs_cis | (L, d/2) | complex64 | 复数指数 |

**示例**：L=16384, d=64 → (16384, 32) 复数张量

## 5. apply_rotary_emb

### 5.1 函数签名

**位置**: `model.py:L406-L426`

```python
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
```

### 5.2 参数说明

| 参数 | 形状 | 说明 |
|------|------|------|
| `x` | (B, S, H, D) 或 (B, S, H × D) | 输入张量 |
| `freqs_cis` | (S, D/2) | 预计算的复数指数 |
| `interleaved` | bool | 是否交错布局 |

### 5.3 交错 vs 非交错布局

**交错 (interleaved=True)**:
[x₀, x₁, x₂, x₃, ...] → [(x₀,x₁), (x₂,x₃), ...]
实部和虚部相邻

**非交错 (interleaved=False)**:
[x₀, x₁, x₂, x₃, ...] → [x₀, x₂, ... | x₁, x₃, ...]
所有实部在前，所有虚部在后

### 5.4 计算流程

```
输入 x (B, S, H, D)
      │
      ▼
interleaved?
      │
      ├─── False ──► 转置为非交错 (B, S, H, D/2, 2)
      │                    │
      └─── True ───► 保持交错
           │              │
           └──────┬───────┘
                  ▼
         reshape 为复数 (B, S, H×D/2)
                  │
                  ▼
         freqs_cis 广播 (1, S, 1, D/2)
                  │
                  ▼
         复数乘法 (z × e^jθ)
                  │
                  ▼
         转回实数 (B, S, H, D)
                  │
                  ▼
         interleaved?
                  │
         ├─── False ──► 拼接实部虚部
         │                    │
         └─── True ───► 保持交错
              │              │
              └──────┬───────┘
                     ▼
                  输出
```

### 5.5 代码详解

#### 5.5.1 非交错模式转换

```python
# model.py:L419-L420
if not interleaved:
    x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
```

**形状变化**：
(B, S, H, D) → (B, S, H, 2, D/2) → (B, S, H, D/2, 2)

#### 5.5.2 复数乘法

```python
# model.py:L421-L423
x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
y = torch.view_as_real(x * freqs_cis).flatten(3)
```

**复数乘法**：
(a + jb) × (cosθ + jsinθ) = (a·cosθ - b·sinθ) + j(a·sinθ + b·cosθ)

**等价于**：
[a']   [cosθ  -sinθ] [a]
[b'] = [sinθ   cosθ] [b]

### 5.6 在模型中的使用

#### 5.6.1 MLA Attention

```python
# model.py:L617
q_pe = apply_rotary_emb(q_pe, freqs_cis)  # 交错模式
```

#### 5.6.2 Indexer

```python
# model.py:L490, L496
q_pe = apply_rotary_emb(q_pe, freqs_cis, False)  # 非交错模式
k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
```

**注意**：Indexer 使用**非交错模式**（这是一个重要的实现细节）。

## 6. 多尺度缩放 (mscale)

### 6.1 长度扩展时的缩放

**位置**: `model.py:L587-L589`

```python
if args.max_seq_len > args.original_seq_len:
    mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
    self.softmax_scale = self.softmax_scale * mscale * mscale
```

### 6.2 缩放公式

mscale = 0.1 × mscale × ln(factor) + 1.0

**最终 softmax scale**：
scale = (1 / sqrt(d_k)) × mscale²

### 6.3 数值示例

| factor | ln(factor) | mscale | mscale² |
|--------|------------|--------|---------|
| 1 | 0 | 1.0 | 1.0 |
| 10 | 2.3 | 1.23 | 1.51 |
| 40 | 3.7 | 1.37 | 1.87 |

---

**下一步**: 阅读 [MODEL_HADAMARD.md](MODEL_HADAMARD.md) 了解 Hadamard 变换的实现。
