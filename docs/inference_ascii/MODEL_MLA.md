# MODEL_MLA.md - MLA Attention 模块详解 (ASCII 版本)

## 目录

- [1. 概述](#1-概述)
- [2. MLA 架构原理](#2-mla-架构原理)
- [3. MLA 类定义](#3-mla-类定义)
- [4. forward 方法详解](#4-forward-方法详解)
- [5. Prefill vs Decode](#5-prefill-vs-decode)

## 1. 概述

**MLA (Multi-Head Latent Attention)** 是 DeepSeek-V3.2-Exp 的核心注意力机制，通过低秩分解大幅减少 KV Cache 显存占用。

```
标准 Attention (KV: S × d_k+d_v)
      │
      ▼
MLA (KV: S × d_latent)
      │
      ▼
显存节省 ~4x
```

## 2. MLA 架构原理

### 2.1 低秩分解

**标准 Attention**：
KV Cache = (K, V) ∈ R^(S × (d_k + d_v))

**MLA**：
K_latent ∈ R^(S × d_kv), K_compressed = K_latent × W_kv^b
V_compressed = K_latent × W_v^b

### 2.2 架构对比

```
标准 Attention                    MLA (本模型)
─────────────────                  ─────────────
输入 x                             输入 x
    │                                 │
    ▼                                 ▼
KV 投影                          KV_LoRA 投影
直接投影到 d_k+d_v               压缩到 d_latent=512
    │                                 │
    ▼                                 ▼
KV Cache                         KV Cache
大显存占用                        小显存占用
                                     │
                                     ▼
                               使用时解压
                               投影到 d_k+d_v
```

### 2.3 参数对比

| 参数 | 标准 Attention | MLA |
|------|--------------|-----|
| d_kv (latent) | - | 512 |
| d_k_nope | 128 | 128 |
| d_rope | 64 | 64 |
| d_v | 128 | 128 |

## 3. MLA 类定义

### 3.1 类结构

**位置**: `model.py:L549-L596`

```python
class MLA(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int = -1):
        super().__init__()
        self.layer_id = int(layer_id)
        # 维度配置
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # Q 路径
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        # KV 路径
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank,
                                           self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))

        # 输出
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)

        # Indexer
        self.indexer = Indexer(args)
        self.indexer._trace_layer_id = self.layer_id

        # Cache
        self.register_buffer("kv_cache", torch.zeros(...))
        self.register_buffer("pe_cache", torch.zeros(...))
```

### 3.2 参数形状表

| 参数 | 输入维度 | 输出维度 | 说明 |
|------|----------|----------|------|
| `wq_a` | d | q_rank | Q 第一阶段投影（q_rank=0） |
| `wq_b` | q_rank | n_h × d_k | Q 第二阶段投影 |
| `wkv_a` | d | d_kv + d_rope | KV 压缩投影 |
| `wkv_b` | d_kv | n_h × (d_nope + d_v) | KV 解压投影 |
| `wo` | n_h × d_v | d | 输出投影 |

### 3.3 Cache 配置

| Cache | 形状 | 说明 |
|-------|------|------|
| `kv_cache` | (B, S_max, 512) | KV latent cache |
| `pe_cache` | (B, S_max, 64) | RoPE 位置编码 cache |

## 4. forward 方法详解

**位置**: `model.py:L598-L661`

### 4.1 函数签名

```python
def forward(self, x: torch.Tensor, start_pos: int,
            freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
```

| 参数 | 形状 | 说明 |
|------|------|------|
| `x` | (B, S, d) | 输入隐藏状态 |
| `start_pos` | int | 当前起始位置 |
| `freqs_cis` | (S, d_rope/2) | RoPE 频率 |
| `mask` | (S, S) or None | Attention mask |

### 4.2 完整流程图

```
输入 x (B, S, d)
      │
      ▼
mask 存在? (Prefill)
      │
      ├─── 是 ──► Prefill 分支 (MHA dense)
      │              │
      │              ▼
      │         [Prefill 流程]
      │              │
      └─── 否 ──► Decode 分支 (MQA sparse)
                     │
                     ▼
                [Decode 流程]
                     │
                     ▼
              输出投影 (wo)
                     │
                     ▼
              输出 (B, S, d)
```

### 4.3 Q 路径（通用）

```python
# model.py:L613-L617
qr = self.q_norm(self.wq_a(x))
q = self.wq_b(qr)
q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
q_pe = apply_rotary_emb(q_pe, freqs_cis)
```

**Q 路径数据流**：

```
x: B,S,d
    │
    ▼
wq_a: q_lora_rank
    │
    ▼
q_norm: RMSNorm
    │
    ▼
wq_b: n_h × d_k
    │
    ▼
split: q_nope, q_pe
    │
    ▼
RoPE: q_pe 旋转
```

**形状变化**：

| 阶段 | 形状 | 说明 |
|------|------|------|
| `x` | (B, S, d) | 输入 |
| `qr` | (B, S, 0) | q_rank=0 |
| `q` | (B, S, n_h × d_k) | 投影后 |
| `q` view | (B, S, n_h, d_k) | 分 head |
| `q_nope` | (B, S, n_h, 128) | 非 RoPE 部分 |
| `q_pe` | (B, S, n_h, 64) | RoPE 部分 |

### 4.4 KV 路径（通用）

```python
# model.py:L618-L626
kv = self.wkv_a(x)
kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
kv = self.kv_norm(kv)
k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
kv_fp8, kv_scale = act_quant(kv, block_size, self.scale_fmt)
kv = (kv_fp8.view(-1, block_size).float() * kv_scale.view(-1, 1)).to(kv.dtype).view_as(kv)
self.kv_cache[:bsz, start_pos:end_pos] = kv
self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
```

**形状变化**：

| 阶段 | 形状 | 说明 |
|------|------|------|
| `kv` | (B, S, 576) | d_kv + d_rope = 512 + 64 |
| `kv` (latent) | (B, S, 512) | 压缩的 KV |
| `k_pe` | (B, 1, S, 64) | 位置编码 |
| `kv_fp8` | (B, S, 512) | FP8 格式 |
| `kv_cache` | (B, S_max, 512) | 累积 cache |

## 5. Prefill vs Decode

### 5.1 Prefill 分支 (mask exists)

```python
# model.py:L627-L642
if mask is not None:  # MHA prefill
    q = torch.cat([q_nope, q_pe], dim=-1)
    kv = self.wkv_b(kv)
    kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
    scores = torch.einsum("bshd,bthd->bsht", q, k).mul_(self.softmax_scale)

    # indexer
    topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
    index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device)
    index_mask = index_mask.scatter_(-1, topk_indices, 0)
    index_mask += mask
    scores += index_mask.unsqueeze(2)

    scores = scores.softmax(dim=-1)
    x = torch.einsum("bsht,bthd->bshd", scores, v)
```

**Prefill 数据流**：

```
q_nope, q_pe → 拼接 Q
kv_cache → wkv_b 解压
         → split: k_nope, v
pe_cache → 拼接 K
    Q, K → einsum: scores
          │
          ▼
    Indexer → Top-K mask
          │
          ▼
    scores += mask
          │
          ▼
    softmax
          │
          ▼
    @ V → 输出
```

### 5.2 Decode 分支 (mask is None)

```python
# model.py:L643-L659
else:  # MQA decode
    q_nope = torch.einsum("bshd,bshd->bsh", q_nope, kv_cache[:, :end_pos].unsqueeze(2))
    q_pe_scores = torch.einsum("bshd,bthd->bsht", q_pe,
                                 pe_cache[:, :end_pos].unsqueeze(2))
    q_pe_scores *= self.softmax_scale

    # indexer
    topk_indices = self.indexer(x, qr, start_pos, freqs_cis, None)
    index_mask = torch.full((bsz, seqlen, end_pos), float("-inf"), device=x.device)
    index_mask = index_mask.scatter_(-1, topk_indices, 0)
    q_pe_scores += index_mask

    scores = q_pe_scores.softmax(dim=-1)
    kv = self.wkv_b(kv_cache[:, :end_pos])
    v = kv[:, :, self.qk_nope_head_dim:]
    x = torch.einsum("bshd,bthd->bshd", scores, v)
```

**Decode 数据流**：

```
q_nope → einsum(kv_cache): q_nope_scores
q_pe → einsum(pe_cache): q_pe_scores
                        │
                        ▼
                  Indexer → Top-K mask
                        │
                        ▼
                  scores += mask
                        │
                        ▼
                  softmax
                        │
                        ▼
                  @ V → 输出
```

### 5.3 输出投影

```python
# model.py:L660-L661
x = self.wo(x.flatten(2))
return x
```

## 6. 完整数据流总结

```
输入 x (B, S, d)
      │
      ├─────┬─────┐
      │     │     │
      ▼     ▼     ▼
    wq_a  wkv_a  (准备 Indexer)
      │     │
      ▼     ▼
   q_norm kv_norm
      │     │
      ▼     ▼
    wq_b  (split)
      │     │
      ▼     ▼
   view  k_pe
      │     │
      ▼     ▼
  split  RoPE
      │     │
      ▼     ▼
q_nope, q_pe  │
      │     │
      ▼     ▼
   RoPE  (量化+缓存)
      │
      ▼
┌─────────────────────┐
│ mask 存在?          │
├─────────────────────┤
│ 是: Prefill (MHA)   │
│ 否: Decode (MQA)    │
└─────────────────────┘
      │
      ▼
    wo 投影
      │
      ▼
输出 (B, S, d)
```

---

**下一步**: 阅读 [MODEL_MOE.md](MODEL_MOE.md) 了解混合专家系统的实现。
