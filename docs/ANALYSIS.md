# DSA Trace 分析脚本文档

本文档说明 `analysis/` 目录下分析脚本的**设计思路、输出 schema、每个指标的含义**，以及**面向 KV cache 放置策略研究的使用指南**。

---

## 目录

1. [为什么需要这些脚本](#1-为什么需要这些脚本)
2. [文件说明](#2-文件说明)
3. [运行前提](#3-运行前提)
4. [快速开始](#4-快速开始)
5. [CLI 参数详解](#5-cli-参数详解)
6. [设计说明](#6-设计说明)
7. [输出 JSON schema 逐字段详解](#7-输出-json-schema-逐字段详解)
8. [指标含义与 KV 放置策略的关联](#8-指标含义与-kv-放置策略的关联)
9. [典型使用场景示例](#9-典型使用场景示例)
10. [性能与注意事项](#10-性能与注意事项)

---

## 1. 为什么需要这些脚本

`inference/trace.py` 产生的 JSONL 记录（`trace_steps_*.jsonl`）按 **每 decode step × 每 request × 每 layer** 记录了 DSA indexer 选出的 top-2048 token 绝对位置（`selected_token_pos`）以及一些预计算统计（`unique_blocks`、`touched_block_ratio` 等）。

这些字段是**基于 trace 时指定的 block size** 预计算的，但：

- 如果你想**用不同的 block size 重新计算**（例如把 64-token block 换成 16-token 或 128-token，对应不同 serving 引擎或不同"放置单元"），就需要重新做 `token_pos // block_size`。
- trace 文件里**不存** `selected_block_ids`（为了减小文件体积），所以所有 block 级指标都需要从 `selected_token_pos` 在线推导。
- 更重要的是，JSONL 里**每条记录是一个 layer 的一次 step 的结果**，而很多放置策略关心的是 **"一整个 decode step 里跨 61 层合并的工作集"**，需要聚合计算。

`analysis/analyze_trace.py` 就是这个"离线可重算、任意 block size、多种放置策略相关指标"的分析工具。

---

## 2. 文件说明

```
analysis/
├── analyze_trace.py   主分析脚本（直接运行）
├── trace_utils.py     辅助库：JSONL 流式读取、block_id 计算、目录查找
└── README.md          简短指引（指向本文档）
```

### `trace_utils.py`

不依赖任何第三方库（纯 stdlib），提供：

| 函数/类 | 说明 |
|---|---|
| `find_trace_dirs(root)` | 递归找所有含 `trace_steps_*.jsonl` 的目录，支持直接指向 `block64/` 或上层 `outputs/<run>/` |
| `load_run_meta(trace_dir)` | 读取 `run_meta.json`，返回 `RunMeta` dataclass（含 `block_size_tokens`、`dataset`、`rank` 等） |
| `iter_jsonl_records(trace_dir)` | **流式**迭代目录下所有 `trace_steps_*.jsonl`（不加载全文到内存），按文件名排序 |
| `compute_block_ids(token_positions, block_size_tokens)` | `[pos // block_size for pos in token_positions]`（含重复，用于 per-block token count 统计） |
| `jaccard(a, b)` | Jaccard 相似度 `|a ∩ b| / |a ∪ b|`，用于相邻 step 的 block set 相似度 |

### `analyze_trace.py`

单文件主程序，包含：

- `analyze(trace_dir, ...)` — 核心分析函数，单次 pass 流式处理 JSONL，一次性计算所有指标
- `main()` — CLI 入口，支持多 trace 目录、多 block size 批量分析

---

## 3. 运行前提

### 必要条件：trace 记录中有 `selected_token_pos`

分析脚本需要 **`selected_token_pos` 字段**（记录 DSA 选出的 top-2048 token 绝对位置）。

- 默认启用（`TraceConfig.store_selected_token_pos=True`），除非你在采集时设了 `DS_TRACE_STORE_TOKEN_POS=0`。
- 如果你的 trace 没有这个字段，所有 block 级指标都无法计算，分析脚本会跳过这些记录（输出全 0）。

### 验证 trace 是否有效

```bash
OUT=outputs/<run>/block64
# 取一条记录，确认 selected_token_pos 字段存在
head -1 "$OUT"/trace_steps_*.jsonl | python3 -m json.tool | grep selected_token_pos
```

### 依赖

```
# 无第三方依赖，只需 Python >= 3.8（使用 walrus operator 和 collections.deque 等标准库）
python3 --version
```

---

## 4. 快速开始

### 场景 1：分析某一次 run，使用 trace 时的 block size

```bash
# 脚本和 analysis.json 都在 analysis/ 目录下执行；
# 由于 trace_utils.py 在同目录，直接运行即可。
python3 analysis/analyze_trace.py \
  --input outputs/ruler_1234567890/block64
# 输出：outputs/ruler_1234567890/block64/analysis.json
```

### 场景 2：what-if 分析：同一份 token 访问，用多种 block size

```bash
# 例如对比 16/32/64/128-token block 下的 unique_blocks、working_set 变化
python3 analysis/analyze_trace.py \
  --input outputs/ruler_1234567890/block64 \
  --block-sizes 16,32,64,128 \
  --out outputs/ruler_1234567890/analysis_multi_bs.json
```

### 场景 3：跨 layer 合并（step-level union，适合容量估算）

按 `(request_id, step_idx)` 把 61 层的 block 访问集合合并为一个虚拟 record：

```bash
python3 analysis/analyze_trace.py \
  --input outputs/ruler_1234567890/block64 \
  --step-union \
  --key-mode request
```

### 场景 4：批量分析整个 outputs 目录

```bash
python3 analysis/analyze_trace.py \
  --input outputs/ \
  --block-sizes 64 \
  --out outputs/all_analysis.json
```

---

## 5. CLI 参数详解

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input PATH` | **必填** | trace 根目录。可以是：(1) `outputs/<run>/block64/`（直接指向含 `trace_steps_*.jsonl` 的目录），(2) `outputs/<run>/`（自动找 `block*/` 子目录），(3) `outputs/`（批量处理所有 run） |
| `--out PATH` | `<first_trace_dir>/analysis.json` | 输出 JSON 文件路径。如果指定了多个 trace 目录或多个 block size，结果以数组 `{"results": [...]}` 写到同一文件 |
| `--block-size N` | `0`（自动） | 用指定的 block size（token 数）重新计算 block id。**0 表示从 `run_meta.json` 读取**（推荐）。不能与 `--block-sizes` 同时用 |
| `--block-sizes A,B,C` | `""` | 逗号分隔的多个 block size，批量分析，结果中每个 block size 一条记录。示例：`16,32,64,128` |
| `--key-mode` | `request_layer` | 流（stream）的粒度：`request_layer` 表示每个 `(request_id, layer_id)` 独立跟踪；`request` 把同一 request 的所有 layer 归并到一个 stream |
| `--ws-windows A,B,C` | `1,4,16,64` | 工作集计算的时间窗口（单位：step 数）。多个值同时算，互不干扰 |
| `--recent-token-windows A,B,C` | `256,1024,4096` | 计算"近邻 token 比例"的窗口（单位：token）。例如 `256` 表示"选中的 token 有多少落在最近 256 个 token 内" |
| `--recent-block-windows A,B,C` | `4,16,64` | 计算"近邻 block 比例"的窗口（单位：block 数）。例如 `16` 表示"触达的 block 有多少在距当前尾部 ≤16 块内" |
| `--reuse-cap-steps N` | `2048` | 重用距离直方图的上限（超过 N steps 的重用仍计入，但 bucket 限定在 N） |
| `--lru-caps A,B,C` | `128,256,512,1024` | LRU 模拟器的容量（单位：block 数），每个容量独立模拟。示例：`256,512,1024,2048` |
| `--step-union` | `False` | 按 `(request_id, step_idx)` 把所有层的 block 访问集合求并集，然后当作一条虚拟 record 分析。适合估算"整个 decode step 的总 KV 工作集" |
| `--max-steps-per-stream N` | `0`（无限） | 每个 stream 最多处理 N 个 step 后停止（用于在大规模 trace 上快速采样验证）。0 表示处理全部 |

---

## 6. 设计说明

### 6.1 单 pass 流式处理

`analyze()` 函数**只扫描一遍 JSONL**，所有指标在同一个 pass 里累积。这样：
- 内存开销恒定（不把整个文件加载到 Python list）
- 对特别大的 trace（几十 GB）也可以运行

唯一例外是 `step-union` 模式需要 **维护 pending 缓冲**（当 `(request_id, step_idx)` 发生变化时才 emit），但也只多占一个 step 的数据量。

### 6.2 什么是 "stream"（key_mode）

一个 "stream" 是一组**时序连续、有因果关系**的 decode step 序列。

- `key_mode=request_layer`（默认）：每个 `(request_id, layer_id)` 是一个独立 stream。共有 `N_requests × 61 layers` 个 stream。时间局部性（Jaccard、重用距离、工作集、LRU）都在"同一 layer 的连续 step"上计算。
  - **适合**：分析"某一层随 decode 推进的访问模式"
- `key_mode=request`：忽略 layer_id，把一个 request 的所有记录视为一个 stream，step 之间的顺序是 "layer 0 step 1, layer 1 step 1, ..., layer 60 step 1, layer 0 step 2, ..." 的混排。
  - **适合**：结合 `--step-union` 做 request 级工作集分析

### 6.3 block_id 的计算方式

```python
block_id = token_pos // block_size_tokens
```

这是一个**逻辑块编号**（与 vLLM 的 PagedAttention 物理块地址无关，只是把连续 token 区间等分）。当 `block_size_tokens=64` 时：
- token 0~63 → block 0
- token 64~127 → block 1
- 依此类推

分析脚本用这个方式现场把 `selected_token_pos`（2048 个 token 的绝对位置）映射到 block id，**不读** `selected_block_ids`（它默认不写入 JSONL）。

### 6.4 step_union 模式的数据流

```
JSONL 记录（layer 0 step 5）→ ┐
JSONL 记录（layer 1 step 5）→ ├─ 合并所有 layer → 虚拟 record (request=0, step=5, blocks=union)
JSONL 记录（layer 2 step 5）→ ┘
...
JSONL 记录（layer 0 step 6）→ 触发上一个 step 的虚拟 record 写入，开始新的 pending
```

注意：这要求 JSONL 中同一 `(request_id, step_idx)` 的所有 layer 记录**在文件里是连续的**（或至少同一个 step 的记录不会和下一个 step 的记录交错太远）。由于分片是按 request_id 确定性分桶的，同一 step 的记录通常是连续的。

---

## 7. 输出 JSON schema 逐字段详解

顶层结构：

```json
{
  "results": [
    {
      "trace_dir": "...",
      "block_size_tokens": 64,
      "key_mode": "request_layer",
      "run_meta": { ... },
      "record_stats": { ... },
      "temporal_locality": { ... },
      "working_set": { ... },
      "cache_sim": { ... }
    }
  ]
}
```

每个 `OnlineStats` 对象的格式为：

```json
{"count": N, "min": ..., "max": ..., "mean": ..., "std": ...}
```

其中 `count` 是样本数，`std` 是样本标准差。

---

### 7.1 `record_stats`（每条 JSONL 记录的 per-record 统计）

每条 JSONL 记录 = 一次 decode step × 一个 request × 一个 layer（或 step_union 后的虚拟 record）。

#### `record_stats.unique_blocks`

> 类型：`OnlineStats`

每条记录里 DSA 选中的 token 映射到多少个**去重 block**。

- 含义：`|{token_pos // block_size for token_pos in selected_token_pos}|`
- **越大表示越分散**，一次 attention 需要触达更多 block
- DSA top-2048 在长序列下，`unique_blocks` 通常 ≤ 2048，但远小于 seq_len / block_size（否则不叫稀疏注意力了）

---

#### `record_stats.block_span`

> 类型：`OnlineStats`

`max(block_ids) - min(block_ids) + 1`。

- 含义：触达的最远和最近 block 之间的"跨度"（包含所有中间 block，不管是否被触达）
- `block_span >> unique_blocks` → 稀疏散点式访问（远端+近端均有，中间跳过）
- `block_span ≈ unique_blocks` → 连续访问（触达的块是相邻的）

---

#### `record_stats.block_density`

> 类型：`OnlineStats`

`unique_blocks / block_span`。

- 范围：`(0, 1]`，越接近 1 越连续
- 高 density（> 0.8）→ 访问集合在 block 地址上是相邻的 → 预取和近端常驻策略更有效
- 低 density（< 0.3）→ 稀疏散点 → 每个远端 fetch 对应一个块，无法聚合成顺序 I/O

---

#### `record_stats.tokens_per_block_mean`

> 类型：`OnlineStats`

每个被触达的 block 里平均用了多少 token（包含重复计数：同一 block 内被选中的 token 数）。

- `= len(selected_token_pos) / unique_blocks`（近似，忽略去重 token 的差异）
- **越小越差**：block size=64，但 `tokens_per_block_mean=1`，意味着每搬一个 64-token 的 block 只用到 1 个 token → 有效利用率 1/64 = 1.6%，搬运开销极大
- **越大越好**：利用率高，每次 fetch 物有所值

---

#### `record_stats.block_concentration.normalized_entropy`

> 类型：`OnlineStats`

触达各 block 的 token 数量分布的**归一化 Shannon 熵**：

```
H = -sum(p_i * log(p_i))   其中 p_i = tokens_in_block_i / total_tokens_selected
normalized_H = H / log(unique_blocks)
```

- 范围：`[0, 1]`
- `normalized_H ≈ 0`：几乎所有 token 都集中在极少数 block → 极度集中，热点非常明显
- `normalized_H ≈ 1`：每个 block 分到的 token 数量均匀 → 无明显热点，放置策略难以通过"只保留热块"降低 miss rate

---

#### `record_stats.block_concentration.gini`

> 类型：`OnlineStats`

触达各 block 的 token 数量分布的 **Gini 系数**（经济学中的"不平等系数"）：

```
Gini = (2 * sum(i * x_i) / (n * sum(x))) - (n+1)/n
     （x 排序后，i 从 1 到 n）
```

- 范围：`[0, 1]`
- `Gini ≈ 0`：均匀分布（每块被 touch 的 token 数相近）
- `Gini ≈ 1`：极端不均（少数 block 包含绝大多数 selected tokens）
- **Gini 高 + normalized_entropy 低 → 热点明显 → 适合 hot-block 常驻/多副本策略**

---

#### `record_stats.recent_locality.recent_token_ratio`

> 类型：`{window_size: OnlineStats}`，默认窗口 256/1024/4096

对每条记录，`selected_token_pos` 中有多少比例落在 **"尾部（最近）N tokens"** 内：

```
ratio = |{pos : seq_len - 1 - pos <= N}| / len(selected_token_pos)
```

- **高 ratio（接近 1.0）**：DSA 主要关注最近生成的 token，局部性强 → 近端缓存的最近 KV 就够用，不需要保留很久以前的 block
- **低 ratio（接近 0）**：DSA 广泛访问历史 token → 必须保留较长的历史 KV

---

#### `record_stats.recent_locality.recent_block_ratio`

> 类型：`{window_size: OnlineStats}`，默认窗口 4/16/64

对每条记录，触达的 block 中有多少比例落在 **"距当前尾部 block ≤ W 块"** 内：

```
cur_block = (seq_len - 1) // block_size
ratio = |{b : cur_block - b <= W}| / unique_blocks
```

- 比 `recent_token_ratio` 的粒度更粗，直接对应"近端 block 池的窗口大小"
- 例如：`recent_block_ratio[16].mean = 0.7` → 平均 70% 的访问落在最近 16 blocks 内 → "只保留最近 16 blocks 在近端"就能满足 70% 的需求

---

#### `record_stats.recent_locality.mean_block_distance`

> 类型：`OnlineStats`

触达的 block 到当前序列尾部 block 的平均距离（`cur_block - b`）：

```
mean_dist = mean([cur_block - b for b in unique_block_ids])
```

- 越大表示 DSA 更频繁访问远端历史 → 做远端 KV fetch 代价大
- 结合 `unique_blocks` 可以估算：`mean_dist * bytes_per_block = 一次 decode step 平均访问的 KV 距当前尾部的加权距离`

---

### 7.2 `temporal_locality`（时间维度的局部性指标）

#### `temporal_locality.step_jaccard`

> 类型：`OnlineStats`

相邻两个 decode step 的 touched-block set 之间的 Jaccard 相似度：

```
Jaccard(S_t, S_{t+1}) = |S_t ∩ S_{t+1}| / |S_t ∪ S_{t+1}|
```

- 每个 "stream"（`key_mode` 决定粒度）独立计算，然后聚合
- **Jaccard 高（> 0.5）**：相邻 step 的访问集合高度重叠 → 缓存预热后命中率高；每次 decode step 新增的工作集增量小
- **Jaccard 低（< 0.2）**：访问集合变化剧烈 → 基于 LRU 的缓存效果差；可能需要预测性 prefetch 或更大的工作集窗口

---

#### `temporal_locality.reuse_distance_hist_steps` 和 `reuse_distance_cdf_steps`

> 类型：`{distance_in_steps: count}` 直方图 + CDF 数组

对每一个 touched block，记录距离上次被访问经过了多少 decode step（重用距离）：

- 若 block 从未被访问过 → 计入 `first_touch_blocks`（冷启动，不进直方图）
- 否则：`reuse_distance = current_step - last_seen_step`，clip 到 `reuse_cap_steps`

**CDF 的用法**：

```
reuse_distance_cdf_steps 是按 distance 排序的累积概率
CDF[d] = P(重用距离 ≤ d)
```

示例解读：
- CDF 在 d=10 处 = 0.8 → 80% 的重用在 10 步以内发生 → 如果缓存能 "cover 最近 10 步的工作集"，可以命中 80% 的重用访问
- 结合 `working_set` 指标：`W=10 时的 working_set.mean` 就是"覆盖 10 步工作集需要的 block 容量均值"

**`first_touch_blocks`**：从未被重复触达的 block 数量（只被访问过一次），适合评估"有多少 KV block 是 one-shot 的，放到近端是浪费"。

---

### 7.3 `working_set`（工作集大小）

> 类型：`{window_size: OnlineStats}`，默认 W=1/4/16/64

`working_set[W]` 表示每个统计时刻（每处理完一条 record），**当前 stream 最近 W 个 step** 内触达的 block 的总去重数量。

数学定义：

```
WS(t, W) = |⋃ S_{t-W+1} ... S_t|
```

- 每个时刻都往 `OnlineStats` 里加一次 `len(WS(t, W))` → 输出是 WS 大小在时间维度上的分布（均值、std、min、max）

**用法**：

- `working_set["1"].mean` ≈ 每个 decode step 平均触达多少个不同 block（≈ `unique_blocks.mean`）
- `working_set["64"].mean` → 如果你的 KV 缓存可以常驻"一个 stream 最近 64 步的工作集"，大概需要多少 blocks 容量
- `working_set["W"].max` → 峰值需求，用于容量规划的上界

---

### 7.4 `cache_sim`（LRU 缓存模拟）

> 类型：`{capacity: {hits, misses, hit_rate}}`，默认容量 128/256/512/1024 blocks

对每个 stream，使用**经典 LRU（Least Recently Used）**模拟一个固定容量的 block 缓存：

- 每个 decode step 的 touched block set 作为一组 batch access
- 命中（block 在缓存中）→ `hits += 1`，并更新到 MRU 位置
- 缺失（block 不在缓存中）→ `misses += 1`，加入缓存，若超出容量则驱逐 LRU

```json
"cache_sim": {
  "128": {"capacity_blocks": 128, "hits": 41200, "misses": 8800, "hit_rate": 0.824},
  "256": {"capacity_blocks": 256, "hits": 46300, "misses": 3700, "hit_rate": 0.926},
  ...
}
```

**用法**：

- 直接回答"如果近端只有 N 个 block 的缓存，LRU 命中率是多少"
- 观察命中率随容量的增长曲线，找到"边际收益递减"的 knee point（性价比最优容量）
- 注意：这是 per-stream 的 LRU（每个 request-layer pair 独立）。实际 serving 里多 request 共享同一物理缓存，命中率可能更低（竞争驱逐）或更高（共享 prefix）

---

## 8. 指标含义与 KV 放置策略的关联

下面给出"你手头有这些指标数据，如何决定采用哪种放置策略"的决策框架。

### 8.1 策略判断速查表

| 指标值 | 建议策略 |
|---|---|
| `unique_blocks.mean` 很大（例如 >100 @ block_size=64） | 分散型访问，无法靠小容量近端覆盖全部，需要远端大池；考虑 prefetch 而非 pin |
| `gini.mean` 高（>0.5） | 热点明显，少数 block 贡献大部分 token；**热点常驻（Hot-Block Pinning）**策略有效 |
| `normalized_entropy.mean` 低（<0.3） | 与 gini 高配合，进一步确认集中度 |
| `tokens_per_block_mean.mean` 低（<5 @ block_size=64） | 每 block 利用率差，搬运性价比低；考虑 sub-block 压缩或增大 block size |
| `step_jaccard.mean` 高（>0.6） | 相邻 step 访问集合稳定，**LRU 很有效**；近端缓存预热后命中率高 |
| `step_jaccard.mean` 低（<0.2） | 访问集合波动大，LRU 效果差；考虑**Working-Set 型策略**或 prefetch |
| `reuse_distance_cdf` 在 d=10 处 CDF > 0.8 | 大部分重用在 10 步以内；**小容量近端缓存**足够，`working_set["10"].mean` 即是所需容量 |
| `reuse_distance_cdf` 在 d=100 处 CDF < 0.5 | 重用发生较晚；需要**更大的近端缓存**或**分层缓存**（HBM + local DRAM） |
| `recent_block_ratio["16"].mean` 高（>0.7） | 绝大多数访问集中在最近 16 blocks；**Sliding Window 策略**（保留最新 N blocks 在近端）有效 |
| `recent_block_ratio["64"].mean` 低 | 即使窗口放大到 64 blocks 覆盖率仍低；需要 history-aware 策略，而不是简单时间窗口 |
| `cache_sim["256"].hit_rate > 0.9` | 256 blocks 的 LRU 缓存命中率超过 90%；放置策略工作量不大，近端容量容易满足需求 |
| `working_set["64"].mean >> cache_sim["256"].hit_rate 低` | 工作集大但 LRU 命中率也低，说明访问模式对 LRU 不友好（存在频繁的 cache thrashing）；考虑 LFU 或 hot-block 策略 |

### 8.2 分层放置设计思路（三层：HBM ↔ 本机 DRAM ↔ 远端）

用这些指标可以分层决策：

**近端（HBM，小容量）**
- 容量：`working_set[W_hbm].mean`（取 W_hbm 为你的 HBM 容量约束对应的 step 数）
- 填什么：`recent_block_ratio` 或 `step_jaccard` 高时，保留最近窗口；`gini` 高时保留 top-hot blocks

**中层（本机 DRAM，中等容量）**
- 容量：`working_set[W_dram].mean`（通常 W_dram >> W_hbm）
- 填什么：hot blocks（热度由跨多个 request 的 touch count 排名）

**远端（远程存储/不缓存）**
- 剩余 blocks：放置在这里，每次 fetch 有网络延迟
- 用 `mean_block_distance` 和 `reuse_distance` 评估 fetch 代价

---

## 9. 典型使用场景示例

### 场景一：评估 block size 对 unique_blocks 的影响

```bash
python3 analysis/analyze_trace.py \
  --input outputs/ruler_12345/block64 \
  --block-sizes 16,32,64,128 \
  --out outputs/ruler_12345/bs_sweep.json
```

读取结果：

```python
import json
with open("outputs/ruler_12345/bs_sweep.json") as f:
    results = json.load(f)["results"]

for r in results:
    bs = r["block_size_tokens"]
    ub = r["record_stats"]["unique_blocks"]
    print(f"block_size={bs:3d} | unique_blocks mean={ub['mean']:.1f} std={ub['std']:.1f}")
```

---

### 场景二：确定近端缓存容量（LRU 命中率曲线）

```bash
python3 analysis/analyze_trace.py \
  --input outputs/ruler_12345/block64 \
  --lru-caps 64,128,256,512,1024,2048
```

```python
import json
with open("outputs/ruler_12345/block64/analysis.json") as f:
    r = json.load(f)["results"][0]

for cap, sim in sorted(r["cache_sim"].items(), key=lambda kv: int(kv[0])):
    print(f"cap={cap:5s} blocks | hit_rate={sim['hit_rate']:.3f}")
```

输出示例：

```
cap=  64 blocks | hit_rate=0.612
cap= 128 blocks | hit_rate=0.741
cap= 256 blocks | hit_rate=0.852
cap= 512 blocks | hit_rate=0.924
cap=1024 blocks | hit_rate=0.961
cap=2048 blocks | hit_rate=0.979
```

在 512 处命中率已 >90%，增加到 1024 只多 3.7%，可以判断 512 blocks 是性价比的 knee point。

---

### 场景三：整个 decode step 的总工作集（step_union）

```bash
python3 analysis/analyze_trace.py \
  --input outputs/ruler_12345/block64 \
  --step-union \
  --key-mode request \
  --ws-windows 1,4,16 \
  --out outputs/ruler_12345/step_union_analysis.json
```

结果里的 `working_set["1"].mean` 就是"一个 decode step 跨所有 61 层的 unique block 数"。乘以 `bytes_per_block` 就是每步真实的 KV 工作集大小。

---

### 场景四：快速采样（大规模 trace 的探索）

```bash
python3 analysis/analyze_trace.py \
  --input outputs/ruler_12345/block64 \
  --max-steps-per-stream 100   # 每个 stream 只分析前 100 步
```

---

### 场景五：跨数据集对比

```bash
for ds in ruler longbenchv2 sharegpt burstgpt; do
  LATEST=$(ls -td outputs/${ds}_* | head -1)
  python3 analysis/analyze_trace.py \
    --input "$LATEST" \
    --block-sizes 64 \
    --out "$LATEST/analysis.json"
  echo "=== $ds ==="
  python3 -c "
import json, sys
r = json.load(open('$LATEST/block64/analysis.json'))['results'][0]
ub = r['record_stats']['unique_blocks']
j = r['temporal_locality']['step_jaccard']
sim = r['cache_sim'].get('512', {})
print(f'  unique_blocks mean={ub[\"mean\"]:.1f}')
print(f'  step_jaccard  mean={j[\"mean\"]:.3f}')
print(f'  LRU-512 hit_rate={sim.get(\"hit_rate\",0):.3f}')
"
done
```

---

## 10. 性能与注意事项

### 10.1 内存

- 脚本流式处理 JSONL，不会把文件整体读入内存
- 主要内存消耗来自 **per-stream state**（`last_seen_step_by_block`、`ws_deques`、`lru_deques`）
- 估算：假设 N_streams = N_requests × 61 layers，每个 stream 维护最多 `max(ws_windows)` 个 step 的 block set 和一个 LRU deque。如果有 100 requests × 61 layers = 6100 streams，每个 stream 的 deque 最大 64 个 step，每 step 最多 2048 blocks，约 `6100 × 64 × 2048 × 8 bytes ≈ 6 GB`（极端上界，实际远小于此）
- **如果内存有压力**：用 `--max-steps-per-stream 100` 截断，或 `--key-mode request` 减少 stream 数

### 10.2 速度

- 纯 Python（stdlib only），没有 numpy/pandas 加速
- 大 trace（几十 GB）可能需要几分钟到几十分钟
- 可以用 `--max-steps-per-stream` 快速预览，确认指标方向后再跑全量

### 10.3 `selected_token_pos` 缺失

- 如果采集 trace 时设了 `DS_TRACE_STORE_TOKEN_POS=0`，JSONL 记录中没有 `selected_token_pos`
- 脚本会跳过这些记录，`record_stats` 的 `count=0`
- 解决办法：重新采集 trace，或使用 trace 里已经预计算的 `unique_blocks` 字段（但它是固定 block_size 的，无法重算）

### 10.4 数值精度

- `gini` 和 `normalized_entropy` 用 float64 计算，精度足够
- `OnlineStats` 用 Welford 近似的简化版（用 `sumsq/n - mean^2` 估计方差），在 n 很大时可能有数值误差，但对分析目的足够

### 10.5 LRU 实现说明

当前 LRU 实现用 `deque + set` 模拟，时间复杂度 O(capacity) per access（因为需要在 deque 中 remove 已有元素）。如果 `capacity` 很大（如 8192）且 trace 很长，可能较慢。实际 analysis 中 LRU capacity 建议不超过 4096 blocks。
