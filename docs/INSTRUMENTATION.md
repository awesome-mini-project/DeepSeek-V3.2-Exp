# DSA Trace Instrumentation

## 1. 项目背景与目标

本仓库（`inference/`）是 DeepSeek-V3.2-Exp 的**单机推理 demo**（非 vLLM 本体）。
我们在这份 demo 上加入了 **A/B/C/D 四类插桩点**，目标是：

> 在 DeepSeek V3.2-Exp（DSA top-2048）的推理路径上，导出可复现的 **token→block 访问轨迹**，
> 并同时记录外部 KV（MemFabric/MemCache/Mooncake）取数代价与 prefix cache 复用关系。

这一步**只做采数，不做 KV 策略**。采集到的轨迹是后续设计"MemFabric subblock/packing、距离感知放置/副本/迁移、fetch vs recompute 边界"的必要输入。

补充：如果你需要把**系统论文评测到底用的哪个数据集变体**（精确到 HF ID / 文件名 / split / trace schema）记录清楚，见：

- `docs/SYSTEM_WORKLOAD_DATASETS.md`

---

## 2. DeepSeek-V3.2-Exp 模型长度限制

### 2.1 官方 HF 权重的上限

从 HuggingFace 官方 `config.json` 可以读到：

| 参数 | 值 | 含义 |
|---|---|---|
| `max_position_embeddings` | **163,840** | 模型位置编码支持的最大 token 数（~160K） |
| `rope_scaling.original_max_position_embeddings` | 4,096 | 原始训练长度 |
| `rope_scaling.factor` | 40 | YaRN 扩展倍率（4096 x 40 = 163,840） |
| tokenizer `model_max_length` | **131,072** | tokenizer 警告上限（128K） |

### 2.2 本推理 demo 的实际上限

`ModelArgs.max_seq_len` 的 **dataclass 默认值**是 `4096 * 4 = 16,384`。
`config_671B_v3.2.json` 现已显式设置 `"max_seq_len": 163840`（与官方 `max_position_embeddings` 对齐）。

KV cache / indexer cache 都是按 `max_seq_len` **预分配**的固定张量，显存开销估算：

| max_seq_len | 8 batch x 61 layers KV cache |
|---|---|
| 16,384 | ~10 GB |
| 65,536 | ~39 GB |
| 131,072 | ~78 GB |
| 163,840 (当前配置) | ~97 GB |

8x B200（每卡 192GB HBM）下 163,840 完全放得下（每卡约 12 GB KV cache）。
如果你显存不够，可以在 config 里减小此值或减小 `max_batch_size`。

### 2.3 对数据集的影响（基于 max_seq_len=163840）

- **RULER**（debug 包 ~4K tokens）：绝大部分样本能跑进去
- **ShareGPT**（对话通常 < 4K tokens）：大部分样本可跑
- **LongBench v2**（上下文可达 272K tokens）：超过 163K 的样本仍会被跳过，但大部分（8K~128K）可以跑
- **BurstGPT**（合成 prompt，长度可控）：按 `req_tokens` 生成，可以完全控制在限制内

---

## 3. 四个插桩点详解

### 3.1 插桩点 A：DSA top-2048 选择输出

- **位置**：`inference/model.py` → `Indexer.forward()` → `index_score.topk(...)` 之后
- **触发条件**：仅在 **decode 阶段**（`mask is None and seqlen == 1`）记录
- **记录字段**：

| 字段 | 说明 |
|---|---|
| `request_id` | 请求 ID（batch index 或 dataset runner 分配） |
| `layer_id` | 当前 MLA 层编号（0~60） |
| `step_idx` | 当前 decode step（= `start_pos`） |
| `seq_len_current` | 当前已有的序列长度（= `end_pos`） |
| `selected_token_pos` | DSA 选出的 top-2048 token 绝对位置（列表，最多 2048） |
| `unique_token_pos_count` | 去重后的 token 位置数量 |
| `offset_min/p50/max` | 距当前 token 的距离分布（`(end_pos-1) - pos`） |
| `scores_stats` | top-k 分数的 min/mean/max（默认只写 stats，不写全量） |
| `scores_topk` | 可选：开启 `--trace-store-scores` 后写全部 2048 分数 |

### 3.2 插桩点 B：token→KV block 映射

- **位置**：与 A 点同一记录，在 `trace.py` 的 `record_dsa_topk()` 内部计算
- **逻辑**：`block_id = token_pos // kv_block_size_tokens`（逻辑块近似，非真实 PagedAttention）
- **记录字段**：

| 字段 | 说明 |
|---|---|
| `block_size_tokens` | 逻辑 block 大小（由 `--kv-block-size` 控制，默认 16） |
| `selected_block_ids` | 去重后的 block ID 列表 |
| `unique_blocks` | `\|B_t\|`（去重 block 数） |
| `tokens_per_touched_block` | 每个被触及 block 里用了多少 token 的统计：`{mean, p50, p95}` |

### 3.3 插桩点 C：KV 取数与搬运代价

- **位置**：与 A/B 同一记录
- **当前实现**：**HBM-only 占位**（所有 block 都算 HBM 命中，local/remote = 0）
- **可插拔接口**：`kv_fetch` 字段保持三层结构（`hbm/local_pool/remote_pool`），后续接入真实外部 KV 时直接填入
- **记录字段**：

| 字段 | 说明 |
|---|---|
| `kv_fetch.hbm.hit_blocks` | HBM 命中的 block 列表（= `selected_block_ids`） |
| `kv_fetch.hbm.bytes_read` | 估算的读取字节数（按 block 数 × bytes_per_token × block_size） |
| `kv_fetch.hbm.read_ops` | 读操作次数（默认 1） |
| `kv_fetch.hbm.latency_us` | step wall-time（微秒），由 `generate.py` 用 `cuda.synchronize()` + perf_counter 测量 |
| `kv_fetch.hbm.batch_size` | 一次 batch_get 覆盖的 block 数 |
| `kv_fetch.local_pool.*` | 预留占位（全 0） |
| `kv_fetch.remote_pool.*` | 预留占位（全 0） |

### 3.4 插桩点 D：Prefix cache 交互

- **位置**：与 A/B/C 同一记录（prefix 信息由 dataset runner 在每个 request 上生成）
- **当前实现**：**复用关系分析**（基于 prompt 前 N token 的 hash），不做真实 KV 复用
- **记录字段**：

| 字段 | 说明 |
|---|---|
| `prefix.prefix_cache_hit` | 该 request 的前缀是否已被之前的 request 见过 |
| `prefix.prefix_cached_blocks` | 可复用前缀覆盖的 block 数 |
| `prefix.prefix_key` | 前缀 hash |
| `prefix.intersection_ratio` | `\|touched_blocks ∩ prefix_blocks\| / \|touched_blocks\|` |
| `prefix.intersection_blocks` | 交集 block 列表 |

---

## 4. 代码架构与文件说明

```
DeepSeek-V3.2-Exp/
├── inference/
│   ├── model.py              # 模型定义；Indexer.forward() 是 A 点插桩位置
│   ├── generate.py           # 生成循环；在 model.forward() 外围测 step wall-time
│   ├── trace.py              # 插桩核心：TraceConfig / Tracer / TraceWriter / PrefixCacheAnalyzer
│   ├── run_dataset.py        # 数据集 runner（RULER / LongBench v2 / ShareGPT）
│   ├── run_burstgpt.py       # BurstGPT 到达过程模拟 runner
│   ├── sanity_trace_no_torch.py  # 不依赖 torch 的 schema 自检脚本
│   ├── convert.py            # HF → demo 格式的权重转换
│   ├── kernel.py             # tilelang fp8 kernel
│   ├── config_671B_v3.2.json # 671B 模型配置（注意：未覆盖 max_seq_len）
│   ├── requirements.txt      # Python 依赖
│   ├── INSTRUMENTATION.md    # 本文档
│   └── README.md             # 推理 demo 简要说明
├── scripts/
│   ├── run_trace_ruler.sh        # RULER 跑数脚本
│   ├── run_trace_longbenchv2.sh  # LongBench v2 跑数脚本
│   ├── run_trace_sharegpt.sh     # ShareGPT 跑数脚本
│   ├── run_trace_burstgpt.sh     # BurstGPT 跑数脚本
│   └── datasets/
│       ├── download_ruler.sh         # 下载+解压 RULER
│       ├── download_longbench_v2.sh  # 下载 LongBench v2
│       ├── download_sharegpt.sh      # 下载 ShareGPT
│       └── download_burstgpt.sh      # 下载 BurstGPT（导出 CSV）
├── outputs/    # trace 输出目录（.gitignore）
├── data/       # 数据集缓存+导出目录（.gitignore）
└── logs/       # 运行日志（.gitignore）
```

### 4.1 数据流

```
[数据集] ──tokenize──→ [prompt_tokens]
                           │
                    ┌──────▼──────┐
                    │ generate()  │  ← 在每个 model.forward() 外围测 step wall-time
                    │  ┌────────┐ │
                    │  │ model  │ │
                    │  │ .forward│ │
                    │  │  ┌───┐ │ │
                    │  │  │MLA│ │ │  ← 每层的 MLA 包含一个 Indexer
                    │  │  │   │ │ │
                    │  │  │ Indexer.forward()
                    │  │  │   │ │ │  ← A点：topk_indices + topk_values
                    │  │  │   │ │ │  ← B点：token_pos → block_id 映射
                    │  │  │   │ │ │  ← C点：HBM-only bytes/ops/latency
                    │  │  │   │ │ │  ← D点：prefix ∩ touched 交集
                    │  │  └───┘ │ │
                    │  └────────┘ │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ TraceWriter │ → trace_steps.jsonl
                    │   Tracer    │ → summary.json
                    └─────────────┘
```

---

## 5. 输出文件与 JSONL Schema

### 5.1 输出目录

由 `--trace-out` 指定（或自动生成到 `outputs/<dataset>_<timestamp>/`）。

```
outputs/ruler_1771772829/
├── trace_steps.jsonl    # 逐 step 逐 layer 的 DSA 轨迹（核心产物）
└── summary.json         # 汇总统计
```

### 5.2 `trace_steps.jsonl` 每行 schema

每行是一个 `event=dsa_topk` 记录（**每 decode step × 每 request × 每 layer**）：

```json
{
  "event": "dsa_topk",
  "run_name": "ruler_1771772829",
  "dataset": "ruler",
  "rank": 0,
  "world_size": 8,
  "request_id": 0,
  "layer_id": 5,
  "step_idx": 42,
  "seq_len_current": 43,
  "unique_token_pos_count": 43,
  "offset_min": 0,
  "offset_p50": 21,
  "offset_max": 42,
  "block_size_tokens": 16,
  "selected_block_ids": [0, 1, 2],
  "unique_blocks": 3,
  "tokens_per_touched_block": {"mean": 14.3, "p50": 16, "p95": 16},
  "kv_fetch": {
    "hbm":        {"hit_blocks": [0,1,2], "bytes_read": 55296, "read_ops": 1, "latency_us": 417074, "batch_size": 3},
    "local_pool": {"hit_blocks": [],      "bytes_read": 0,     "read_ops": 0, "latency_us": null,   "batch_size": 0},
    "remote_pool":{"hit_blocks": [],      "bytes_read": 0,     "read_ops": 0, "latency_us": null,   "batch_size": 0}
  },
  "prefix": {
    "prefix_cache_hit": false,
    "prefix_cached_blocks": 0,
    "prefix_key": "a1b2c3...",
    "intersection_ratio": 0.0,
    "intersection_blocks": []
  },
  "selected_token_pos": [0, 5, 1, 2, 4, 3, ...],
  "scores_stats": {"min": -4.77, "mean": -2.13, "max": 1.22}
}
```

### 5.3 `summary.json` schema

```json
{
  "run_name": "ruler_xxx",
  "dataset": "ruler",
  "config": {"kv_block_size_tokens": 16, ...},
  "unique_blocks":              {"count": N, "min": ..., "p50": ..., "p95": ..., "max": ..., "mean": ...},
  "tokens_per_touched_block":   {"count": N, "min": ..., "p50": ..., "p95": ..., "max": ..., "mean": ...},
  "offsets":                    {"count": N, "min": ..., "p50": ..., "p95": ..., "max": ..., "mean": ...},
  "prefix_intersection_ratio":  {"count": N, "min": ..., "p50": ..., "p95": ..., "max": ..., "mean": ...},
  "prefix_hot_blocks": {"0": [{"block_id": 0, "touch_count": 42}, ...], ...},
  "serving": {  // BurstGPT 专有
    "requests": 256,
    "latency_ms": {"p50": ..., "p95": ..., "p99": ..., "max": ...},
    "batch_size": 4,
    "max_new_tokens_cap": 64
  }
}
```

---

## 6. 四类数据集详解

### 6.1 RULER（长上下文、可控长度与复杂度）

| 项目 | 说明 |
|---|---|
| **用途** | 最大化/放大 DSA 的"非局部选择"，测 unique_blocks、tokens_per_touched_block、offset 分布 |
| **来源论文** | RULER: What's the Real Context Size of Your Long-Context Language Models? (arXiv:2404.06654) |
| **HF 来源** | `allenai/ruler_data`（注意：不是标准 json/parquet，是 tgz 压缩包） |
| **本项目处理** | runner 自动下载 tgz → 解压到 `data/ruler/` → 逐行读 jsonl 并统一成 `{"input": prompt}` schema |
| **包含任务** | NIAH single/multi-key/multi-value/multi-query、CWE、FWE、QA、Variable Tracking 等 13 类 |
| **典型 token 长度** | debug 包 ~4K tokens（适合当前 demo 的 16K 限制） |
| **切换大包** | `RULER_TGZ=data_100_samples.tgz ./scripts/run_trace_ruler.sh` |

### 6.2 LongBench v2（真实长任务、多场景）

| 项目 | 说明 |
|---|---|
| **用途** | 证明 RULER 上观测到的访问特性不是合成数据特例；采集 DSA 访问图与跨层取数代价 |
| **来源论文** | LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks (arXiv:2412.15204) |
| **HF 来源** | `zai-org/LongBench-v2`（503 条多选题，train split only） |
| **典型 token 长度** | 8K ~ 2M words；**大量样本超过 demo 的 16K 限制**，会被自动跳过 |
| **注意** | 当前 demo 下有效样本可能很少（只有 `length=short` 的能跑进去）；如需跑更多，需在 config 里加大 `max_seq_len` 并确保显存足够 |

### 6.3 ShareGPT（对话类请求分布）

| 项目 | 说明 |
|---|---|
| **用途** | 采集"典型对话请求"下 DSA 访问图 + prefix cache 交互（多轮/模板化带来共享前缀） |
| **论文可追溯引用** | OpenChat 论文在实验设置中明确采用 ShareGPT 作为常用 SFT/对话数据集 |
| **HF 来源** | 默认 `anon8231489123/ShareGPT_Vicuna_unfiltered`（~53K 对话） |
| **格式** | `conversations` 列表，每条含 `from: human/gpt` 和 `value` 字段 |
| **典型 token 长度** | 大部分 < 4K tokens，**适合当前 demo** |
| **多轮模式** | `--sharegpt-turn-mode full`（默认：整段对话一个 request）或 `per_user_turn`（每个 user turn 一个 request，更贴近 serving 形态） |
| **也支持本地 JSON** | `--sharegpt-json /path/to/sharegpt.json` |

### 6.4 BurstGPT（真实到达过程 / 突发并发）

| 项目 | 说明 |
|---|---|
| **用途** | 把插桩扩展到"并发+队列+tail latency"情形；记录跨层命中、远端 bytes/ops、p99 延迟随突发变化 |
| **来源论文** | BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems (arXiv:2401.17644) |
| **HF 来源** | `lzzmm/BurstGPT`（10.31M traces） |
| **格式** | CSV：`Timestamp, Request tokens, Response tokens, Total tokens, Model` |
| **本项目处理** | 建议先用 `LIMIT=2000 ./scripts/datasets/download_burstgpt.sh` 导出小 CSV |
| **runner 行为** | `run_burstgpt.py` 按 CSV 里的 timestamp 模拟 FIFO 到达+批处理，用合成 prompt（token 长度按 `Request tokens` 填充） |
| **额外输出** | `summary.json` 里包含 `serving.latency_ms{p50,p95,p99,max}` |

---

## 7. 数据集下载与缓存

### 7.1 下载发生在什么时候？

- **RULER / LongBench v2 / ShareGPT**：直接跑 `./scripts/run_trace_*.sh` 时，runner 内部的 `load_dataset(...)` 或 `hf_hub_download(...)` 会**首次运行时**自动从 HuggingFace 下载，之后复用缓存
- **BurstGPT**：建议先手动运行 `./scripts/datasets/download_burstgpt.sh` 导出 CSV

### 7.2 下载到哪里？

本项目**不会写 `~/.cache/huggingface/`**。所有脚本已统一设置：

- `HF_HOME` = `<repo>/data/huggingface/`
- `HF_HUB_CACHE` = `<repo>/data/huggingface/hub/`
- `HF_DATASETS_CACHE` = `<repo>/data/huggingface/datasets/`

如果你想自定义数据根目录：

```bash
export DATAS_DIR=/path/to/datas
./scripts/run_trace_ruler.sh
```

则缓存会落在 `/path/to/datas/huggingface/...`，导出文件在 `/path/to/datas/ruler/...` 等。

### 7.3 `CKPT_PATH` 是什么？

`CKPT_PATH` **不是数据集路径**，而是**模型权重（convert 后）的目录**：

```bash
export CKPT_PATH=/data/models/deepseek-v3.2-exp-s
```

里面应当有：`model0-mp8.safetensors ... model7-mp8.safetensors` + tokenizer 文件。

---

## 8. 完整运行指南

### 8.0 环境准备

```bash
cd inference
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt

# 可选但强烈建议
python3 -m pip install --no-build-isolation git+https://github.com/Dao-AILab/fast-hadamard-transform.git
python3 -m pip install git+https://github.com/tile-ai/tilelang
```

### 8.1 权重转换

```bash
cd inference
export EXPERTS=256
export MP=8
export HF_CKPT_PATH=/data/models/deepseek-v3.2-exp
export SAVE_PATH=/data/models/deepseek-v3.2-exp-s

python3 convert.py \
  --hf-ckpt-path "${HF_CKPT_PATH}" \
  --save-path "${SAVE_PATH}" \
  --n-experts "${EXPERTS}" \
  --model-parallel "${MP}"
```

### 8.2 Schema 自检（可选，不需要 GPU）

```bash
cd inference
python3 sanity_trace_no_torch.py
# 检查 outputs/sanity_no_torch_*/trace_steps.jsonl 和 summary.json
```

### 8.3 交互模式 + trace

```bash
cd inference
torchrun --nproc-per-node 8 generate.py \
  --ckpt-path /data/models/deepseek-v3.2-exp-s \
  --config config_671B_v3.2.json \
  --interactive \
  --trace-enable \
  --trace-out "../outputs/interactive_trace_$(date +%s)" \
  --kv-block-size 16
```

### 8.4 四类 workload 跑数

**在仓库根目录运行**（不是 `inference/` 目录）：

```bash
export CKPT_PATH=/data/models/deepseek-v3.2-exp-s

# RULER（推荐首选，样本短、能跑通）
./scripts/run_trace_ruler.sh

# LongBench v2（大量超长样本会被跳过；先验证管线为主）
./scripts/run_trace_longbenchv2.sh

# ShareGPT（对话分布，大部分样本可跑）
./scripts/run_trace_sharegpt.sh

# BurstGPT（需要先导出 CSV）
LIMIT=2000 ./scripts/datasets/download_burstgpt.sh
export BURSTGPT_CSV="data/burstgpt/burstgpt_train_limit2000.csv"
./scripts/run_trace_burstgpt.sh
```

### 8.5 验证输出

```bash
# 以 RULER 为例
OUT=$(ls -td outputs/ruler_* | head -1)
ls -lh "$OUT"
wc -l "$OUT/trace_steps.jsonl"
head -1 "$OUT/trace_steps.jsonl" | python3 -m json.tool
cat "$OUT/summary.json" | python3 -m json.tool
```

### 8.6 通用参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--kv-block-size` | 16 | 逻辑 block 大小（token 数），建议对齐后续 PagedAttention |
| `--trace-store-scores` | 关 | 开启后写全部 2048 分数（日志很大） |
| `--trace-sample-rate` | 1.0 | step 采样率（0.1 = 10%） |
| `--trace-prefix-key-tokens` | 256 | prefix hash 最大 token 数 |
| `--trace-no-sync-cuda` | 关 | 关掉 cuda.synchronize 减少开销 |
| `--limit` | 64 | 数据集样本数上限 |
| `--max-new-tokens` | 64/32 | 每条样本最多生成的 token 数 |
| `--batch-size` | 1 | 每批同时推理的 request 数 |
| `--ruler-tgz` | data_debug.tgz | RULER 包选择（或 data_100_samples.tgz） |
| `--sharegpt-turn-mode` | full | ShareGPT 多轮模式（full / per_user_turn） |

---

## 9. `/clear` 语义

- `generate.py --interactive` 的 `/clear` 只影响交互模式：清空 `messages`，相当于"新会话"
- **dataset runner（批处理）默认行为**：每个样本/每个 request 都是独立的（等价于"样本之间都 clear"）
- **ShareGPT `per_user_turn` 模式**：同一段对话内部不 clear；不同对话之间 clear

---

## 10. 已知限制与后续对齐点

- 当前没有真实的 vLLM PagedAttention，因此 **B 点 block_id 是逻辑近似**
- 当前没有外部 KV，因此 **C 点为 HBM-only**；`kv_fetch` 字段保持 tier 结构，便于后续接入 MemFabric/MemCache/Mooncake
- 当前没有真实 prefix cache，因此 **D 点是复用关系分析**（基于 prefix hash）
- 当前 demo `max_seq_len=16384`，无法跑 LongBench v2 的超长样本；后续如需真实长上下文轨迹，需接入 vLLM + PagedAttention

---

## 11. 常见问题排查

### `ModuleNotFoundError: torch`

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### HuggingFace 下载失败 / 需要权限

- `huggingface-cli login`
- 或 `export HF_TOKEN=hf_xxx`
- 有些数据集需要在 HF 网页先点"Agree and access"

### 日志太大

插桩是 **每 step x 每 request x 每 layer**，61 层模型日志会非常大：

- `--limit 8` + `--max-new-tokens 16`（先跑小样本验证管线）
- `--trace-sample-rate 0.1`
- 不要开 `--trace-store-scores`

### LongBench v2 跑完但 trace 为空

大量样本超过 `max_seq_len=16384` 被跳过。运行后检查：

```bash
wc -l outputs/longbenchv2_*/trace_steps.jsonl
```

如果是 0，说明没有样本短到能跑进去。

### `DataFilesNotFoundError: No (supported) data files found in allenai/ruler_data`

你的代码不是最新版本。运行 `git pull --ff-only` 更新。

### `CastError: column names don't match`（RULER）

同上：更新到最新版本，RULER 加载已改为逐行读取+统一 schema。

### `Token indices sequence length is longer than ... (272555 > 131072)`

这是 tokenizer 的警告（不是崩溃）。超长样本会被 runner 的 `max_prompt_tokens` 过滤跳过。
