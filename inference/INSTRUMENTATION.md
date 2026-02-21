## 目标与范围

本仓库当前只包含 `inference/` 的推理 demo（不是 vLLM 本体）。因此本次“插桩采数”实现的是：

- **A (DSA top-2048)**：在 `Indexer.forward()` 的 top-k 选择输出处记录每个 decode step、每个 request 的 `selected_token_pos`（可选分数）与轻量统计。
- **B (token→block)**：当前没有 PagedAttention block table，所以用**逻辑块近似**：`block_id = token_pos // kv_block_size_tokens`。
- **C (KV 取数代价)**：实现一个**可插拔的 tier 统计接口**；现阶段输出 **HBM-only**（local/remote=0），`bytes_read/read_ops/latency_us/batch_size` 可用于后续接外部 KV。
- **D (prefix cache 交互)**：不做 KV 复用，只做“复用关系分析”并输出 `prefix_cache_hit/prefix_cached_blocks` 与 `touched_blocks ∩ prefix_blocks` 的比例和热点。

这些日志用于后续把真实的外部 KV / prefix cache 接入 vLLM 时做对齐与扩展。

## Quickstart（推荐最小路径）

### 0) Python / CUDA 环境

- **Python**：建议 `python3.10+`
- **GPU**：需要 NVIDIA CUDA（本 demo 默认把 `tokens` 放在 `device="cuda"`）
- **依赖**：

```bash
cd inference
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt

# 可选但强烈建议（用于 indexer 的 fast hadamard + tilelang）
python3 -m pip install --no-build-isolation git+https://github.com/Dao-AILab/fast-hadamard-transform.git
python3 -m pip install git+https://github.com/tile-ai/tilelang
```

> 说明：如果不装 `fast_hadamard_transform`，代码会回退到纯 PyTorch Hadamard（更慢但能跑）。

### 1) 权重转换（HF → 本 demo 格式）

```bash
cd inference
export EXPERTS=256
export MP=8              # 默认建议 8；按实际 GPU 数设置；需满足 256 % MP == 0
export HF_CKPT_PATH=/data/models/deepseek-v3.2-exp
export SAVE_PATH=/data/models/deepseek-v3.2-exp-s

python3 convert.py \
  --hf-ckpt-path "${HF_CKPT_PATH}" \
  --save-path "${SAVE_PATH}" \
  --n-experts "${EXPERTS}" \
  --model-parallel "${MP}"
```

### 2) 先跑一个“无 torch 的 schema 自检”（可选）

这一步不跑模型，只验证 **trace_steps.jsonl / summary.json** 的输出结构，适合先在 CPU 环境确认“产物形状”。

```bash
cd inference
python3 sanity_trace_no_torch.py
```

它会打印输出目录，例如 `outputs/sanity_no_torch_xxx/`。

### 3) 直接在交互模式开插桩（最直观）

```bash
cd inference
export CONFIG=config_671B_v3.2.json
export MP=8

torchrun --nproc-per-node "${MP}" generate.py \
  --ckpt-path "${SAVE_PATH}" \
  --config "${CONFIG}" \
  --interactive \
  --trace-enable \
  --trace-out "../outputs/interactive_trace_$(date +%s)" \
  --kv-block-size 16
```

生成的 `trace_steps.jsonl` 会在你指定的目录下（这里是 `outputs/interactive_trace_*`）。

### 4) 跑四类 workload（RULER / LongBench v2 / ShareGPT / BurstGPT）

仓库根目录下直接运行脚本（它们会调用 `inference/run_dataset.py` 或 `inference/run_burstgpt.py`）：

```bash
export CKPT_PATH="${SAVE_PATH}"

./scripts/run_trace_ruler.sh
./scripts/run_trace_longbenchv2.sh
./scripts/run_trace_sharegpt.sh

# BurstGPT 需要 CSV（见下面“数据集下载/准备”）
export BURSTGPT_CSV="data/burstgpt/burstgpt_train_limit2000.csv"
./scripts/run_trace_burstgpt.sh
```

## 插桩点与代码位置

- **A 点（必须）**：`inference/model.py` → `Indexer.forward()`：`index_score.topk(...` 后。
- **B 点（必须）**：同 A 点记录里直接做映射（基于 `kv_block_size_tokens`）。
- **C 点（必须）**：同 A 点记录里输出 `kv_fetch`（HBM-only 占位，未来可替换为真实 tier 命中/搬运）。
- **D 点（必须）**：由 dataset runner 在每个 request 上生成 `prefix_info`，插桩记录中输出交集比例与热点。

实现模块：

- `inference/trace.py`：`TraceConfig / Tracer / TraceWriter / PrefixCacheAnalyzer`
- `inference/generate.py`：在每次 `model.forward()` 外围测 step wall-time 并写回 tracer
- `inference/run_dataset.py`：RULER / LongBench v2 / ShareGPT 跑数入口
- `inference/run_burstgpt.py`：BurstGPT 到达过程（FIFO + batch）跑数入口

## 输出文件与 schema（JSONL）

默认输出目录：由 `--trace-out` 指定（或自动生成到 `outputs/<dataset>_<timestamp>/`）。

### `trace_steps.jsonl`

每行是一个 `event=dsa_topk` 记录（**每 step × 每 request × 每 layer**；当前只在 decode 阶段写入）。

关键字段：

- **标识**：`request_id`, `layer_id`, `step_idx`, `seq_len_current`
- **A 点**：`selected_token_pos`（最多 2048），`unique_token_pos_count`，`offset_min/offset_p50/offset_max`
- **B 点**：`block_size_tokens`, `selected_block_ids`, `unique_blocks`, `tokens_per_touched_block{mean,p50,p95}`
- **C 点**：`kv_fetch.{hbm,local_pool,remote_pool}.{hit_blocks,bytes_read,read_ops,latency_us,batch_size}`
- **D 点**：`prefix.{prefix_cache_hit,prefix_cached_blocks,prefix_key,intersection_ratio,intersection_blocks}`
- **可选**：`scores_topk`（开启 `--trace-store-scores` 后写 2048 分数），否则只写 `scores_stats`（min/mean/max）

### `summary.json`

在线/离线汇总的分布统计：

- `unique_blocks` 分布（min/p50/p95/max/mean）
- `tokens_per_touched_block` 分布
- `offsets` 分布
- `prefix_intersection_ratio` 分布
- `prefix_hot_blocks`：每个 request 的交集热点 block top-20
- BurstGPT 额外包含 `serving.latency_ms{p50,p95,p99,max}`

## 数据集下载 / 准备

### RULER

- Runner 直接从 HF 拉取：`allenai/ruler_data`
- 或使用脚本导出 JSONL：

```bash
./scripts/datasets/download_ruler.sh
```

### LongBench v2

- Runner 使用：`zai-org/LongBench-v2`（503 条，多选题）
- 导出 JSONL：

```bash
./scripts/datasets/download_longbench_v2.sh
```

### ShareGPT（对话分布）

本项目默认用一个可以被 `load_dataset()` 直接加载的 ShareGPT 派生集，避免 “NOT compatible with HF loader” 的坑：

- 默认 HF：`anon8231489123/ShareGPT_Vicuna_unfiltered`
- 也支持你提供 **ShareGPT JSON 文件**：
  - 用法：`--sharegpt-json /path/to/sharegpt.json`

导出 JSONL（HF→本地）：

```bash
./scripts/datasets/download_sharegpt.sh
```

### BurstGPT（到达过程 / tail）

BurstGPT 数据量很大，建议先用脚本导出一个小 CSV（用于插桩阶段）：

```bash
LIMIT=2000 ./scripts/datasets/download_burstgpt.sh
```

会生成例如：`data/burstgpt/burstgpt_train_limit2000.csv`，然后运行：

```bash
export BURSTGPT_CSV="data/burstgpt/burstgpt_train_limit2000.csv"
./scripts/run_trace_burstgpt.sh
```

> 注意：HF 下载可能需要登录/Token，或需要你在 HF 页面先点一次 “Agree and access”。

## `/clear` 语义（你问的“每轮对话结束是否 clear”）

- `inference/generate.py --interactive` 的 `/clear` 只影响交互模式：它会清空 `messages`，相当于“新会话”。  
- **dataset runner（批处理）默认行为**：每个样本/每个 request 都是独立的（等价于“样本之间都 clear”）。  
- **ShareGPT 的多轮对话**：在 `run_dataset.py` 里提供两种模式：  
  - `--sharegpt-turn-mode full`（默认）：整段对话作为一个 request（一次性把历史喂给模型并生成结尾）。  
  - `--sharegpt-turn-mode per_user_turn`：每个 user turn 作为一个 request（同一段对话内部不 clear；不同对话之间 clear）。这更贴近 serving 中“多轮对话”的请求形态，也更容易在 D 点看到 DSA 与 prefix 的交集热点。  

## 如何跑四类数据集

### 依赖

`inference/requirements.txt` 增加了 `datasets`。建议使用 `python3`。

### 通用参数（强烈建议你在插桩阶段就设好）

- **`--kv-block-size`**：逻辑 block 大小（token 数）。建议对齐你后续准备接入的 PagedAttention block size（常见 16/32）。
- **`--trace-store-scores`**：如果要存每步的 top-2048 分数会很大；默认只写 `scores_stats`。
- **`--trace-sample-rate`**：默认 1.0（全量）。如果日志太大可设成 `0.1`（10% step 采样）。
- **`--trace-prefix-key-tokens`**：prefix hash 的最大 token 数（默认 256）。越大越容易区分前缀但更难命中“共享模板”的复用。
- **`--trace-no-sync-cuda`**：默认会 `cuda.synchronize()` 来测更稳定的 step wall-time；若你只关心趋势，可关掉减少同步开销。

### RULER / LongBench v2 / ShareGPT

统一入口 `inference/run_dataset.py`：

```bash
python3 inference/run_dataset.py --ckpt-path "$CKPT_PATH" --config inference/config_671B_v3.2.json \\
  --dataset ruler --split train --limit 64 --batch-size 1 --max-new-tokens 64 \\
  --kv-block-size 16 --trace-out outputs/ruler_test
```

LongBench v2（固定 `zai-org/LongBench-v2`，split 为 `train`）：

```bash
python3 inference/run_dataset.py --ckpt-path "$CKPT_PATH" --config inference/config_671B_v3.2.json \\
  --dataset longbenchv2 --limit 64 --max-new-tokens 32 --trace-out outputs/longbenchv2_test
```

ShareGPT（默认使用可直接加载的 HF 数据集；或指定本地 ShareGPT JSON）：

```bash
python3 inference/run_dataset.py --ckpt-path "$CKPT_PATH" --config inference/config_671B_v3.2.json \\
  --dataset sharegpt --limit 64 --max-new-tokens 64 --trace-out outputs/sharegpt_test \\
  --sharegpt-dataset anon8231489123/ShareGPT_Vicuna_unfiltered --sharegpt-turn-mode per_user_turn

# 或本地 JSON（ShareGPT 格式）
python3 inference/run_dataset.py --ckpt-path "$CKPT_PATH" --config inference/config_671B_v3.2.json \\
  --dataset sharegpt --limit 64 --trace-out outputs/sharegpt_local \\
  --sharegpt-json /path/to/sharegpt.json
```

### BurstGPT（到达过程 / 并发 / tail）

```bash
python3 inference/run_burstgpt.py --ckpt-path "$CKPT_PATH" --config inference/config_671B_v3.2.json \\
  --burstgpt-csv data/burstgpt/burstgpt_train_limit2000.csv \\
  --limit 256 --batch-size 4 --max-new-tokens-cap 64 --trace-out outputs/burstgpt_test
```

## 已知限制与后续对齐点

- 当前没有真实的 vLLM PagedAttention，因此 **B 点 block_id 是逻辑近似**。  
- 当前没有外部 KV，因此 **C 点为 HBM-only**；`kv_fetch` 字段保持 tier 结构，便于后续接入 MemFabric/MemCache/Mooncake。  
- 当前没有真实 prefix cache，因此 **D 点是复用关系分析**（基于 prefix hash），用于回答“DSA 是否频繁触达共享前缀块/热点块”。  

## 常见问题排查（踩坑指南）

### 1) `ModuleNotFoundError: torch`

你当前环境没装 PyTorch。按你的 CUDA 版本安装对应 wheel（示例）：

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2) HuggingFace 下载失败 / 需要权限

- 先 `huggingface-cli login`
- 或设置环境变量 `HF_TOKEN`
- 有些数据集需要在网页上先点一次 “Agree and access”

### 3) 日志太大

这套插桩是 **每 step × 每 request × 每 layer**，在 61 层模型上日志会非常快变大。

- 先把 `--limit` 调小（比如 8/16）
- `--max-new-tokens` 调小（比如 16/32）
- 设 `--trace-sample-rate 0.1`
- 不要开启 `--trace-store-scores`（默认只写 stats）


