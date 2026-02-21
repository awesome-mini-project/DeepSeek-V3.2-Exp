## 目标与范围

本仓库当前只包含 `inference/` 的推理 demo（不是 vLLM 本体）。因此本次“插桩采数”实现的是：

- **A (DSA top-2048)**：在 `Indexer.forward()` 的 top-k 选择输出处记录每个 decode step、每个 request 的 `selected_token_pos`（可选分数）与轻量统计。
- **B (token→block)**：当前没有 PagedAttention block table，所以用**逻辑块近似**：`block_id = token_pos // kv_block_size_tokens`。
- **C (KV 取数代价)**：实现一个**可插拔的 tier 统计接口**；现阶段输出 **HBM-only**（local/remote=0），`bytes_read/read_ops/latency_us/batch_size` 可用于后续接外部 KV。
- **D (prefix cache 交互)**：不做 KV 复用，只做“复用关系分析”并输出 `prefix_cache_hit/prefix_cached_blocks` 与 `touched_blocks ∩ prefix_blocks` 的比例和热点。

这些日志用于后续把真实的外部 KV / prefix cache 接入 vLLM 时做对齐与扩展。

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

## `/clear` 语义（你问的“每轮对话结束是否 clear”）

- `inference/generate.py --interactive` 的 `/clear` 只影响交互模式：它会清空 `messages`，相当于“新会话”。\n
- **dataset runner（批处理）默认行为**：每个样本/每个 request 都是独立的（等价于“样本之间都 clear”）。\n
- **ShareGPT 的多轮对话**：在 `run_dataset.py` 里提供两种模式：\n
  - `--sharegpt-turn-mode full`（默认）：整段对话作为一个 request（一次性把历史喂给模型并生成结尾）。\n
  - `--sharegpt-turn-mode per_user_turn`：每个 user turn 作为一个 request（同一段对话内部不 clear；不同对话之间 clear）。这更贴近 serving 中“多轮对话”的请求形态，也更容易在 D 点看到 prefix 交集热点。\n

## 如何跑四类数据集

### 依赖

`inference/requirements.txt` 增加了 `datasets`。建议使用 `python3`。

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

- 当前没有真实的 vLLM PagedAttention，因此 **B 点 block_id 是逻辑近似**。\n
- 当前没有外部 KV，因此 **C 点为 HBM-only**；`kv_fetch` 字段保持 tier 结构，便于后续接入 MemFabric/MemCache/Mooncake。\n
- 当前没有真实 prefix cache，因此 **D 点是复用关系分析**（基于 prefix hash），用于回答“DSA 是否频繁触达共享前缀块/热点块”。\n

