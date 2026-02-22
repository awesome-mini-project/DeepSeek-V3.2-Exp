# 本仓库相对 fork 基线的增量指南（面向后续迁移到 vLLM / SGLang 插桩）

这份文档回答两个问题：

1. **当前仓库相对最初 fork 的版本，新增了什么？改动在哪里？**
2. **如果未来要在 vLLM / SGLang 等真实 serving 引擎做同样的插桩（A/B/C/D），应该照着哪里改、怎么迁移？**

> 说明：本文把“fork 基线”定义为 commit `87e509a`（在本 repo 历史中，最早一批推理 demo/README 修订之后、我们开始做 trace 插桩之前的状态）。
> 从 `87e509a` 到当前 `HEAD` 的主要改动由 commit `0e8d126` 起逐步引入。

---

## 1. 一句话总结：我们把“单机 demo”改造成了“可跑四类 workload 的可复现 trace 采集管线”

相对 fork 基线，本仓库新增/强化了：

- **插桩核心**：`inference/trace.py`（TraceConfig/Tracer/TraceWriter/PrefixCacheAnalyzer）
  - 异步写盘（避免 NCCL timeout）
  - JSONL 按 request 分片（并保证同一 request 不跨文件）
  - `run_meta.json`（把重复元信息从每条 JSONL 中移走）
  - 可通过环境变量控制 trace 字段是否写出（block ids、kv_fetch latency、read_ops、空 tiers 等）
- **插桩点 A 的接入**：在 `inference/model.py` 的 `Indexer.forward()` decode 路径记录 top-k
- **插桩点 B/C/D 的统一落点**：在 `Tracer.record_dsa_topk()` 内完成 token→block 映射、KV fetch 占位统计、prefix 交集统计
- **四类数据集 runner**：
  - `inference/run_dataset.py`：RULER / LongBench-v2 / ShareGPT
  - `inference/run_burstgpt.py`：BurstGPT 到达过程模拟
- **脚本化跑数与复现**：`scripts/run_trace_*.sh`、`scripts/run_all_traces.sh`、`scripts/run_trace_chunked.sh`、`scripts/datasets/download_*.sh`
- **文档化**：`docs/INSTRUMENTATION.md`（完整参考）、`docs/SYSTEM_WORKLOAD_DATASETS.md`、`docs/KV_BLOCK_SIZE.md`、`docs/HADAMARD_TRANSFORM.md`
- **稳定性改造**：
  - 避免长 prompt prefill OOM 的参数/过滤（`MAX_PROMPT_TOKENS`、`max_seq_len` 说明）
  - 分片与 chunked runner 用于崩溃恢复

---

## 2. 代码改动清单（按文件/模块）

下面按“你需要去哪里改什么”来列出增量内容。

### 2.1 插桩核心：`inference/trace.py`

**新增模块**（fork 基线没有）：

- **`TraceConfig`**：集中管理 trace 行为与 schema 开关
  - 既支持显式传参，也支持环境变量覆盖（例如 `DS_TRACE_RECORD_BLOCK_IDS=1`）
- **`Tracer`**：把插桩点 A/B/C/D 的信息统一组织成 JSONL 记录，并做在线汇总统计（输出 `summary.json`）
- **`TraceWriter`**：异步 JSONL writer
  - **关键修复**：分片按 `request_id` 做确定性分桶（避免 batch decode 写入交错导致同一 request 跨文件）
- **`PrefixCacheAnalyzer`**：仅用于“分析 prefix 复用关系”的近似实现（不做真实 KV 复用）

**输出形态（新增/变更）**：

- `outputs/<run>/block{N}/trace_steps_{start}_{end}.jsonl`
- `outputs/<run>/block{N}/run_meta.json`：固定元信息与 schema 配置（减少 JSONL 重复字段）
- `outputs/<run>/block{N}/summary.json`：在线汇总统计（unique_blocks、offset、tokens_per_touched_block、prefix 交集等）

**为什么这样设计（给迁移到 vLLM/SGLang 的启发）**：

- **把“记录格式/写盘/分片/统计”与“模型/引擎内部 hook 点”解耦**：
  - 迁移到 vLLM/SGLang 时，你可以复用 `TraceConfig/TraceWriter/Tracer` 的绝大部分逻辑
  - 只需替换/补齐：如何拿到 top-k（A）、如何做 token→block 映射（B）、如何拿到分层 fetch 代价（C）、如何拿到真实 prefix cache 命中与 block 列表（D）
- **异步写盘是必须项**：
  - 多 GPU 多进程下，rank0 同步 I/O 会造成其他 rank 在 barrier/collective 处超时

### 2.2 插桩点 A：`inference/model.py`（Indexer.forward）

**我们做了什么**：

- 在 `Indexer.forward()` 中捕获：
  - `index_score.topk(k, dim=-1)` 的 `topk_indices`（以及可选 `topk_values`）
- 只在 **decode 阶段**记录（`mask is None and seqlen == 1`）
- 通过给 `MLA/Indexer` 传 `layer_id`，让 trace 记录能标注层号

**迁移到 vLLM/SGLang 的要点**：

- A 点本质是：**“在 sparse attention 的 indexer/selector 输出处”拿到 top-k token positions**  
  vLLM/SGLang 里对应的模块/函数名会不同，但“topk 输出”是最稳定的 hook 位置。

### 2.3 插桩点 B/C/D：`Tracer.record_dsa_topk()`（`inference/trace.py`）

在 demo-only 插桩里，我们把 B/C/D 都放在同一个函数里做，是因为 demo 的 KV cache 不是 paged/block manager。

- **B（token→block）**：用近似映射 `block_id = token_pos // kv_block_size_tokens`
  - 该 `kv_block_size_tokens` 由 `--kv-block-size`/`KV_BLOCK_SIZE` 控制
  - 真实 vLLM/SGLang 会有“逻辑块→物理块”的映射表；迁移时建议额外记录物理块（如果你研究跨设备迁移/复用）
- **C（KV fetch 代价）**：当前实现是 **HBM-only 占位**
  - 默认只写 `bytes_read/batch_size`，延迟/read_ops/hit_blocks 默认不写（可用 env 开）
  - 迁移到真实系统时，应把这里替换成真实的 tier hit / latency / bytes / ops
- **D（prefix cache 交互）**：当前实现是“近似分析”
  - 真实系统（vLLM prefix caching / SGLang RadixAttention）应在 cache 命中时能拿到复用的 blocks/pages

### 2.4 生成循环插桩：`inference/generate.py`

**我们新增/修改**：

- 生成 loop 增加：
  - `request_ids`（让 batch 内每条 request 有稳定 id）
  - `prefix_infos`（每 request 的 prefix 分析结果）
- 记录 step wall-time（可选写入 `kv_fetch.hbm.latency_us`；默认关闭）
- 语义改造：`max_new_tokens <= 0` 表示“直到 EOS（或 max_seq_len）”

迁移到 vLLM/SGLang 时，生成 loop 往往不在 Python，而在 engine/scheduler；但你仍需要一个“**request_id 生命周期**”来关联所有记录。

### 2.5 数据集与 workload runner

#### `inference/run_dataset.py`（RULER / LongBench-v2 / ShareGPT）

新增能力：

- **RULER**：HF 上是 `.tgz`，且不同 task JSONL schema 不一致  
  → runner 做了“下载 + 解压 + 手动解析 + 统一成 `{"input": prompt}`”来避开 datasets cast error
- **LongBench-v2**：修正 HF dataset id（`zai-org/LongBench-v2`）与 split
- **ShareGPT**：支持可 load 的 ShareGPT 派生集，并支持本地 JSON
- **运行稳定性**：
  - 进度打印（rank0-only）
  - `--max-prompt-tokens` 过滤避免 prefill OOM
  - `--batch-size` > config 的 `max_batch_size` 时自动扩大预分配（避免因 config 太小崩溃）
  - `--max-requests-per-file` 默认 4（并由脚本 `MAX_REQUESTS_PER_FILE` 透传）

#### `inference/run_burstgpt.py`（BurstGPT）

新增能力：

- 读取 BurstGPT CSV（来自 `scripts/datasets/download_burstgpt.sh` 导出）
- 构造“固定 token 长度”的合成 prompt，模拟到达过程与 batch 服务
- 输出服务延迟统计到 `summary.json`

### 2.6 脚本与工程化

新增：

- `scripts/run_trace_{ruler,longbenchv2,sharegpt,burstgpt}.sh`
  - 默认 `MP=8`
  - 强制 HF cache 目录到 repo `data/huggingface/`（避免污染 `~/.cache`）
  - 透传 `MAX_REQUESTS_PER_FILE`
- `scripts/run_all_traces.sh`
  - `DATA`（样本规模）与 `GEN`（生成长度）两个维度
  - `GEN=full` 时自动 `MAX_REQUESTS_PER_FILE=1`（自然生成 trace 体积最大）
  - `CONTINUE_ON_ERROR=1` 可跳过失败继续
- `scripts/run_trace_chunked.sh`
  - 大规模跑数时“分 chunk 重启”以做崩溃恢复
- `scripts/datasets/download_*.sh`
  - 下载 RULER/LongBench-v2/ShareGPT/BurstGPT 所需数据并复用缓存

---

## 3. 插桩点 A/B/C/D：从 demo 迁移到 vLLM / SGLang 的“对照表”

### 3.1 你应该复用什么？

强烈建议复用本仓库的：

- `inference/trace.py` 的 **TraceWriter/分片/异步 I/O/run_meta.json/schema 开关**
- JSONL schema（尤其是 `request_id/step_idx/layer_id/selected_token_pos` 这条主线）

迁移时主要替换：

- A 点：top-k 输出来源（从 demo 的 `index_score.topk` → vLLM/SGLang 的 indexer kernel 输出）
- B 点：token→block 映射（从 `pos//block_size` → 引擎真实的 block table / page table）
- C 点：KV fetch 代价（从 HBM-only 占位 → 真实 tiered fetch/offload/remote latency）
- D 点：prefix cache 命中（从 hash 近似 → 引擎真实 prefix cache / radix tree 命中与复用 blocks）

### 3.2 迁移到 vLLM：建议 hook 点（思路）

> 这里不写具体文件名/行号（因为 vLLM 版本迭代快），只给“应该在什么层次 hook”。

- **A（DSA top-k）**：
  - 在 vLLM 的 DeepSeek DSA indexer 计算完 top-k indices 的地方
  - decode 阶段：每个新 token 都会产生一次 top-k
  - 需要拿到：`seq_id/request_id`、`layer_id`、`token_pos range`、`topk_indices`（可选 `scores`）
- **B（token→block）**：
  - vLLM 有真实 paged KV cache 的 block table（逻辑块→物理块）
  - 建议记录两种 block id：
    - `logical_block_id`（pos//block_size）
    - `physical_block_id`（block_table 映射的物理页号；如果你研究搬运/远端缓存，这个更关键）
- **C（分层 KV fetch）**：
  - 如果只用 GPU KV（无 offload/remote），C 点意义不大
  - 一旦启用 KV offload / 外部 KV（lmcache / NIXL / 自研 MemFabric），应在“block/page 被拉取/拷贝/预取”的路径埋点：
    - tier hit（HBM/local/remote）
    - bytes/ops
    - latency（队列等待/传输/拷贝分解若可得）
- **D（prefix cache）**：
  - vLLM prefix caching / block reuse 的管理器会在新 request 到来时做匹配
  - 建议拿到：命中的 prefix blocks/pages 列表（或至少长度/块数），以及与 A 点 touched blocks 的交集

### 3.3 迁移到 SGLang：建议 hook 点（思路）

SGLang 的关键差异是 **RadixAttention（radix tree）**：

- **D（prefix cache）**在 SGLang 往往更“原生”：radix tree 本身就是跨 request 的 KV 复用结构
  - 你应优先在 radix tree 的 match/insert/evict 处记录 prefix 命中长度与对应 blocks/pages
- **B（block/page）**仍然存在：KV 存储仍有页表/块管理器（具体实现视版本）
- **A（top-k）**取决于你是否在 SGLang 跑 DeepSeek DSA 及其 indexer kernel；hook 思路与 vLLM 类似
- **C（分层 fetch）**同样取决于是否启用 offload/remote 或外部 KV

---

## 4. 迁移时的“坑位清单”（这也是本仓库为何要做这些工程化）

### 4.1 多 GPU + 写盘 = NCCL timeout

如果你在 vLLM/SGLang 里做 trace：

- **不要在关键路径同步写大文件**
- 建议复用我们这里的异步 writer 设计，或用 mmap/缓冲区 + 后台 flush

### 4.2 分片必须按 request_id 确定性分桶

batch decode 下记录会交错产生：

- 如果你按“见到 N 个 request 就 rotate”，同一 request 会被切开
- 正确做法：**`shard_id = request_id // N`**（本仓库已实现）

### 4.3 “模型能跑 160K”不等于“你当前实现的 prefill 能跑 160K”

本仓库 demo 的 prefill 是 dense attention，长 prompt 会 OOM。我们通过：

- 配置与过滤（`max_seq_len`、`MAX_PROMPT_TOKENS`）让 trace 跑得稳

迁移到 vLLM/SGLang 时，prefill 通常是高效实现，但仍要注意：

- top-k / logits materialization / mask layout 可能在长上下文时成为显存热点

---

## 5. 与其他文档的关系（建议阅读顺序）

1. `docs/INSTRUMENTATION.md`：怎么跑、输出是什么、字段含义（最全）
2. `docs/SYSTEM_WORKLOAD_DATASETS.md`：四类 workload 与论文对应关系
3. `docs/KV_BLOCK_SIZE.md`：不同引擎常见 block/page size（tokens）
4. `docs/HADAMARD_TRANSFORM.md`：Hadamard transform 的实现分析（与 trace 无直接关系，但解释了模型实现细节）

