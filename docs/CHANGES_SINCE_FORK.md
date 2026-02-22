# 本仓库相对 fork 基线的完整增量指南

**目标读者**：需要在 vLLM / SGLang 等真实 serving 引擎做类似 DSA 插桩的人。

> **fork 基线**：commit `87e509a`（最初推理 demo/README，插桩前的最后一个上游 commit）。
> 从 `0e8d126` 起开始插桩改造，至当前 `HEAD` 共约 4300 行增量。

---

## 0. 一句话总结

我们把 DeepSeek 官方 demo（只能交互聊天）改成了**可跑 4 类 workload、可复现、可配置 schema 的 trace 采集管线**——但**没有改动模型的推理逻辑本身**（所有改动对推理结果透明）。

---

## 1. 完整文件变更清单

### 1.1 新增文件（fork 基线中不存在）

| 文件 | 行数 | 作用 |
|---|---:|---|
| `inference/trace.py` | ~775 | **插桩核心**：TraceConfig / Tracer / TraceWriter / PrefixCacheAnalyzer / apply_env_overrides |
| `inference/run_dataset.py` | ~477 | 数据集 runner：RULER / LongBench-v2 / ShareGPT |
| `inference/run_burstgpt.py` | ~245 | BurstGPT 到达过程模拟 runner |
| `inference/sanity_trace_no_torch.py` | ~130 | 不依赖 torch 的 schema 自检脚本 |
| `scripts/run_trace_ruler.sh` | ~77 | RULER trace 脚本 |
| `scripts/run_trace_longbenchv2.sh` | ~75 | LongBench-v2 trace 脚本 |
| `scripts/run_trace_sharegpt.sh` | ~87 | ShareGPT trace 脚本 |
| `scripts/run_trace_burstgpt.sh` | ~74 | BurstGPT trace 脚本 |
| `scripts/run_all_traces.sh` | ~162 | 一键跑全部 4 类 workload |
| `scripts/run_trace_chunked.sh` | ~163 | 大规模跑数时分 chunk 重启（崩溃恢复） |
| `scripts/datasets/download_ruler.sh` | ~45 | 下载 RULER tgz |
| `scripts/datasets/download_longbench_v2.sh` | ~38 | 下载 LongBench-v2 |
| `scripts/datasets/download_sharegpt.sh` | ~43 | 下载 ShareGPT |
| `scripts/datasets/download_burstgpt.sh` | ~59 | 导出 BurstGPT CSV |
| `docs/INSTRUMENTATION.md` | ~914 | 完整插桩参考（怎么跑/schema/字段/troubleshoot） |
| `docs/SYSTEM_WORKLOAD_DATASETS.md` | ~214 | 论文 → 数据集精确对照表 |
| `docs/KV_BLOCK_SIZE.md` | ~71 | 各引擎常见 block/page size |
| `docs/HADAMARD_TRANSFORM.md` | ~173 | Hadamard 变换实现分析 |
| `.gitignore` | 14 | 忽略 outputs/data/logs/local/__pycache__ 等 |
| `analysis/analyze_trace.py` | ~509 | trace 离线分析脚本（dispersion / locality / cache sim） |
| `analysis/trace_utils.py` | ~N | 分析辅助函数 |

### 1.2 修改文件（在 fork 基线上改动）

| 文件 | 改动要点 |
|---|---|
| `inference/model.py` | **A 点插桩**：Indexer.forward() decode 路径记录 top-k；MLA/Block 传 layer_id；Hadamard 回退实现 |
| `inference/generate.py` | 生成 loop 加 request_ids/prefix_infos/step timing；trace init/close；config 路径解析；max_new_tokens<=0 语义 |
| `inference/config_671B_v3.2.json` | 新增 `max_batch_size:1`、`max_seq_len:16384`（控制显存预分配） |
| `inference/kernel.py` | 注释掉 `TL_DISABLE_FAST_MATH` |
| `inference/requirements.txt` | 新增 `datasets`、`huggingface_hub` |
| `inference/README.md` | 增加 pip 安装指引、环境变量示例、trace quickstart |
| `README.md` | 新增 "DSA Trace Instrumentation" 段落，链接到 docs/ |

---

## 2. 四个插桩点的精确代码位置

### 2.1 插桩点 A：DSA top-k 选择输出

**文件**：`inference/model.py` → `Indexer.forward()`

**精确位置**（约 L519-L537）：

```python
k = min(self.index_topk, end_pos)
tracer = ds_trace.get_tracer()
trace_decode_only = (mask is None and seqlen == 1)
if tracer.enabled and trace_decode_only:
    topk_values, topk_indices = index_score.topk(k, dim=-1)
else:
    topk_values = None
    topk_indices = index_score.topk(k, dim=-1)[1]
# ... broadcast & assert ...
if tracer.enabled and trace_decode_only:
    tracer.record_dsa_topk(
        layer_id=int(self._trace_layer_id),
        start_pos=int(start_pos),
        end_pos=int(end_pos),
        topk_indices=topk_indices,
        topk_scores=topk_values,
    )
```

**关键设计决策**：
- **只在 decode 阶段记录**（`mask is None and seqlen == 1`）—— prefill 阶段 indexer 也会运行，但 DSA sparse selection 只在 decode 时有意义
- `topk_values` 只在 trace 启用时才保留（否则丢弃以省显存）
- `layer_id` 通过 `MLA.__init__` → `self.indexer._trace_layer_id` 传入

**辅助改动**（也在 model.py）：

- `Block.__init__` 把 `layer_id` 传给 `MLA`（约 L876）
- `MLA.__init__` 接收 `layer_id` 并赋值给 `self.indexer._trace_layer_id`（约 L589）
- 新增 `_hadamard_transform_pytorch()` 作为 `fast_hadamard_transform` 的纯 PyTorch fallback

### 2.2 插桩点 B/C/D：trace.py 的 `Tracer.record_dsa_topk()`

**文件**：`inference/trace.py` → `Tracer.record_dsa_topk()`（约 L475-L650）

这个函数接收 A 点的 `topk_indices` 和 `topk_scores`，在内部完成 B/C/D：

- **B（token→block 映射）**：`block_id = token_pos // kv_block_size_tokens`，然后计算 `unique_blocks`、`tokens_per_touched_block`、`touched_block_ratio` 等统计
- **C（KV fetch 代价）**：HBM-only 占位，按 `bytes_per_token × block数` 估算 `bytes_read`
- **D（prefix cache 交互）**：如果 `enable_prefix_analysis=True`，计算 DSA touched blocks ∩ prefix cached blocks 的交集比例

### 2.3 生成 loop 的 timing 接入

**文件**：`inference/generate.py` → `generate()`（约 L82-L108）

```python
tracer = ds_trace.get_tracer()
if tracer.enabled:
    tracer.set_batch(request_ids=request_ids, prefix_infos=prefix_infos)
for cur_pos in range(min(prompt_lens), total_len):
    if tracer.enabled:
        tracer.set_step_timing(step_idx=prev_pos, step_wall_us=None)
        if tracer.cfg.sync_cuda_for_timing:
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
    logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
    if tracer.enabled:
        if tracer.cfg.sync_cuda_for_timing:
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        tracer.set_step_timing(step_idx=prev_pos, step_wall_us=(t1 - t0) // 1000)
```

---

## 3. trace.py 核心架构

### 3.1 类结构

```
TraceConfig          数据类，集中管理所有 trace 行为和 schema 开关
  ├─ from_env()      从环境变量构建（用于 interactive/generate.py）
  └─ 各 record_* 布尔值控制哪些字段写入 JSONL

apply_env_overrides(cfg)  对显式构建的 TraceConfig 叠加环境变量覆盖

TraceWriter          异步 JSONL writer
  ├─ 后台线程 + queue（防 NCCL timeout）
  └─ 分片策略：shard_id = request_id // max_requests_per_file（确定性分桶）

Tracer               全局单例，持有 TraceWriter + 在线统计
  ├─ record_dsa_topk()    A/B/C/D 统一入口
  ├─ set_batch()          设置当前 batch 的 request_ids 和 prefix_infos
  ├─ set_step_timing()    填充 step wall-time
  ├─ get_summary()        返回汇总统计 dict
  ├─ _write_run_meta()    写 run_meta.json（固定元信息，不每行重复）
  └─ close()              排空队列 + 关闭文件 + 重命名最后一片

PrefixCacheAnalyzer  基于 prompt 前缀 hash 的近似 prefix cache 分析器
```

### 3.2 环境变量完整表

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `DS_TRACE_ENABLE` | 0 | 全局开关（`generate.py` 交互模式用） |
| `DS_TRACE_OUT` | 空 | 输出目录 |
| `DS_TRACE_KV_BLOCK_SIZE` | 64 | 逻辑 block 大小（tokens） |
| `DS_TRACE_STORE_SCORES` | 0 | 写全部 2048 分数 |
| `DS_TRACE_STORE_TOKEN_POS` | 1 | 写 selected_token_pos 列表 |
| `DS_TRACE_SAMPLE_RATE` | 1.0 | step 采样率 |
| `DS_TRACE_RANK0_ONLY` | 1 | 只 rank0 写 trace |
| `DS_TRACE_SYNC_CUDA` | 1 | 测 step wall-time 前 sync |
| `DS_TRACE_MAX_REQUESTS_PER_FILE` | 4 | 每文件最多几个 request |
| `DS_TRACE_RECORD_META_PER_RECORD` | 0 | 每行重复 event/run_name/rank 等 |
| `DS_TRACE_RECORD_BLOCK_IDS` | 0 | 写 selected_block_ids / hit_blocks |
| `DS_TRACE_RECORD_KV_FETCH` | 1 | 写 kv_fetch 字段 |
| `DS_TRACE_RECORD_KV_FETCH_LATENCY_US` | 0 | 写 latency_us |
| `DS_TRACE_RECORD_KV_FETCH_READ_OPS` | 0 | 写 read_ops |
| `DS_TRACE_RECORD_EMPTY_TIERS` | 0 | 写空的 local_pool/remote_pool |
| `DS_TRACE_KV_BYTES_PER_TOKEN` | 自动 | 每 token KV 字节数（由 runner 设置） |

### 3.3 输出结构

```
outputs/<run>/block64/
├── run_meta.json                 固定元信息 + schema 配置
├── trace_steps_0_4.jsonl         request 0~3 的全部记录
├── trace_steps_4_8.jsonl         request 4~7
├── trace_steps_8_11.jsonl        request 8~10（最后一片，实际 end）
└── summary.json                  在线汇总统计
```

---

## 4. 数据集 runner 做了什么（以及为什么需要这些 hack）

### 4.1 `run_dataset.py`（RULER / LongBench-v2 / ShareGPT）

| 数据集 | 难点 / hack | 代码位置 |
|---|---|---|
| **RULER** | HF 上是 `.tgz`，非标准格式；不同 task 的 JSONL schema 不一致 | `_prepare_ruler_dataset()` + `_load_ruler_examples_as_input()`：下载 tgz → 解压 → 手动 JSON/JSONL 解析 → 统一成 `{"input": prompt}` |
| **LongBench-v2** | dataset id 是 `zai-org/LongBench-v2`（非 `THUDM/`），split 只有 `train` | 直接 `load_dataset`，修正了 id 和 split |
| **ShareGPT** | `openchat/openchat_sharegpt_v3` 不兼容 `load_dataset` | 改用 `anon8231489123/ShareGPT_Vicuna_unfiltered` 或本地 JSON；支持 `full`/`per_user_turn` 两种多轮模式 |

### 4.2 `run_burstgpt.py`

- 读取 BurstGPT CSV（timestamp + request/response tokens）
- 用"固定 token 长度"合成 prompt（不关心内容，只关心长度 → DSA 行为）
- 按 timestamp 模拟到达排队和 batch 服务
- 输出 serving latency 统计（p50/p95/p99）到 `summary.json`

---

## 5. 从 demo 迁移到 vLLM / SGLang 的对照表

### 5.1 你应该直接复用什么

| 模块 | 复用建议 |
|---|---|
| `TraceConfig` + `apply_env_overrides` | **直接复用**。schema 开关和环境变量控制在任何引擎都适用 |
| `TraceWriter`（异步分片） | **直接复用**。多 GPU 写盘的异步/分片/分桶逻辑是通用的 |
| `run_meta.json` + JSONL schema | **直接复用**。分析脚本期望这个 schema |
| 数据集 runner（`run_dataset.py` / `run_burstgpt.py`） | **改造复用**。把 `generate()` 调用替换成 vLLM/SGLang 的 API |
| `analysis/` 分析脚本 | **直接复用** |

### 5.2 你需要替换什么（逐插桩点）

#### A 点（DSA top-k 输出）

| demo 实现 | vLLM/SGLang 应改成 |
|---|---|
| `Indexer.forward()` 里 `index_score.topk()` 后直接调 `tracer.record_dsa_topk()` | 在引擎的 DeepSeek indexer kernel（DeepGEMM `fp8_mqa_logits` → topk）输出处 hook |

**伪代码模式**（vLLM 为例）：

```python
# 在 vLLM 的 DeepSeek attention layer 的 decode 路径：
topk_indices = indexer_topk(...)  # 引擎实际的 top-k 计算
# --- 插桩点 A ---
if tracer.enabled:
    tracer.record_dsa_topk(
        layer_id=layer_idx,
        start_pos=current_pos,
        end_pos=current_pos + 1,
        topk_indices=topk_indices,
        topk_scores=topk_values,  # 可选
    )
```

#### B 点（token→block 映射）

| demo 实现 | vLLM/SGLang 应改成 |
|---|---|
| `pos // block_size`（近似） | 查 block table：`block_table[seq_id][pos // block_size]` 得到**物理 block id** |

**建议同时记录两种 id**（用于不同分析目的）：

```python
logical_block_id = token_pos // block_size          # 与 demo 一致，便于对比
physical_block_id = block_table[seq_id][logical_block_id]  # 真实物理页号
```

#### C 点（KV fetch 代价）

| demo 实现 | vLLM/SGLang 应改成 |
|---|---|
| HBM-only 占位（`bytes_read = blocks × block_size × bytes_per_token`） | 在 block/page 被从某个 tier 拉取的路径埋点 |

vLLM offload / lmcache / NIXL 等场景下，应记录：

```python
kv_fetch = {
    "hbm":        {"hit_blocks": [...], "bytes_read": N, "latency_us": T},
    "local_pool": {"hit_blocks": [...], "bytes_read": N, "latency_us": T},
    "remote_pool":{"hit_blocks": [...], "bytes_read": N, "latency_us": T},
}
```

如果只有 GPU KV（无 offload/remote），C 点意义不大，保留 HBM-only 占位即可。

#### D 点（prefix cache 交互）

| demo 实现 | vLLM/SGLang 应改成 |
|---|---|
| hash 近似（`PrefixCacheAnalyzer`） | 引擎真实 prefix cache 命中时记录复用的 block/page 列表 |

**vLLM**：prefix caching 的 block reuse 管理器在新 request 到来时做匹配，你应拿到：
- `prefix_cache_hit: bool`
- `prefix_cached_blocks: int`（或 block 列表）
- 然后在每条 A 点记录里计算 `|touched ∩ prefix| / |touched|`

**SGLang**：RadixAttention 本身就是 KV 复用结构，在 radix tree 的 match/insert 处记录命中长度与对应 pages 即可。

---

## 6. 迁移时的坑位清单

### 6.1 多 GPU + 写盘 = NCCL timeout

- **不要在关键路径同步写大文件**
- 本仓库的异步 writer（后台线程 + queue）是实测验证过的方案：rank0 把 JSON 序列化好放入 queue，后台线程持续 flush
- 如果换成 mmap/ring buffer 也行，关键是**不阻塞 GPU compute 线程**

### 6.2 分片必须按 request_id 确定性分桶

batch decode 下记录会交错产生（同一 step 里多个 request × 多个 layer 的记录混着写）：

- **错误做法**：按"见到 N 个 unique request 就 rotate 文件"→ 同一 request 会被切开
- **正确做法**：`shard_id = request_id // N`（本仓库当前实现）

### 6.3 prefill 路径和 decode 路径要分开对待

- 在 demo 里，我们只在 decode（`seqlen==1`）时记录 —— prefill 阶段 indexer 也跑，但 DSA sparse selection 只对 decode 有意义
- 在 vLLM/SGLang 里，prefill 和 decode 通常已经是不同的 kernel / code path，hook 时要注意只 hook decode

### 6.4 "模型能跑 160K" ≠ "你的 prefill 能跑 160K"

- demo 的 prefill 是 dense attention（O(S²) 显存），长 prompt 会 OOM
- vLLM/SGLang 的 prefill 通常高效很多，但 top-k logits materialization 在长上下文时仍可能是显存热点

### 6.5 request_id 生命周期

- 在 vLLM/SGLang 里，request 有引擎分配的 `seq_id` / `request_id`，但它们可能在 preemption/migration 后变化
- 建议在 trace 里使用**你自己分配的 stable request_id**（例如输入顺序）而非引擎内部 id

---

## 7. 与其他文档的关系（建议阅读顺序）

| 序号 | 文档 | 内容 |
|---|---|---|
| 1 | **本文档**（`docs/CHANGES_SINCE_FORK.md`） | 改了什么、在哪里、为什么、怎么迁移 |
| 2 | `docs/INSTRUMENTATION.md` | 怎么跑、输出是什么、字段含义、troubleshoot（最全最长） |
| 3 | `docs/SYSTEM_WORKLOAD_DATASETS.md` | 四类 workload 与论文的精确对照关系 |
| 4 | `docs/KV_BLOCK_SIZE.md` | 不同引擎常见 block/page size |
| 5 | `docs/HADAMARD_TRANSFORM.md` | Hadamard 变换实现分析（与 trace 无直接关系，解释模型细节） |
