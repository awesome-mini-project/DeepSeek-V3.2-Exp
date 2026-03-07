# TRACE.md - DSA Trace 插桩系统详解 (ASCII 版本)

## 目录

- [1. 概述](#1-概述)
- [2. TraceConfig 配置](#2-traceconfig-配置)
- [3. Tracer 类](#3-tracer-类)
- [4] (TraceWriter 异步写入](#4-tracewriter-异步写入)
- [5. PrefixCacheAnalyzer](#5-prefixcacheanalyzer)
- [6. 使用方法](#6-使用方法)

## 1. 概述

`trace.py` 实现了 DSA (DeepSeek Sparse Attention) 的 trace 插桩系统，用于：
- 记录每次推理的 Top-K 选择
- 分析稀疏注意力的访问模式
- 测量推理性能
- 支持前缀缓存分析

```
generate.py (生成入口)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Trace 插桩系统                           │
├─────────────────────────────────────────────────────────────┤
│  TraceConfig  │  Tracer  │  TraceWriter  │  PrefixCacheAnalyzer │
│  配置管理      │  追踪器  │  异步写入     │  前缀缓存分析       │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
输出: JSONL trace 文件
```

## 2. TraceConfig 配置

### 2.1 数据类定义

**位置**: `trace.py:L72-L90`

```python
@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = False
    out_dir: str = ""
    kv_block_size_tokens: int = 64
    store_scores_topk: bool = False
    store_selected_token_pos: bool = True
    sample_rate: float = 1.0
    rank0_only: bool = True
    sync_cuda_for_timing: bool = True
    # Output schema controls
    record_meta_per_record: bool = False
    record_block_ids: bool = False
    record_kv_fetch: bool = True
    record_kv_fetch_latency_us: bool = False
    record_kv_fetch_read_ops: bool = False
    record_empty_tiers: bool = False
    enable_prefix_analysis: bool = False
    prefix_cache_key_tokens: int = 256
    max_requests_per_file: int = 4
```

### 2.2 配置参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | False | 是否启用 trace |
| `out_dir` | "" | 输出目录 |
| `kv_block_size_tokens` | 64 | KV cache 块大小（token 数） |
| `store_scores_topk` | False | 是否存储 Top-K 分数 |
| `store_selected_token_pos` | True | 是否存储选中的 token 位置 |
| `sample_rate` | 1.0 | 采样率 (0-1) |
| `rank0_only` | True | 仅在 rank 0 上记录 |
| `sync_cuda_for_timing` | True | 是否同步 CUDA 以精确计时 |
| `enable_prefix_analysis` | False | 是否启用前缀分析 |
| `prefix_cache_key_tokens` | 256 | 前缀缓存 key 的 token 数 |

### 2.3 环境变量配置

| 环境变量 | 说明 |
|----------|------|
| `DS_TRACE_ENABLE` | 启用 trace |
| `DS_TRACE_OUT` | 输出目录 |
| `DS_TRACE_KV_BLOCK_SIZE` | KV 块大小 |
| `DS_TRACE_STORE_SCORES` | 存储分数 |
| `DS_TRACE_SAMPLE_RATE` | 采样率 |
| `DS_TRACE_RANK0_ONLY` | 仅 rank 0 记录 |
| `DS_TRACE_SYNC_CUDA` | 同步 CUDA |
| `DS_TRACE_PREFIX_KEY_TOKENS` | 前缀 key token 数 |

## 3. Tracer 类

### 3.1 核心功能

Tracer 负责在推理过程中收集 DSA 相关信息：

```
初始化 (set_run_meta)
      │
      ▼
┌───────────────────────────────────────────────────────────────┐
│                     请求级别追踪                          │
├───────────────────────────────────────────────────────────────┤
│  begin_request: 记录请求开始                               │
│  set_step_timing: 记录每步时间                             │
│  record_dsa_topk: 记录 Top-K 选择                           │
│  end_request: 记录请求结束                                 │
└───────────────────────────────────────────────────────────────┘
      │
      ▼
提交到 TraceWriter (异步写入)
```

### 3.2 主要方法

#### begin_request

```python
def begin_request(
    self,
    request_id: int,
    prompt_tokens: int,
    prompt_hash: Optional[int],
    prefix_info: Optional[RequestPrefixInfo],
):
```

**功能**：
- 初始化请求级别的追踪信息
- 记录 prompt token 数
- 记录 prompt hash（用于去重）
- 记录前缀缓存信息

#### set_step_timing

```python
def set_step_timing(
    self,
    step_idx: int,
    step_wall_us: Optional[int],
):
```

**功能**：
- 记录每个生成步骤的时间
- `step_wall_us`: 该步骤的墙钟时间（微秒）

#### record_dsa_topk

```python
def record_dsa_topk(
    self,
    step_idx: int,
    topk_indices: torch.Tensor,  # (B, S, K)
):
```

**功能**：
- 记录 DSA Indexer 的 Top-K 选择结果
- 分析稀疏访问模式

## 4. TraceWriter 异步写入

### 4.1 功能

TraceWriter 负责将 trace 数据异步写入文件，避免阻塞推理：

```
Tracer 记录数据
      │
      ▼
写入内存队列
      │
      ▼
后台线程
      │
      ▼
批量写入 JSONL 文件
      │
      ▼
分片 (每 N 个请求)
      │
      ▼
输出目录/
    ├─── trace_0001.jsonl
    ├─── trace_0002.jsonl
    └─── ...
```

### 4.2 输出格式

每条 trace 记录包含：

```json
{
  "version": 1,
  "run_meta": {
    "run_name": "experiment_1",
    "dataset": "interactive",
    "model": "DeepSeek-V3.2-Exp",
    "timestamp_ms": 1234567890
  },
  "request": {
    "request_id": 0,
    "prompt_tokens": 10,
    "generated_tokens": 50,
    "prompt_hash": "abc123"
  },
  "steps": [
    {
      "step_idx": 0,
      "wall_time_us": 1234,
      "dsa_topk_indices": [...],
      "kv_fetch": {...}
    }
  ]
}
```

## 5. PrefixCacheAnalyzer

### 5.1 功能

PrefixCacheAnalyzer 分析前缀缓存效率：

```
请求序列
    │
    ▼
计算 prefix key
    │
    ▼
查找已缓存块
    │
    ├─────► 命中 ──► 复用缓存块
    │
    └───► 未命中 ──► 计算并缓存
```

### 5.2 RequestPrefixInfo

```python
@dataclass
class RequestPrefixInfo:
    prefix_cache_hit: bool       # 是否命中缓存
    prefix_cached_blocks: int    # 命中的块数
    prefix_key: str              # 前缀 key
```

## 6. 使用方法

### 6.1 启用 Trace

```bash
# 设置环境变量
export DS_TRACE_ENABLE=true
export DS_TRACE_OUT=outputs/trace_$(date +%s)
export DS_TRACE_KV_BLOCK_SIZE=16
export DS_TRACE_SYNC_CUDA=true

# 运行推理
python generate.py --ckpt_path /path/to/ckpt --config config.json
```

### 6.2 在代码中集成

```python
import trace as ds_trace

# 初始化配置
cfg = ds_trace.TraceConfig(
    enabled=True,
    out_dir="outputs/trace",
    kv_block_size_tokens=16,
    store_scores_topkw=True,
    sample_rate=0.1,
)

# 应用环境变量覆盖
cfg = ds_trace.apply_env_overrides(cfg)

# 创建 tracer
tracer = ds_trace.init_tracer(cfg)

# 设置运行元数据
tracer.set_run_meta(
    run_name="experiment_1",
    dataset="sharegpt",
)

# 在生成循环中使用
for req_id, tokens in enumerate(requests):
    tracer.begin_request(req_id, len(tokens), ...)
    for step in generation_loop:
        # ... 模型推理 ...
        tracer.set_step_timing(step_idx, time_us)
        tracer.record_dsa_topk(step_idx, topk_indices)
    tracer.end_request()
```

### 6.3 分析 Trace 输出

```python
import json

# 读取 trace 文件
with open("outputs/trace_1234567890/trace_0001.jsonl", "r") as f:
    for line in f:
        record = json.loads(line)

        # 分析请求
        print(f"Request ID: {record['request']['request_id']}")
        print(f"Generated tokens: {record['request']['generated_tokens']}")

        # 分析步骤
        for step in record['steps']:
            print(f"Step {step['step_idx']}: {step['wall_time_us']} us")
            # 分析 Top-K 选择
            topk = step['dsa_topk_indices']
            print(f"Top-K 位置: {topk[:10]}...")
```

### 6.4 可视化分析

```
Trace 数据
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                    分析维度                              │
├─────────────────────────────────────────────────────────────┤
│  1. Top-K 选择分布 (每个位置选择了哪些 cache block)          │
│  2. 推理时间分析 (每 step 的延迟)                            │
│  3. 前缀缓存命中率                                           │
│   4. KV Cache 访问模式                                        │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
可视化图表 (matplotlib/grafana)
```

## 7. 性能影响

### 7.1 开销分析

| 组件 | 开销 | 说明 |
|------|------|------|
| Trace 记录 | < 1% | 轻量级操作 |
| 同步 CUDA | ~5-10% | 仅计时需要 |
| 异步写入 | <1% | 后台线程 |
| Top-K 记录 | <1% | CPU 上收集 |

### 7.2 采样率控制

使用 `sample_rate` 减少 trace 数量：

```python
cfg = TraceConfig(
    enabled=True,
    sample_rate=0.1,  # 仅记录 10% 的请求
)
```

---

**ASCII 文档导航**

| 文档 | 内容 |
|------|------|
| [OVERVIEW.md](OVERVIEW.md) | 项目总览 |
| [KERNEL.md](KERNEL.md) | FP8 算子内核 |
| [MODEL_BASE.md](MODEL_BASE.md) | 模型配置与基础类 |
| [MODEL_LINEAR.md](MODEL_LINEAR.md) | 线性层与嵌入层 |
| [MODEL_NORM.md](MODEL_NORM.md) | 归一化层 |
| [MODEL_ROPE.md](MODEL_ROPE.md) | 旋转位置编码 |
| [MODEL_HADAMARD.md](MODEL_HADAMARD.md) | Hadamard 变换 |
| [MODEL_INDEXER.md](MODEL_INDEXER.md) | DSA Indexer 模块 |
| [MODEL_MLA.md](MODEL_MLA.md) | MLA 注意力模块 |
| [MODEL_MOE.md](MODEL_MOE.md) | 混合专家系统 |
| [MODEL_MLP.md](MODEL_MLP.md) | 前馈网络 |
| [MODEL_BLOCK.md](MODEL_BLOCK.md) | Transformer Block |
| [MODEL_TRANSFORMER.md](MODEL_TRANSFORMER.md) | 完整 Transformer |
| [GENERATE.md](GENERATE.md) | 生成循环 |
| [CONVERT.md](CONVERT.md) | 权重格式转换 |
| [TRACE.md](TRACE.md) | 插桩系统 |
