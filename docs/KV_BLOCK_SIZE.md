# KV block/page size（tokens_per_block）常见取值整理

这份笔记回答一个常见问题：**“KV block size 到底应该设多少？”**

结论先说：**这通常是 serving 引擎的参数，而不是模型参数。**同一个模型（例如 LLaMA/Qwen/DeepSeek）在不同引擎里，常见的 block/page size（按 token）可能不同。

---

## 1. 名词对齐：block_size / page_size / tokens_per_block 是一回事吗？

不同项目命名不同，但通常都表示同一个概念：

- **每个 KV cache block/page 能容纳的 token 数**（token granularity 的“分页大小”）
- 分页的目标是：降低碎片、支持动态长度、支持复用/换入换出等

本仓库 `--kv-block-size` 是为了**把 token→block 映射**（block_id = token_pos // block_size）做统计，并不是真实 paged-KV 的物理布局。

---

## 2. 常见 serving 引擎的默认值 / 常用值（按 tokens）

> 下面表格刻意用“引擎/库”而不是“模型”，因为该参数通常是引擎级别的。

| 引擎/库 | 参数名 | 默认值 | 常见取值 | 约束/备注 | 资料 |
|---|---|---:|---|---|---|
| **vLLM（旧版本，v0.6.0）** | `--block-size` | **16** | 8 / 16 / 32 | 文档明确：可选 8/16/32，默认 16 | vLLM v0.6.0 Engine Args 文档（`--block-size`） |
| **vLLM（新版本）** | `CacheConfig.block_size` / `--block-size` | *无静态默认* | （CUDA 常见 8/16/32） | 文档说明：**无静态默认**，未指定时由 `Platform.check_and_update_config()` 按平台决定；并注明 CUDA 上仅支持到 32 | vLLM cache config API 文档 |
| **vLLM（DeepSeek-V3.2/3.2-Exp）** | `--block-size` | **64（固定）** | 64 | vLLM 博文明确说明：该模型的 indexer key cache 按 per-block 存储，且 FlashMLA 针对 64 优化，因此“**only support block size 64**” | vLLM Blog：DeepSeek-V3.2-Exp |
| **FlashInfer（paged KV）** | `page_size` | *用户指定* | 常见 16（示例） | 文档说明：`page_size` 是“每页容纳的 token 数”；示例代码中 `page_size = 16` | FlashInfer KV layout / append_paged_kv_cache 示例 |
| **TensorRT-LLM（KV cache system）** | `tokens_per_block`（概念） | *用户指定* | 常见 16/32/64（实践） | 官方文档说明：每个 block 存固定 token 数，且**必须是 >1 的 2 的幂**；内存 FAQ 日志示例出现 “Number of tokens per block: 64.” | TensorRT-LLM KV cache system / memory FAQ |

---

## 3. 为什么常见是 16、32、64？

这是工程权衡的结果（不同引擎/核实现会偏好不同值）：

- **更小的 block（例如 8/16）**：
  - 优点：减少末尾碎片（最后一页浪费更小），短请求更省
  - 缺点：页表/元数据开销更大；有的 kernel 更难做高效向量化
- **更大的 block（例如 32/64）**：
  - 优点：元数据更少；更利于某些 kernel 的访存/向量化；也常用于“按 block 复用/迁移”的策略单元
  - 缺点：碎片变大（尤其短请求），在高度不均匀长度下可能浪费更多

---

## 4. 对本仓库 trace 的建议

因为我们只是做 **token→block 访问集合统计**（不是真实 paged KV），所以选择 block size 的主要影响是：

- block id 的粒度（更小→更“分散”）
- `unique_blocks` / `tokens_per_touched_block` 的数值分布会随 block size 变化

建议：

- **默认 64**：与我们当前文档/脚本默认保持一致，且更接近“以 block 为迁移/复用单元”的系统研究口径
- 需要对比时，可以跑多组 `KV_BLOCK_SIZE=16/32/64`，离线统一分析趋势

---

## 5. 参考链接

- vLLM v0.6.0 Engine Args（`--block-size` 默认 16，choices=8/16/32）：`https://docs.vllm.ai/en/v0.6.0/models/engine_args.html`
- vLLM CacheConfig（无静态默认；CUDA <=32）：`https://docs.vllm.ai/en/v0.16.0/api/vllm/config/cache/`
- vLLM Blog（DeepSeek-V3.2-Exp 仅支持 block size 64）：`https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html`
- FlashInfer KV layout（page_size 定义）：`https://docs.flashinfer.ai/tutorials/kv_layout.html`
- FlashInfer append_paged_kv_cache 示例（page_size=16）：`https://docs.flashinfer.ai/generated/flashinfer.page.append_paged_kv_cache.html`
- TensorRT-LLM KV cache system（tokens_per_block 必须是 2 的幂）：`https://nvidia.github.io/TensorRT-LLM/features/kvcache.html`
- TensorRT-LLM memory FAQ（日志示例含 tokens per block=64）：`https://nvidia.github.io/TensorRT-LLM/reference/memory.html`

