# 系统论文评测用 Workload / 数据集对照（含变体）

本文档用于把“**系统论文里到底用的哪个 workload / 哪个数据集变体**”记录清楚，精确到：

- Hugging Face **dataset ID**
- 具体 **文件名**（如果论文 / artifact / 官方 benchmark 写死了某个 JSON/CSV）
- **split / subset**
- 是否是 **trace replay**（是否带 timestamp）或 **合成到达过程**（Poisson/Gamma）

同时给出与本仓库脚本（`scripts/` + `inference/`）的对应关系，方便复现实验与对齐指标。

---

## 1. 常见 Workload 形态速览

- **真实到达过程（trace replay）**：每条请求带 `timestamp`，严格按时间回放。典型：BurstGPT、Mooncake traces。
- **无 timestamp 的离线数据集**：只有 prompt/response 或长度统计，需要用 **Poisson / Gamma** 合成到达过程。典型：ShareGPT、LongBench、HumanEval。
- **合成数据（synthetic prompt）**：完全由脚本生成（控制输入/输出长度、共享前缀比例、burstiness 等），用于可控压测与 ablation。典型：vLLM `random`/`prefix_repetition`，SGLang `generated-shared-prefix`。

---

## 2. 论文 → 具体数据集/变体（强约束清单）

### 2.1 Mooncake（FAST'25 / arXiv: `2407.00079`）

**开源 trace 文件（GitHub）**：`kvcache-ai/Mooncake`

- FAST'25 traces 目录：`FAST25-release/traces/`
  - `conversation_trace.jsonl`
  - `synthetic_trace.jsonl`
  - `toolagent_trace.jsonl`
- 早期 arxiv trace：`FAST25-release/arxiv-trace/mooncake_trace.jsonl`

**trace schema（论文描述 + 文件一致）**

- 字段：`timestamp`, `input_length`, `output_length`, `hash_ids`
- `hash_ids`：用于描述 prefix caching 复用关系；论文说明其来自对 token block（**block size=512**）的 hash + remap

**参考链接**

- 论文：`https://arxiv.org/abs/2407.00079`
- traces 目录：`https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces`

---

### 2.2 Sarathi-Serve（OSDI'24 / arXiv: `2403.02310`）

论文明确使用两套“长度分布来源”，并用 Poisson 生成到达时间（因为数据本身无 timestamp）：

- **openchat_sharegpt4（ChatGPT-4 对话）**
  - HF：`openchat/openchat_sharegpt4_dataset`
- **arxiv_summarization（论文/摘要 summarization）**
  - 常用 HF：`ccdv/arxiv-summarization`

**到达过程**

- Poisson arrival（论文写明）

**参考链接**

- 论文：`https://arxiv.org/abs/2403.02310`
- openchat sharegpt4：`https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset`
- arxiv summarization：`https://huggingface.co/datasets/ccdv/arxiv-summarization`

---

### 2.3 DistServe（OSDI'24 / arXiv: `2401.09670`）

DistServe 的 artifact 把下载链接写死到具体文件名（强约束，最适合做“变体对齐”）：

- **Chatbot：ShareGPT**
  - 文件：`ShareGPT_V3_unfiltered_cleaned_split.json`
  - 下载：`https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json`
- **Code completion：HumanEval**
  - 文件：`HumanEval.jsonl`（由 `HumanEval.jsonl.gz` 解压）
  - 下载：`https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz`
- **Summarization：LongBench**
  - 下载：`https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip?download=true`（解压后目录名 `longbench/`）

**参考链接（artifact 文档）**

- `repro-dataset.md`：`https://raw.githubusercontent.com/LLMServe/DistServe/main/evaluation/docs/repro-dataset.md`

---

### 2.4 Llumnix（OSDI'24 / arXiv: `2406.03243`）

论文使用“真实对话数据的长度分布”+“合成到达过程”的混合评测：

- **ShareGPT (GPT-4)**：
  - HF：`shibing624/sharegpt_gpt4`
- **BurstGPT (GPT4-Conversation)**：
  - 论文引用 BurstGPT 的 GPT-4 conversation 子集（用于长度分布/trace）；实际公开数据以 `HPMLL/BurstGPT` 为主（CSV 里区分 GPT-3.5 vs GPT-4、Conversation log vs API log）
- **额外合成长度分布**：power-law 长尾（Short/Medium/Long），用于模拟“短交互 + 少量长文档”的混合场景

**到达过程**

- Poisson & Gamma；Gamma 通过 CV 调节 burstiness（论文写明）

**参考链接**

- 论文：`https://arxiv.org/abs/2406.03243`
- ShareGPT GPT4：`https://huggingface.co/datasets/shibing624/sharegpt_gpt4`
- BurstGPT：`https://github.com/HPMLL/BurstGPT`

---

### 2.5 LMCache（arXiv: `2510.09665`）

LMCache 论文/tech report 的公开可对齐 workload 信息主要是：

- **Long-context QA**：来自 **LongBench 的 TriviaQA 任务**（文中直接写 “TriviaQA dataset from LongBench”）
- **random workload**：来自 vLLM 官方 benchmarking 的 `random` 数据集（合成输入/输出长度）
- **real trace**：来自企业（Company F/G）私有 trace（不可公开复现，只能对齐其公开 workload 描述）

**到达过程**

- 文中说明遵循 vLLM 官方 benchmark：按指定 QPS 生成（Poisson arrival）

**参考链接**

- 论文：`https://arxiv.org/abs/2510.09665`
- tech report：`https://lmcache.ai/tech_report.pdf`

---

## 3. vLLM 与 SGLang 的“合成数据 / 合成到达过程”

### 3.1 vLLM（`vllm bench serve`）

**支持的数据集类型（包含 synthetic）**

- `random`（合成文本长度）
- `prefix_repetition`（合成重复前缀，适合评测 prefix cache）
- 以及 `sharegpt`, `burstgpt`, `sonnet`, `spec_bench`, `hf` 等

**合成到达过程**

- `--request-rate` 非 `inf` 时：使用 **Poisson** 或 **Gamma**（由 `--burstiness` 控制）来生成请求到达间隔（vLLM 文档写明）

**ShareGPT 具体变体（vLLM 官方文档给出 wget）**

- 文件：`ShareGPT_V3_unfiltered_cleaned_split.json`
- 数据集：`anon8231489123/ShareGPT_Vicuna_unfiltered`

参考链接：

- `vllm bench serve --help`：`https://docs.vllm.ai/en/stable/cli/bench/serve/`
- vLLM benchmarking CLI（含 ShareGPT wget 表）：`https://docs.vllm.ai/en/latest/benchmarking/cli/`

---

### 3.2 SGLang（`python -m sglang.bench_serving`）

**支持的 synthetic 数据集**

- `random` / `random-ids`
- `generated-shared-prefix`（合成“共享长 system prompt”的前缀复用场景）

**支持 Mooncake trace replay**

- `--dataset-name mooncake`
- `--mooncake-workload {conversation,synthetic,toolagent,mooncake}`
- 并支持用 trace timestamps 做调度（time-based scheduler；此时会忽略 `--request-rate`）

参考链接：

- SGLang bench_serving 指南：`https://sgl-project.github.io/developer_guide/bench_serving.html`

---

## 4. 与本仓库（DeepSeek-V3.2-Exp demo）脚本的对应关系

本仓库当前面向“插桩采集 token→block 访问轨迹”，选的数据源与系统论文常用集合**高度重合**：

- **RULER**
  - HF：`allenai/ruler_data`
  - 具体变体文件：`data_debug.tgz` / `data_100_samples.tgz`
  - 脚本：`scripts/datasets/download_ruler.sh` + `scripts/run_trace_ruler.sh`
- **LongBench v2**
  - HF：`zai-org/LongBench-v2`（也可用 `THUDM/LongBench-v2` 的 `data.json` 链接）
  - 脚本：`scripts/datasets/download_longbench_v2.sh` + `scripts/run_trace_longbenchv2.sh`
- **ShareGPT（默认 Vicuna_unfiltered 变体）**
  - 目标文件：`ShareGPT_V3_unfiltered_cleaned_split.json`
  - 脚本：`scripts/datasets/download_sharegpt.sh` + `scripts/run_trace_sharegpt.sh`
- **BurstGPT（真实到达过程）**
  - 来源：`HPMLL/BurstGPT`（建议用 release CSV）
  - 脚本：`scripts/datasets/download_burstgpt.sh` + `scripts/run_trace_burstgpt.sh`

**补充：如何对齐 Sarathi-Serve / Llumnix 的 ShareGPT 变体**

- Sarathi-Serve：`openchat/openchat_sharegpt4_dataset`（GPT-4 对话）
- Llumnix：`shibing624/sharegpt_gpt4`（GPT-4 对话）

本仓库默认使用的是 Vicuna_unfiltered 的 ShareGPT（更常见于 serving benchmark，如 vLLM/DistServe），如果你要严格对齐 GPT-4 对话分布，可以把 `scripts/datasets/download_sharegpt.sh` 的默认下载源替换为上述 HF dataset（或新增一个下载脚本）。

---

## 5. 备注：为什么“精确到变体”很重要

同叫 “ShareGPT”，不同论文/系统可能用的是：

- `anon8231489123/ShareGPT_Vicuna_unfiltered` 的 `ShareGPT_V3_unfiltered_cleaned_split.json`（系统 benchmark 最常见）
- `openchat/openchat_sharegpt4_dataset`（更偏 “GPT-4 高质量过滤对话”）
- `shibing624/sharegpt_gpt4`（Llumnix 明确引用）

这些变体会显著影响：

- prompt/output 长度分布
- 多轮会话结构（prefix reuse 强弱）
- 触发 KV cache / prefix cache 的命中率与热点

因此，做系统对比时建议固定到具体 dataset ID + 具体文件名（如果存在）。

