import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
import tarfile
import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
from datasets.exceptions import DataFilesNotFoundError
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import ModelArgs, Transformer
import trace as ds_trace
from generate import generate


def _init_dist() -> Tuple[int, int, int]:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return world_size, rank, local_rank


def _resolve_config_path(p: str) -> str:
    """
    Resolve config path robustly across different working directories.
    If p is relative and not found, try relative to this file's directory.
    """
    if not p:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    base = Path(__file__).resolve().parent
    cand = base / p
    if cand.exists():
        return str(cand)
    return p


def _data_root() -> Path:
    # Prefer explicit env overrides; else infer from HF_HOME when scripts set it.
    for k in ("DATAS_DIR", "DATA_ROOT"):
        v = os.getenv(k)
        if v:
            return Path(v).resolve()
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).resolve().parent
    return (Path.cwd() / "data").resolve()


def _prepare_ruler_dataset(split: str, tgz_name: str) -> List[str]:
    """
    RULER HF dataset (allenai/ruler_data) ships as tgz archives, not directly loadable.
    We download + extract under data/ruler/ and return json/jsonl file paths.
    """
    data_root = _data_root()
    out_dir = data_root / "ruler"
    extract_dir = out_dir / f"extracted_{tgz_name.replace('.tgz','')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Download to HF hub cache (which scripts redirect under data/huggingface/hub).
    tgz_path = hf_hub_download(
        repo_id="allenai/ruler_data",
        filename=tgz_name,
        repo_type="dataset",
    )

    # Extract once (best-effort marker file).
    marker = extract_dir / ".extracted.ok"
    if not marker.exists():
        # Clean partial extraction (best effort) by re-extracting over directory.
        with tarfile.open(tgz_path, "r:gz") as tf:
            # Future-proof extraction behavior (Python 3.14+ changes default filter).
            try:
                tf.extractall(path=extract_dir, filter="data")
            except TypeError:
                tf.extractall(path=extract_dir)
        marker.write_text("ok\n", encoding="utf-8")

    # Collect candidate json/jsonl files.
    cand: List[str] = []
    for root, _, files in os.walk(extract_dir):
        for fn in files:
            low = fn.lower()
            if low.endswith(".jsonl") or low.endswith(".json"):
                cand.append(str(Path(root) / fn))

    # Prefer files matching split keyword if present.
    split = (split or "").lower()
    if split:
        split_hits = [p for p in cand if split in Path(p).name.lower()]
        if split_hits:
            cand = split_hits

    return sorted(cand)


def _prepare_sharegpt_json_file(repo_id: str, filename: str) -> str:
    """
    Download a ShareGPT-style JSON file from a dataset repo to local cache and return its path.
    This is used as a robust fallback when `load_dataset(repo_id)` can't infer data files.
    """
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")


def _ruler_prompt_from_raw(ex: Dict[str, Any]) -> str:
    """
    Normalize various RULER jsonl schemas into a single prompt string.
    We intentionally keep it simple since this runner is for instrumentation/trace.
    """
    for k in ("input", "prompt", "question", "instruction", "text"):
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # Common alternative schema: context + query.
    ctx = ex.get("context")
    q = ex.get("query") or ex.get("question")
    if isinstance(ctx, str) and ctx.strip() and isinstance(q, str) and q.strip():
        return f"Context:\n{ctx}\n\nQuestion:\n{q}\n\nAnswer:"
    return json.dumps(ex, ensure_ascii=False)


def _load_ruler_examples_as_input(files: Sequence[str], limit: int) -> List[Dict[str, str]]:
    """
    Load extracted RULER json/jsonl files and return a list of examples with a stable schema:
    {"input": <prompt_text>}.

    This avoids HuggingFace JSON builder schema-cast errors across heterogeneous RULER tasks.
    """
    out: List[Dict[str, str]] = []
    remaining = limit if limit > 0 else 0

    def _maybe_add(raw: Dict[str, Any]) -> None:
        nonlocal remaining
        if remaining == 0 and limit > 0:
            return
        prompt = _ruler_prompt_from_raw(raw)
        if not prompt:
            return
        out.append({"input": prompt})
        if limit > 0:
            remaining -= 1

    for fp in files:
        if limit > 0 and remaining <= 0:
            break
        low = fp.lower()
        if low.endswith(".jsonl"):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if limit > 0 and remaining <= 0:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(raw, dict):
                        _maybe_add(raw)
        elif low.endswith(".json"):
            try:
                raw = json.load(open(fp, "r", encoding="utf-8"))
            except Exception:
                continue
            if isinstance(raw, list):
                for item in raw:
                    if limit > 0 and remaining <= 0:
                        break
                    if isinstance(item, dict):
                        _maybe_add(item)
            elif isinstance(raw, dict):
                _maybe_add(raw)
    return out


def _infer_prompt_text(dataset: str, ex: Dict[str, Any]) -> str:
    # Keep heuristics simple and robust: fall back to concatenating string fields.
    if dataset == "longbenchv2":
        ctx = ex.get("context", "")
        q = ex.get("question", "")
        choices = []
        for k in ("choice_A", "choice_B", "choice_C", "choice_D"):
            if ex.get(k):
                choices.append(f"{k[-1]}. {ex[k]}")
        choice_text = ("\n".join(choices) + "\n") if choices else ""
        return f"Context:\n{ctx}\n\nQuestion:\n{q}\n\nOptions:\n{choice_text}\nAnswer:"
    if dataset == "ruler":
        for k in ("input", "prompt", "question", "instruction"):
            if isinstance(ex.get(k), str) and ex[k].strip():
                return ex[k]
        if isinstance(ex.get("text"), str) and ex["text"].strip():
            return ex["text"]
    # Generic fallback.
    parts: List[str] = []
    for k in sorted(ex.keys()):
        v = ex[k]
        if isinstance(v, str) and v.strip():
            parts.append(f"{k}:\n{v}")
    if parts:
        return "\n\n".join(parts)
    return json.dumps(ex, ensure_ascii=False)


def _sharegpt_messages(ex: Dict[str, Any]) -> List[Dict[str, str]]:
    conv = ex.get("conversations") or ex.get("conversation") or ex.get("messages")
    if not isinstance(conv, list):
        # Fallback: treat as single user prompt.
        return [{"role": "user", "content": _infer_prompt_text("sharegpt", ex)}]
    out: List[Dict[str, str]] = []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        frm = str(turn.get("from", turn.get("role", ""))).lower()
        val = turn.get("value", turn.get("content", ""))
        if not isinstance(val, str) or not val.strip():
            continue
        if frm in ("human", "user"):
            out.append({"role": "user", "content": val})
        elif frm in ("gpt", "assistant", "model"):
            out.append({"role": "assistant", "content": val})
    if not out:
        out = [{"role": "user", "content": _infer_prompt_text("sharegpt", ex)}]
    return out


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["ruler", "longbenchv2", "sharegpt"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--trace-out", type=str, default="")
    parser.add_argument("--kv-block-size", type=int, default=64)
    parser.add_argument("--trace-store-scores", action="store_true")
    parser.add_argument("--trace-sample-rate", type=float, default=1.0)
    parser.add_argument("--trace-prefix-key-tokens", type=int, default=256)
    parser.add_argument("--trace-no-sync-cuda", action="store_true")
    parser.add_argument("--max-records-per-file", type=int, default=10000)
    parser.add_argument("--max-prompt-tokens", type=int, default=0)
    parser.add_argument("--chat-system-prompt", type=str, default="")
    parser.add_argument("--sharegpt-json", type=str, default="")
    parser.add_argument("--sharegpt-dataset", type=str, default="anon8231489123/ShareGPT_Vicuna_unfiltered")
    parser.add_argument("--sharegpt-hf-file", type=str, default="ShareGPT_V3_unfiltered_cleaned_split.json")
    parser.add_argument("--sharegpt-turn-mode", type=str, default="full", choices=["full", "per_user_turn"])
    parser.add_argument("--ruler-tgz", type=str, default="data_debug.tgz", choices=["data_debug.tgz", "data_100_samples.tgz"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.config = _resolve_config_path(args.config)

    world_size, rank, _ = _init_dist()
    if rank != 0:
        global print
        print = lambda *_, **__: None
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335 + args.seed)

    with open(args.config) as f:
        margs = ModelArgs(**json.load(f))
    if int(args.batch_size) > margs.max_batch_size:
        margs.max_batch_size = int(args.batch_size)
        if rank == 0:
            print(f"[warn] --batch-size ({args.batch_size}) > config max_batch_size; overriding to {args.batch_size}")
    with torch.device("cuda"):
        model = Transformer(margs)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    load_model(model, os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    trace_out = args.trace_out or os.path.join("outputs", f"{args.dataset}_{int(time.time() * 1000)}")
    cfg = ds_trace.TraceConfig(
        enabled=True,
        out_dir=trace_out,
        kv_block_size_tokens=int(args.kv_block_size),
        store_scores_topk=bool(args.trace_store_scores),
        store_selected_token_pos=True,
        sample_rate=float(args.trace_sample_rate),
        rank0_only=True,
        sync_cuda_for_timing=not bool(args.trace_no_sync_cuda),
        prefix_cache_key_tokens=int(args.trace_prefix_key_tokens),
        max_records_per_file=int(args.max_records_per_file),
    )
    tracer = ds_trace.init_tracer(cfg)
    tracer.set_run_meta(run_name=os.path.basename(trace_out.rstrip("/")), dataset=args.dataset)
    bytes_per_token = int((model.layers[0].attn.kv_cache.size(-1) * model.layers[0].attn.kv_cache.element_size()) +
                          (model.layers[0].attn.pe_cache.size(-1) * model.layers[0].attn.pe_cache.element_size()))
    os.environ["DS_TRACE_KV_BYTES_PER_TOKEN"] = str(bytes_per_token)

    prefix_analyzer = ds_trace.PrefixCacheAnalyzer(prefix_cache_key_tokens=args.trace_prefix_key_tokens)

    total_dataset_size: Optional[int] = None
    selected_size: Optional[int] = None

    if args.dataset == "ruler":
        # allenai/ruler_data ships tgz archives; extract under repo data/ and load json/jsonl.
        if rank == 0:
            files = _prepare_ruler_dataset(args.split, args.ruler_tgz)
        else:
            files = []
        if world_size > 1 and dist.is_initialized():
            obj_list = [files]
            dist.broadcast_object_list(obj_list, src=0)
            files = obj_list[0]
            dist.barrier()
        if not files:
            raise RuntimeError("RULER dataset files not found after extraction. Try --ruler-tgz data_100_samples.tgz.")
        # Load and normalize locally to avoid heterogeneous-schema cast errors.
        all_examples = _load_ruler_examples_as_input(files, limit=0)
        total_dataset_size = len(all_examples)
        if args.limit > 0:
            all_examples = all_examples[:int(args.limit)]
        selected_size = len(all_examples)
        ds_iter: Iterable[Dict[str, Any]] = all_examples
    elif args.dataset == "longbenchv2":
        # HF LongBench v2 is commonly published as zai-org/LongBench-v2 with train split only.
        ds = load_dataset("zai-org/LongBench-v2", split="train")
        total_dataset_size = len(ds)
        if args.limit > 0:
            ds = ds.select(range(min(args.limit, len(ds))))
        selected_size = len(ds)
        ds_iter = ds
    else:
        if args.sharegpt_json:
            ds_iter = load_dataset("json", data_files=args.sharegpt_json, split="train", streaming=True)
        else:
            try:
                ds = load_dataset(args.sharegpt_dataset, split=args.split)
                total_dataset_size = len(ds)
                if args.limit > 0:
                    ds = ds.select(range(min(args.limit, len(ds))))
                selected_size = len(ds)
                ds_iter = ds
            except DataFilesNotFoundError:
                if rank == 0:
                    sharegpt_path = _prepare_sharegpt_json_file(args.sharegpt_dataset, args.sharegpt_hf_file)
                else:
                    sharegpt_path = ""
                if world_size > 1 and dist.is_initialized():
                    obj_list = [sharegpt_path]
                    dist.broadcast_object_list(obj_list, src=0)
                    sharegpt_path = obj_list[0]
                    dist.barrier()
                if not sharegpt_path:
                    raise RuntimeError("ShareGPT fallback download failed; please provide --sharegpt-json.")
                ds_iter = load_dataset("json", data_files=sharegpt_path, split="train", streaming=True)

    if total_dataset_size is not None:
        print(f"[dataset] {args.dataset}: total={total_dataset_size}, selected={selected_size} (--limit={args.limit})")
    else:
        print(f"[dataset] {args.dataset}: streaming mode (--limit={args.limit})")

    max_prompt_tokens = int(args.max_prompt_tokens)
    if max_prompt_tokens <= 0:
        max_prompt_tokens = int(model.max_seq_len - args.max_new_tokens)

    batch_prompt_tokens: List[List[int]] = []
    batch_request_ids: List[int] = []
    batch_prefix_infos: List[ds_trace.RequestPrefixInfo] = []
    next_request_id = 0
    n_samples_seen = 0
    n_skipped_long = 0

    def _flush_batch() -> None:
        nonlocal batch_prompt_tokens, batch_request_ids, batch_prefix_infos
        if not batch_prompt_tokens:
            return
        generate(
            model,
            batch_prompt_tokens,
            max_new_tokens=int(args.max_new_tokens),
            eos_id=tokenizer.eos_token_id,
            temperature=float(args.temperature),
            request_ids=batch_request_ids,
            prefix_infos=batch_prefix_infos,
        )
        batch_prompt_tokens = []
        batch_request_ids = []
        batch_prefix_infos = []

    system_prompt = (args.chat_system_prompt or "").strip()

    def _enqueue_request(prompt_toks: List[int]) -> None:
        nonlocal next_request_id, n_skipped_long
        if len(prompt_toks) > max_prompt_tokens:
            n_skipped_long += 1
            return
        pinfo = prefix_analyzer.analyze_prompt_tokens(prompt_toks, kv_block_size_tokens=int(args.kv_block_size))
        batch_prompt_tokens.append(prompt_toks)
        batch_request_ids.append(next_request_id)
        batch_prefix_infos.append(pinfo)
        next_request_id += 1
        if len(batch_prompt_tokens) >= int(args.batch_size):
            _flush_batch()

    def _print_progress() -> None:
        label = f"[progress] samples={n_samples_seen}"
        if selected_size is not None:
            label += f"/{selected_size}"
        label += f"  requests={next_request_id}  skipped_long={n_skipped_long}"
        sys.stderr.write(f"\r{label}")
        sys.stderr.flush()

    if args.dataset == "sharegpt" and args.limit > 0 and not hasattr(ds_iter, "select"):
        ds_iter = itertools.islice(ds_iter, int(args.limit))

    for ex in ds_iter:
        n_samples_seen += 1
        if n_samples_seen % 1 == 0:
            _print_progress()
        if args.dataset == "sharegpt":
            messages = _sharegpt_messages(ex)
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
            if args.sharegpt_turn_mode == "full":
                ptoks = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                _enqueue_request(ptoks)
            else:
                # Emit one request per user turn to mimic multi-round serving.
                hist: List[Dict[str, str]] = []
                for m in messages:
                    hist.append(m)
                    if m["role"] == "user":
                        ptoks = tokenizer.apply_chat_template(hist, add_generation_prompt=True)
                        _enqueue_request(ptoks)
        else:
            prompt_text = _infer_prompt_text(args.dataset, ex)
            msgs = [{"role": "user", "content": prompt_text}]
            if system_prompt:
                msgs = [{"role": "system", "content": system_prompt}] + msgs
            ptoks = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
            _enqueue_request(ptoks)
    _flush_batch()
    sys.stderr.write("\n")
    print(f"[done] samples_seen={n_samples_seen}  requests_generated={next_request_id}  skipped_too_long={n_skipped_long}")

    if rank == 0:
        summary = tracer.get_summary()
        summary_path = os.path.join(tracer.trace_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[output] {tracer.trace_dir}  ({tracer.writer.total_records} records)")
    tracer.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

