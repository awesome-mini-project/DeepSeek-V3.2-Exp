import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
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
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--trace-store-scores", action="store_true")
    parser.add_argument("--trace-sample-rate", type=float, default=1.0)
    parser.add_argument("--trace-prefix-key-tokens", type=int, default=256)
    parser.add_argument("--trace-no-sync-cuda", action="store_true")
    parser.add_argument("--max-prompt-tokens", type=int, default=0)
    parser.add_argument("--chat-system-prompt", type=str, default="")
    parser.add_argument("--sharegpt-json", type=str, default="")
    parser.add_argument("--sharegpt-dataset", type=str, default="anon8231489123/ShareGPT_Vicuna_unfiltered")
    parser.add_argument("--sharegpt-turn-mode", type=str, default="full", choices=["full", "per_user_turn"])
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
    )
    tracer = ds_trace.init_tracer(cfg)
    tracer.set_run_meta(run_name=os.path.basename(trace_out.rstrip("/")), dataset=args.dataset)
    bytes_per_token = int((model.layers[0].attn.kv_cache.size(-1) * model.layers[0].attn.kv_cache.element_size()) +
                          (model.layers[0].attn.pe_cache.size(-1) * model.layers[0].attn.pe_cache.element_size()))
    os.environ["DS_TRACE_KV_BYTES_PER_TOKEN"] = str(bytes_per_token)

    prefix_analyzer = ds_trace.PrefixCacheAnalyzer(prefix_cache_key_tokens=args.trace_prefix_key_tokens)

    if args.dataset == "ruler":
        ds = load_dataset("allenai/ruler_data", split=args.split)
    elif args.dataset == "longbenchv2":
        # HF LongBench v2 is commonly published as zai-org/LongBench-v2 with train split only.
        ds = load_dataset("zai-org/LongBench-v2", split="train")
    else:
        if args.sharegpt_json:
            ds = load_dataset("json", data_files=args.sharegpt_json, split="train")
        else:
            ds = load_dataset(args.sharegpt_dataset, split=args.split)

    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    max_prompt_tokens = int(args.max_prompt_tokens)
    if max_prompt_tokens <= 0:
        max_prompt_tokens = int(model.max_seq_len - args.max_new_tokens)

    batch_prompt_tokens: List[List[int]] = []
    batch_request_ids: List[int] = []
    batch_prefix_infos: List[ds_trace.RequestPrefixInfo] = []
    next_request_id = 0

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
        nonlocal next_request_id
        if len(prompt_toks) > max_prompt_tokens:
            return
        pinfo = prefix_analyzer.analyze_prompt_tokens(prompt_toks, kv_block_size_tokens=int(args.kv_block_size))
        batch_prompt_tokens.append(prompt_toks)
        batch_request_ids.append(next_request_id)
        batch_prefix_infos.append(pinfo)
        next_request_id += 1
        if len(batch_prompt_tokens) >= int(args.batch_size):
            _flush_batch()

    for ex in ds:
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

    if rank == 0:
        summary_path = os.path.join(trace_out, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(tracer.get_summary(), f, ensure_ascii=False, indent=2)
    tracer.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

