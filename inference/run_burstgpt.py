import csv
import json
import os
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import ModelArgs, Transformer
import trace as ds_trace
from generate import generate


@dataclass
class BurstRequest:
    t: float  # seconds from trace start
    req_tokens: int
    resp_tokens: int
    model: str


def _init_dist() -> Tuple[int, int, int]:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return world_size, rank, local_rank


def _resolve_config_path(p: str) -> str:
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


def _read_burstgpt_csv(path: str, limit: int) -> List[BurstRequest]:
    out: List[BurstRequest] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                out.append(
                    BurstRequest(
                        t=float(row.get("Timestamp") or row.get("timestamp") or row.get("time") or 0.0),
                        req_tokens=int(row.get("Request tokens") or row.get("request_tokens") or row.get("prompt_tokens") or 0),
                        resp_tokens=int(row.get("Response tokens") or row.get("response_tokens") or row.get("completion_tokens") or 0),
                        model=str(row.get("Model") or row.get("model") or ""),
                    )
                )
            except Exception:
                continue
            if limit > 0 and len(out) >= limit:
                break
    out.sort(key=lambda x: x.t)
    return out


def _make_prompt_tokens_fixed_length(tokenizer: Any, n_tokens: int) -> List[int]:
    # Deterministic, content-agnostic prompt to match token length.
    # Use a stable token id (encode a single character).
    base_ids = tokenizer.encode("a", add_special_tokens=False)
    tok = int(base_ids[0]) if base_ids else 0
    return [tok] * int(max(n_tokens, 1))


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 1:
        return float(sorted_vals[-1])
    idx = int(round((len(sorted_vals) - 1) * q))
    return float(sorted_vals[idx])


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--burstgpt-csv", type=str, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--max-new-tokens-cap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--trace-out", type=str, default="")
    parser.add_argument("--kv-block-size", type=int, default=64)
    parser.add_argument("--trace-store-scores", action="store_true")
    parser.add_argument("--trace-sample-rate", type=float, default=1.0)
    parser.add_argument("--trace-prefix-key-tokens", type=int, default=256)
    parser.add_argument("--trace-no-sync-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.config = _resolve_config_path(args.config)

    world_size, rank, _ = _init_dist()
    if rank != 0:
        global print
        print = lambda *_, **__: None
        import warnings
        warnings.filterwarnings("ignore")
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

    trace_out = args.trace_out or os.path.join("outputs", f"burstgpt_{int(time.time() * 1000)}")
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
    cfg = ds_trace.apply_env_overrides(cfg)
    tracer = ds_trace.init_tracer(cfg)
    tracer.set_run_meta(run_name=os.path.basename(trace_out.rstrip("/")), dataset="burstgpt")
    bytes_per_token = int((model.layers[0].attn.kv_cache.size(-1) * model.layers[0].attn.kv_cache.element_size()) +
                          (model.layers[0].attn.pe_cache.size(-1) * model.layers[0].attn.pe_cache.element_size()))
    os.environ["DS_TRACE_KV_BYTES_PER_TOKEN"] = str(bytes_per_token)

    reqs = _read_burstgpt_csv(args.burstgpt_csv, limit=int(args.limit))
    if rank == 0:
        print(f"[dataset] burstgpt: loaded {len(reqs)} requests from CSV (--limit={args.limit})", flush=True)
    prefix_analyzer = ds_trace.PrefixCacheAnalyzer(prefix_cache_key_tokens=args.trace_prefix_key_tokens)

    import sys as _sys
    now_t = 0.0
    i = 0
    queue: List[Tuple[int, BurstRequest]] = []
    latencies_ms: List[float] = []
    request_id = 0
    n_batches_done = 0

    while i < len(reqs) or queue:
        # Enqueue arrivals up to now_t.
        while i < len(reqs) and reqs[i].t <= now_t:
            queue.append((request_id, reqs[i]))
            request_id += 1
            i += 1
        if not queue:
            # Jump to next arrival.
            now_t = reqs[i].t
            continue

        batch = queue[: int(args.batch_size)]
        queue = queue[int(args.batch_size) :]
        batch_ids = [rid for rid, _ in batch]
        batch_prompt_tokens: List[List[int]] = []
        batch_prefix_infos: List[ds_trace.RequestPrefixInfo] = []
        max_new = 1
        for _, br in batch:
            ptoks = _make_prompt_tokens_fixed_length(tokenizer, br.req_tokens)
            batch_prompt_tokens.append(ptoks)
            batch_prefix_infos.append(prefix_analyzer.analyze_prompt_tokens(ptoks, kv_block_size_tokens=int(args.kv_block_size)))
            max_new = max(max_new, min(int(br.resp_tokens), int(args.max_new_tokens_cap)))

        if tracer.enabled and tracer.cfg.sync_cuda_for_timing:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        generate(
            model,
            batch_prompt_tokens,
            max_new_tokens=int(max_new),
            eos_id=tokenizer.eos_token_id,
            temperature=0.6,
            request_ids=batch_ids,
            prefix_infos=batch_prefix_infos,
        )
        if tracer.enabled and tracer.cfg.sync_cuda_for_timing:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        service_s = float(t1 - t0)

        # Assign same service time to all in batch; waiting time approximated by (now_t - arrival_t).
        for (rid, br) in batch:
            wait_s = max(0.0, now_t - br.t)
            latencies_ms.append((wait_s + service_s) * 1000.0)

        now_t += service_s
        n_batches_done += 1
        if rank == 0:
            _sys.stderr.write(f"\r[progress] batches={n_batches_done}  requests_done={len(latencies_ms)}/{len(reqs)}   ")
            _sys.stderr.flush()

    if rank == 0:
        _sys.stderr.write("\n")
        _sys.stderr.flush()
        print(f"[done] batches={n_batches_done}  requests_done={len(latencies_ms)}", flush=True)

    if rank == 0:
        latencies_ms.sort()
        summary = tracer.get_summary()
        summary["serving"] = {
            "requests": int(len(latencies_ms)),
            "latency_ms": {
                "p50": _quantile(latencies_ms, 0.5),
                "p95": _quantile(latencies_ms, 0.95),
                "p99": _quantile(latencies_ms, 0.99),
                "max": float(latencies_ms[-1]) if latencies_ms else 0.0,
            },
            "batch_size": int(args.batch_size),
            "max_new_tokens_cap": int(args.max_new_tokens_cap),
        }
        with open(os.path.join(tracer.trace_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[output] {tracer.trace_dir}  ({tracer.writer.total_records} records)")

    tracer.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

