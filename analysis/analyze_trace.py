#!/usr/bin/env python3
"""
Trace analysis utilities for KV placement research.

This script reads JSONL trace records produced by inference/trace.py and computes
metrics that are useful for KV cache placement policies:
  - token-to-block dispersion and concentration (unique blocks, entropy, gini)
  - temporal locality (Jaccard between consecutive steps, reuse distance)
  - working set size over sliding step windows
  - simple cache simulations (LRU hit rate under block capacity)

The analysis can recompute block IDs for any block size based on selected token
indices (selected_token_pos).
"""

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from trace_utils import compute_block_ids, find_trace_dirs, iter_jsonl_records, jaccard, load_run_meta


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _mean_from_sum(count: int, total: float) -> float:
    return float(total) / float(count) if count else 0.0


def _quantile_from_hist(hist: Counter, q: float) -> float:
    if not hist:
        return 0.0
    assert 0.0 <= q <= 1.0
    items = sorted(hist.items(), key=lambda kv: kv[0])
    total = sum(int(c) for _, c in items)
    if total <= 0:
        return 0.0
    target = int(math.ceil(q * total))
    cur = 0
    for v, c in items:
        cur += int(c)
        if cur >= target:
            return float(v)
    return float(items[-1][0])


def _gini(counts: Sequence[int]) -> float:
    # Gini coefficient for non-negative counts.
    xs = [int(x) for x in counts if int(x) >= 0]
    if not xs:
        return 0.0
    xs.sort()
    n = len(xs)
    s = sum(xs)
    if s == 0:
        return 0.0
    # G = (2 * sum(i * x_i) / (n * sum x)) - (n + 1) / n
    acc = 0
    for i, x in enumerate(xs, start=1):
        acc += i * x
    return (2.0 * acc) / (n * s) - (n + 1.0) / n


def _entropy(counts: Sequence[int]) -> float:
    xs = [float(x) for x in counts if float(x) > 0]
    if not xs:
        return 0.0
    s = sum(xs)
    if s <= 0:
        return 0.0
    ent = 0.0
    for x in xs:
        p = x / s
        ent -= p * math.log(p + 1e-12)
    return float(ent)


def _norm_entropy(counts: Sequence[int]) -> float:
    k = len([x for x in counts if int(x) > 0])
    if k <= 1:
        return 0.0
    h = _entropy(counts)
    return float(h / math.log(k))


@dataclass
class OnlineStats:
    n: int = 0
    sum: float = 0.0
    sumsq: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def add(self, x: float) -> None:
        x = float(x)
        self.n += 1
        self.sum += x
        self.sumsq += x * x
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x

    def to_dict(self) -> Dict[str, Any]:
        mean = _mean_from_sum(self.n, self.sum)
        var = (self.sumsq / self.n - mean * mean) if self.n else 0.0
        return {
            "count": int(self.n),
            "min": float(self.min) if self.n else 0.0,
            "max": float(self.max) if self.n else 0.0,
            "mean": float(mean),
            "std": float(math.sqrt(max(var, 0.0))),
        }


def _iter_records_with_selected_tokens(trace_dir: Path) -> Iterator[Dict[str, Any]]:
    for rec in iter_jsonl_records(trace_dir):
        toks = rec.get("selected_token_pos")
        if not isinstance(toks, list) or not toks:
            continue
        yield rec


def _stream_key(rec: Dict[str, Any], key_mode: str) -> Tuple[int, int]:
    rid = _safe_int(rec.get("request_id"), 0)
    lid = _safe_int(rec.get("layer_id"), -1)
    if key_mode == "request":
        return rid, -1
    return rid, lid


def analyze(
    trace_dir: Path,
    *,
    block_size_tokens: int,
    key_mode: str,
    ws_windows: Sequence[int],
    recent_token_windows: Sequence[int],
    recent_block_windows: Sequence[int],
    reuse_cap_steps: int,
    lru_caps: Sequence[int],
    step_union_across_layers: bool,
    max_steps_per_stream: int = 0,
) -> Dict[str, Any]:
    # Dispersion/concentration stats per record.
    unique_blocks_stats = OnlineStats()
    blocks_span_stats = OnlineStats()
    density_stats = OnlineStats()
    norm_entropy_stats = OnlineStats()
    gini_stats = OnlineStats()
    tokens_per_block_mean_stats = OnlineStats()

    # Recent locality: share of selected tokens/blocks near the tail.
    recent_token_ratio_stats: Dict[int, OnlineStats] = {int(w): OnlineStats() for w in recent_token_windows if int(w) > 0}
    recent_block_ratio_stats: Dict[int, OnlineStats] = {int(w): OnlineStats() for w in recent_block_windows if int(w) > 0}
    mean_block_distance_stats = OnlineStats()

    # Locality over time: Jaccard between consecutive steps.
    step_jaccard_stats = OnlineStats()

    # Reuse distance histogram (in steps), capped.
    reuse_hist = Counter()
    first_touch = 0
    reuse_samples = 0

    # Working set size stats for multiple windows (in steps).
    ws_stats: Dict[int, OnlineStats] = {int(w): OnlineStats() for w in ws_windows if int(w) > 0}

    # Simple per-stream LRU simulations.
    lru_hit: Dict[int, int] = {int(c): 0 for c in lru_caps if int(c) > 0}
    lru_miss: Dict[int, int] = {int(c): 0 for c in lru_caps if int(c) > 0}

    # Per-stream state.
    last_step_blocks: Dict[Tuple[int, int], List[int]] = {}
    last_seen_step_by_block: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(dict)

    # Working set window state: for each stream and each window length, maintain a deque of step block sets.
    ws_deques: Dict[Tuple[int, int], Dict[int, Deque[List[int]]]] = defaultdict(lambda: {w: deque() for w in ws_stats})
    ws_counters: Dict[Tuple[int, int], Dict[int, Counter]] = defaultdict(lambda: {w: Counter() for w in ws_stats})

    # LRU state: for each stream and capacity, maintain LRU lists + membership sets.
    lru_deques: Dict[Tuple[int, int], Dict[int, Deque[int]]] = defaultdict(lambda: {c: deque() for c in lru_hit})
    lru_sets: Dict[Tuple[int, int], Dict[int, set]] = defaultdict(lambda: {c: set() for c in lru_hit})

    steps_seen_per_stream: Dict[Tuple[int, int], int] = defaultdict(int)

    # Optional step-level union: merge all layers' touched blocks at the same (request, step).
    pending_union_key: Optional[Tuple[int, int]] = None  # (request_id, step_idx)
    pending_union_blocks: Optional[set] = None

    def _emit_virtual_record_for_union(request_id: int, step_idx: int, seq_len: int, blocks: Iterable[int], sample_rec: Dict[str, Any]) -> Dict[str, Any]:
        # Construct a minimal record compatible with the rest of the pipeline.
        return {
            "request_id": int(request_id),
            "layer_id": -1,
            "step_idx": int(step_idx),
            "seq_len_current": int(seq_len),
            "selected_token_pos": sample_rec.get("selected_token_pos", []),
            "_union_blocks_override": sorted(set(int(b) for b in blocks)),
        }

    def _process_record(rec: Dict[str, Any]) -> None:
        nonlocal first_touch, reuse_samples
        step_idx = _safe_int(rec.get("step_idx"), 0)
        seq_len = _safe_int(rec.get("seq_len_current"), 0)
        key = _stream_key(rec, key_mode)

        # Optional truncation to keep analysis fast on very large datasets.
        if max_steps_per_stream > 0:
            if steps_seen_per_stream[key] >= int(max_steps_per_stream):
                return

        toks = rec.get("selected_token_pos")
        if not isinstance(toks, list) or not toks:
            return
        tok_pos = [int(x) for x in toks if isinstance(x, int) or isinstance(x, float)]
        if not tok_pos:
            return

        # Compute blocks for this record at the requested block size, or use union override.
        if "_union_blocks_override" in rec:
            block_ids_set = list(rec["_union_blocks_override"])
            block_ids_all = block_ids_set
        else:
            block_ids_all = compute_block_ids(tok_pos, int(block_size_tokens))
            if not block_ids_all:
                return
            block_ids_set = sorted(set(block_ids_all))

        unique_blocks = len(block_ids_set)
        unique_blocks_stats.add(unique_blocks)

        span = (max(block_ids_set) - min(block_ids_set) + 1) if block_ids_set else 0
        blocks_span_stats.add(span)
        density_stats.add((unique_blocks / span) if span > 0 else 0.0)

        per_block_counts = Counter(block_ids_all)
        counts = list(per_block_counts.values())
        norm_entropy_stats.add(_norm_entropy(counts))
        gini_stats.add(_gini(counts))
        tokens_per_block_mean_stats.add(float(len(block_ids_all)) / float(unique_blocks) if unique_blocks else 0.0)

        # Recent locality ratios (tokens and blocks near the tail).
        if seq_len > 0:
            tail_pos = seq_len - 1
            # token distance windows
            for w in recent_token_ratio_stats:
                near = 0
                for p in tok_pos:
                    if tail_pos - int(p) <= int(w):
                        near += 1
                recent_token_ratio_stats[w].add(float(near) / float(len(tok_pos)) if tok_pos else 0.0)

            # block distance windows
            cur_block = int(tail_pos // int(block_size_tokens)) if int(block_size_tokens) > 0 else 0
            dists = [max(0, cur_block - int(b)) for b in block_ids_set]
            if dists:
                mean_block_distance_stats.add(sum(dists) / float(len(dists)))
            for w in recent_block_ratio_stats:
                near_b = sum(1 for d in dists if d <= int(w))
                recent_block_ratio_stats[w].add(float(near_b) / float(len(dists)) if dists else 0.0)

        # Temporal locality: Jaccard between consecutive steps in the same stream.
        prev = last_step_blocks.get(key)
        if prev is not None:
            step_jaccard_stats.add(jaccard(prev, block_ids_set))
        last_step_blocks[key] = block_ids_set

        # Reuse distance: for each touched block, how many steps since last touch.
        last_seen = last_seen_step_by_block[key]
        for bid in block_ids_set:
            prev_step = last_seen.get(int(bid))
            if prev_step is None:
                first_touch += 1
            else:
                d = int(step_idx - int(prev_step))
                if d < 0:
                    d = 0
                if d > int(reuse_cap_steps):
                    d = int(reuse_cap_steps)
                reuse_hist[d] += 1
                reuse_samples += 1
            last_seen[int(bid)] = int(step_idx)

        # Working set sizes over sliding windows: unique blocks in last W steps.
        if ws_stats:
            dq_map = ws_deques[key]
            ctr_map = ws_counters[key]
            for w in ws_stats:
                dq = dq_map[w]
                ctr = ctr_map[w]
                dq.append(block_ids_set)
                for bid in block_ids_set:
                    ctr[int(bid)] += 1
                while len(dq) > int(w):
                    old = dq.popleft()
                    for bid in old:
                        ctr[int(bid)] -= 1
                        if ctr[int(bid)] <= 0:
                            del ctr[int(bid)]
                ws_stats[w].add(len(ctr))

        # LRU simulations (per stream).
        if lru_hit:
            dq_map = lru_deques[key]
            set_map = lru_sets[key]
            for cap in lru_hit:
                dq = dq_map[cap]
                st = set_map[cap]
                # We treat the step's touched blocks as a batch access.
                for bid in block_ids_set:
                    b = int(bid)
                    if b in st:
                        lru_hit[cap] += 1
                        # Move-to-end (MRU).
                        try:
                            dq.remove(b)
                        except ValueError:
                            pass
                        dq.append(b)
                    else:
                        lru_miss[cap] += 1
                        st.add(b)
                        dq.append(b)
                        while len(dq) > int(cap):
                            ev = int(dq.popleft())
                            st.discard(ev)

        steps_seen_per_stream[key] += 1

    for rec in _iter_records_with_selected_tokens(trace_dir):
        if not step_union_across_layers:
            _process_record(rec)
            continue

        rid = _safe_int(rec.get("request_id"), 0)
        step_idx = _safe_int(rec.get("step_idx"), 0)
        seq_len = _safe_int(rec.get("seq_len_current"), 0)
        toks = rec.get("selected_token_pos")
        if not isinstance(toks, list) or not toks:
            continue
        tok_pos = [int(x) for x in toks if isinstance(x, int) or isinstance(x, float)]
        if not tok_pos:
            continue
        blocks = compute_block_ids(tok_pos, int(block_size_tokens))
        if not blocks:
            continue

        k = (rid, step_idx)
        if pending_union_key is None:
            pending_union_key = k
            pending_union_blocks = set()
        if k != pending_union_key:
            assert pending_union_blocks is not None
            vr = _emit_virtual_record_for_union(pending_union_key[0], pending_union_key[1], seq_len, pending_union_blocks, rec)
            _process_record(vr)
            pending_union_key = k
            pending_union_blocks = set()
        assert pending_union_blocks is not None
        pending_union_blocks.update(blocks)

    if step_union_across_layers and pending_union_key is not None and pending_union_blocks is not None:
        vr = {
            "request_id": int(pending_union_key[0]),
            "layer_id": -1,
            "step_idx": int(pending_union_key[1]),
            "seq_len_current": 0,
            "selected_token_pos": [],
            "_union_blocks_override": sorted(set(int(b) for b in pending_union_blocks)),
        }
        _process_record(vr)

    reuse_total = sum(int(v) for v in reuse_hist.values())
    reuse_cdf = []
    cur = 0
    for d in sorted(reuse_hist.keys()):
        cur += int(reuse_hist[d])
        reuse_cdf.append({"reuse_distance_steps": int(d), "cdf": float(cur) / float(reuse_total) if reuse_total else 0.0})

    return {
        "trace_dir": str(trace_dir),
        "block_size_tokens": int(block_size_tokens),
        "key_mode": str(key_mode),
        "record_stats": {
            "unique_blocks": unique_blocks_stats.to_dict(),
            "block_span": blocks_span_stats.to_dict(),
            "block_density": density_stats.to_dict(),
            "tokens_per_block_mean": tokens_per_block_mean_stats.to_dict(),
            "block_concentration": {
                "normalized_entropy": norm_entropy_stats.to_dict(),
                "gini": gini_stats.to_dict(),
            },
            "recent_locality": {
                "recent_token_ratio": {str(w): recent_token_ratio_stats[w].to_dict() for w in sorted(recent_token_ratio_stats.keys())},
                "recent_block_ratio": {str(w): recent_block_ratio_stats[w].to_dict() for w in sorted(recent_block_ratio_stats.keys())},
                "mean_block_distance": mean_block_distance_stats.to_dict(),
            },
        },
        "temporal_locality": {
            "step_jaccard": step_jaccard_stats.to_dict(),
            "reuse_distance_hist_steps": {str(k): int(v) for k, v in reuse_hist.items()},
            "reuse_distance_cdf_steps": reuse_cdf[:2000],  # cap output size
            "first_touch_blocks": int(first_touch),
            "reuse_samples": int(reuse_samples),
            "reuse_cap_steps": int(reuse_cap_steps),
        },
        "working_set": {str(w): ws_stats[w].to_dict() for w in sorted(ws_stats.keys())},
        "cache_sim": {
            str(cap): {
                "capacity_blocks": int(cap),
                "hits": int(lru_hit[cap]),
                "misses": int(lru_miss[cap]),
                "hit_rate": float(lru_hit[cap]) / float(lru_hit[cap] + lru_miss[cap]) if (lru_hit[cap] + lru_miss[cap]) else 0.0,
            }
            for cap in sorted(lru_hit.keys())
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze DSA trace JSONL for KV placement research.")
    p.add_argument("--input", type=str, required=True, help="Trace root (outputs/<run>/ or block<bs>/).")
    p.add_argument("--out", type=str, default="", help="Output JSON path (default: <trace_dir>/analysis.json).")
    p.add_argument("--block-size", type=int, default=0, help="Override block size in tokens (0=use run_meta.json).")
    p.add_argument("--block-sizes", type=str, default="", help="Comma-separated block sizes to analyze (e.g. 16,32,64).")
    p.add_argument("--key-mode", type=str, default="request_layer", choices=["request_layer", "request"], help="Stream key granularity.")
    p.add_argument("--ws-windows", type=str, default="1,4,16,64", help="Working set windows in steps (comma-separated).")
    p.add_argument("--recent-token-windows", type=str, default="256,1024,4096", help="Recent token windows for locality ratios (comma-separated).")
    p.add_argument("--recent-block-windows", type=str, default="4,16,64", help="Recent block windows for locality ratios (comma-separated).")
    p.add_argument("--reuse-cap-steps", type=int, default=2048, help="Cap reuse distance histogram at this step count.")
    p.add_argument("--lru-caps", type=str, default="128,256,512,1024", help="LRU capacities in blocks (comma-separated).")
    p.add_argument("--step-union", action="store_true", help="Union blocks across layers per (request_id, step_idx).")
    p.add_argument("--max-steps-per-stream", type=int, default=0, help="Optional truncation for speed (0=no limit).")
    args = p.parse_args()

    inp = Path(args.input)
    trace_dirs = find_trace_dirs(inp)
    if not trace_dirs:
        raise SystemExit(f"No trace_steps_*.jsonl found under: {inp}")

    key_mode = "request_layer" if args.key_mode == "request_layer" else "request"
    ws_windows = [int(x) for x in args.ws_windows.split(",") if x.strip()]
    recent_token_windows = [int(x) for x in args.recent_token_windows.split(",") if x.strip()]
    recent_block_windows = [int(x) for x in args.recent_block_windows.split(",") if x.strip()]
    lru_caps = [int(x) for x in args.lru_caps.split(",") if x.strip()]

    # Determine block sizes to run.
    bs_list: List[int] = []
    if args.block_sizes:
        bs_list = [int(x) for x in args.block_sizes.split(",") if x.strip()]
    elif int(args.block_size) > 0:
        bs_list = [int(args.block_size)]

    all_results: List[Dict[str, Any]] = []
    for td in trace_dirs:
        meta = load_run_meta(td)
        if not bs_list:
            if meta.block_size_tokens is None:
                raise SystemExit(f"Missing run_meta.json block_size_tokens under {td}; pass --block-size.")
            cur_bs_list = [int(meta.block_size_tokens)]
        else:
            cur_bs_list = bs_list

        for bs in cur_bs_list:
            res = analyze(
                td,
                block_size_tokens=int(bs),
                key_mode=key_mode,
                ws_windows=ws_windows,
                recent_token_windows=recent_token_windows,
                recent_block_windows=recent_block_windows,
                reuse_cap_steps=int(args.reuse_cap_steps),
                lru_caps=lru_caps,
                step_union_across_layers=bool(args.step_union),
                max_steps_per_stream=int(args.max_steps_per_stream),
            )
            res["run_meta"] = {
                "run_name": meta.run_name,
                "dataset": meta.dataset,
                "rank": meta.rank,
                "world_size": meta.world_size,
                "schema_version": meta.schema_version,
            }
            all_results.append(res)

    # Write output(s). If multiple trace_dirs or block sizes are analyzed, write a bundle JSON.
    if args.out:
        out_path = Path(args.out)
    else:
        # Default to the first trace dir.
        out_path = trace_dirs[0] / "analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": all_results}, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

