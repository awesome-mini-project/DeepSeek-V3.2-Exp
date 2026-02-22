import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--kv-block-size", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out or os.path.join("outputs", f"sanity_no_torch_{int(time.time() * 1000)}"))
    _ensure_dir(out_dir)

    trace_path = out_dir / "trace_steps.jsonl"
    kv_block_size = int(args.kv_block_size)

    unique_blocks_vals: List[int] = []
    offsets_vals: List[int] = []
    tpb_vals: List[int] = []

    with open(trace_path, "w", encoding="utf-8") as f:
        for req_id in range(int(args.requests)):
            prefix_cached_blocks = 8 if (req_id % 2 == 0) else 0
            for step in range(int(args.steps)):
                end_pos = 128 + step
                for layer_id in range(int(args.layers)):
                    selected_token_pos = list(range(max(0, end_pos - 64), end_pos, 2))[:2048]
                    unique_pos = sorted(set(selected_token_pos))
                    offsets = [(end_pos - 1) - p for p in unique_pos]
                    offsets_vals.extend(offsets)

                    block_ids = [p // kv_block_size for p in unique_pos]
                    unique_block_ids = sorted(set(block_ids))
                    unique_blocks_vals.append(len(unique_block_ids))

                    per_block: Dict[int, int] = {}
                    for bid in block_ids:
                        per_block[bid] = per_block.get(bid, 0) + 1
                    tpb = list(per_block.values())
                    tpb_vals.extend(tpb)

                    inter_blocks = [b for b in unique_block_ids if b < prefix_cached_blocks]
                    inter_ratio = (len(inter_blocks) / len(unique_block_ids)) if unique_block_ids else 0.0

                    rec: Dict[str, Any] = {
                        "event": "dsa_topk",
                        "run_name": out_dir.name,
                        "dataset": "sanity_no_torch",
                        "rank": 0,
                        "world_size": 1,
                        "request_id": req_id,
                        "layer_id": layer_id,
                        "step_idx": step,
                        "seq_len_current": end_pos,
                        "unique_token_pos_count": len(unique_pos),
                        "offset_min": min(offsets) if offsets else 0,
                        "offset_p50": sorted(offsets)[len(offsets) // 2] if offsets else 0,
                        "offset_max": max(offsets) if offsets else 0,
                        "block_size_tokens": kv_block_size,
                        "selected_block_ids": unique_block_ids,
                        "unique_blocks": len(unique_block_ids),
                        "tokens_per_touched_block": {
                            "mean": (sum(tpb) / len(tpb)) if tpb else 0.0,
                            "p50": sorted(tpb)[len(tpb) // 2] if tpb else 0,
                            "p95": sorted(tpb)[max(0, int(round((len(tpb) - 1) * 0.95)))] if tpb else 0,
                        },
                        "kv_fetch": {
                            "hbm": {
                                "hit_blocks": unique_block_ids,
                                "bytes_read": 0,
                                "read_ops": 1 if unique_block_ids else 0,
                                "latency_us": 100,
                                "batch_size": len(unique_block_ids),
                            },
                            "local_pool": {"hit_blocks": [], "bytes_read": 0, "read_ops": 0, "latency_us": None, "batch_size": 0},
                            "remote_pool": {"hit_blocks": [], "bytes_read": 0, "read_ops": 0, "latency_us": None, "batch_size": 0},
                        },
                        "prefix": {
                            "prefix_cache_hit": prefix_cached_blocks > 0,
                            "prefix_cached_blocks": prefix_cached_blocks,
                            "prefix_key": "sanity",
                            "intersection_ratio": inter_ratio,
                            "intersection_blocks": inter_blocks,
                        },
                        "selected_token_pos": selected_token_pos,
                        "scores_stats": {"min": 0.0, "mean": 0.0, "max": 0.0},
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _dist(vals: List[int]) -> Dict[str, Any]:
        if not vals:
            return {"count": 0}
        s = sorted(vals)
        return {
            "count": len(s),
            "min": int(s[0]),
            "p50": int(s[len(s) // 2]),
            "p95": int(s[max(0, int(round((len(s) - 1) * 0.95)))]),
            "max": int(s[-1]),
            "mean": float(sum(s)) / float(len(s)),
        }

    summary = {
        "run_name": out_dir.name,
        "dataset": "sanity_no_torch",
        "config": {"kv_block_size_tokens": kv_block_size},
        "unique_blocks": _dist(unique_blocks_vals),
        "tokens_per_touched_block": _dist(tpb_vals),
        "offsets": _dist(offsets_vals),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(str(out_dir))


if __name__ == "__main__":
    main()

