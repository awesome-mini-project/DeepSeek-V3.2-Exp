import json
import os
import time
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


def _now_ms() -> int:
    return int(time.time() * 1000)


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _quantile(sorted_vals: Sequence[int], q: float) -> int:
    if not sorted_vals:
        return 0
    if q <= 0:
        return int(sorted_vals[0])
    if q >= 1:
        return int(sorted_vals[-1])
    idx = int(round((len(sorted_vals) - 1) * q))
    return int(sorted_vals[idx])


def _mean(vals: Sequence[int]) -> float:
    if not vals:
        return 0.0
    return float(sum(vals)) / float(len(vals))


def _p50(vals: Sequence[int]) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    return _quantile(s, 0.5)


def _p95(vals: Sequence[int]) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    return _quantile(s, 0.95)


def _sha1_ints(ints: Sequence[int]) -> str:
    h = hashlib.sha1()
    for x in ints:
        h.update(int(x).to_bytes(4, "little", signed=True))
    return h.hexdigest()


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
    # Output schema controls (make large/derivable fields opt-in).
    record_meta_per_record: bool = False
    record_block_ids: bool = False
    record_kv_fetch: bool = True
    record_kv_fetch_latency_us: bool = False
    record_kv_fetch_read_ops: bool = False
    record_empty_tiers: bool = False
    enable_prefix_analysis: bool = False
    prefix_cache_key_tokens: int = 256
    max_requests_per_file: int = 4  # shard JSONL every N requests; 0 = no split

    @staticmethod
    def from_env() -> "TraceConfig":
        def _get_bool(k: str, default: bool) -> bool:
            v = os.getenv(k)
            if v is None:
                return default
            return v.strip().lower() in ("1", "true", "yes", "y", "on")

        def _get_int(k: str, default: int) -> int:
            v = os.getenv(k)
            if v is None:
                return default
            try:
                return int(v)
            except ValueError:
                return default

        def _get_float(k: str, default: float) -> float:
            v = os.getenv(k)
            if v is None:
                return default
            try:
                return float(v)
            except ValueError:
                return default

        enabled = _get_bool("DS_TRACE_ENABLE", False)
        out_dir = os.getenv("DS_TRACE_OUT", "")
        if enabled and not out_dir:
            out_dir = str(Path("outputs") / f"trace_{_now_ms()}")
        return TraceConfig(
            enabled=enabled,
            out_dir=out_dir,
            kv_block_size_tokens=_get_int("DS_TRACE_KV_BLOCK_SIZE", 64),
            store_scores_topk=_get_bool("DS_TRACE_STORE_SCORES", False),
            store_selected_token_pos=_get_bool("DS_TRACE_STORE_TOKEN_POS", True),
            sample_rate=_get_float("DS_TRACE_SAMPLE_RATE", 1.0),
            rank0_only=_get_bool("DS_TRACE_RANK0_ONLY", True),
            sync_cuda_for_timing=_get_bool("DS_TRACE_SYNC_CUDA", True),
            record_meta_per_record=_get_bool("DS_TRACE_RECORD_META_PER_RECORD", False),
            record_block_ids=_get_bool("DS_TRACE_RECORD_BLOCK_IDS", False),
            record_kv_fetch=_get_bool("DS_TRACE_RECORD_KV_FETCH", True),
            record_kv_fetch_latency_us=_get_bool("DS_TRACE_RECORD_KV_FETCH_LATENCY_US", False),
            record_kv_fetch_read_ops=_get_bool("DS_TRACE_RECORD_KV_FETCH_READ_OPS", False),
            record_empty_tiers=_get_bool("DS_TRACE_RECORD_EMPTY_TIERS", False),
            prefix_cache_key_tokens=_get_int("DS_TRACE_PREFIX_KEY_TOKENS", 256),
            max_requests_per_file=_get_int("DS_TRACE_MAX_REQUESTS_PER_FILE", 4),
        )


def apply_env_overrides(cfg: TraceConfig) -> TraceConfig:
    """
    Apply environment-variable overrides to an explicit TraceConfig.

    This lets bash scripts control trace schema without adding more CLI flags.
    """

    def _get_bool(k: str) -> Optional[bool]:
        v = os.getenv(k)
        if v is None:
            return None
        return v.strip().lower() in ("1", "true", "yes", "y", "on")

    def _get_int(k: str) -> Optional[int]:
        v = os.getenv(k)
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            return None

    updates: Dict[str, Any] = {}
    b = _get_bool("DS_TRACE_RECORD_META_PER_RECORD")
    if b is not None:
        updates["record_meta_per_record"] = b
    b = _get_bool("DS_TRACE_RECORD_BLOCK_IDS")
    if b is not None:
        updates["record_block_ids"] = b
    b = _get_bool("DS_TRACE_RECORD_KV_FETCH")
    if b is not None:
        updates["record_kv_fetch"] = b
    b = _get_bool("DS_TRACE_RECORD_KV_FETCH_LATENCY_US")
    if b is not None:
        updates["record_kv_fetch_latency_us"] = b
    b = _get_bool("DS_TRACE_RECORD_KV_FETCH_READ_OPS")
    if b is not None:
        updates["record_kv_fetch_read_ops"] = b
    b = _get_bool("DS_TRACE_RECORD_EMPTY_TIERS")
    if b is not None:
        updates["record_empty_tiers"] = b
    i = _get_int("DS_TRACE_MAX_REQUESTS_PER_FILE")
    if i is not None:
        updates["max_requests_per_file"] = int(i)
    i = _get_int("DS_TRACE_KV_BLOCK_SIZE")
    if i is not None:
        updates["kv_block_size_tokens"] = int(i)
    return replace(cfg, **updates) if updates else cfg


@dataclass
class RequestPrefixInfo:
    prefix_cache_hit: bool
    prefix_cached_blocks: int
    prefix_key: str


class PrefixCacheAnalyzer:
    """
    An analyzer that approximates prefix-cache reuse relationship. It does not reuse KV,
    it only emits (hit/miss, cached blocks) given tokenized prompts.
    """

    def __init__(self, prefix_cache_key_tokens: int):
        self.prefix_cache_key_tokens = int(prefix_cache_key_tokens)
        # hash -> cached_prefix_len_tokens (approx)
        self._seen: Dict[str, int] = {}

    def analyze_prompt_tokens(self, prompt_tokens: Sequence[int], kv_block_size_tokens: int) -> RequestPrefixInfo:
        """
        We approximate prefix cache reuse by matching hashes of prompt prefixes.
        To make it stable for growing prompts (multi-turn chat), we check multiple
        prefix lengths and pick the longest match.
        """
        n = len(prompt_tokens)
        if n <= 0:
            return RequestPrefixInfo(prefix_cache_hit=False, prefix_cached_blocks=0, prefix_key="")

        # Candidate lengths (tokens). Keep small set to control overhead.
        max_k = min(n, self.prefix_cache_key_tokens)
        candidates = [32, 64, 128, 256, 512, 1024]
        lens = [k for k in candidates if k <= max_k]
        if max_k not in lens:
            lens.append(max_k)
        lens = sorted(set(lens))

        best_key = ""
        best_len = 0
        for k in reversed(lens):
            key = _sha1_ints(prompt_tokens[:k])
            if key in self._seen:
                best_key = key
                best_len = int(self._seen[key])
                break

        hit = best_len > 0
        if not hit:
            # Insert all candidate prefix hashes for future matches.
            for k in lens:
                key = _sha1_ints(prompt_tokens[:k])
                self._seen.setdefault(key, int(k))
            best_key = _sha1_ints(prompt_tokens[:lens[-1]])
            best_len = int(lens[-1])

        cached_blocks = (best_len + kv_block_size_tokens - 1) // kv_block_size_tokens
        return RequestPrefixInfo(prefix_cache_hit=hit, prefix_cached_blocks=int(cached_blocks), prefix_key=best_key)


class TraceWriter:
    """
    Writes JSONL trace records asynchronously, with optional sharding by request count.

    IO is offloaded to a background thread so that rank 0's GPU compute path
    is not blocked by disk writes (which caused NCCL timeouts on large runs).

    When max_requests_per_file > 0, a new file is created every N requests.
    File naming: trace_steps_{start}_{end}.jsonl (start/end are request IDs)
    """

    def __init__(self, out_dir: str, filename: str = "trace_steps.jsonl", max_requests_per_file: int = 4):
        self.out_dir = Path(out_dir)
        _ensure_dir(self.out_dir)
        self.filename = filename
        self.max_requests_per_file = int(max_requests_per_file)
        self._fh = None
        self._total_records = 0
        self._cur_req_start = 0
        self._cur_req_end = 0
        self._seen_requests: set = set()
        self._requests_in_file = 0

        import threading
        import queue as _queue_mod
        self._queue: _queue_mod.Queue = _queue_mod.Queue(maxsize=50000)
        self._writer_thread = threading.Thread(target=self._bg_writer, daemon=True)
        self._writer_thread.start()

    def _shard_path(self, start: int, end: int) -> Path:
        stem = Path(self.filename).stem
        suffix = Path(self.filename).suffix
        return self.out_dir / f"{stem}_{start}_{end}{suffix}"

    def _open_if_needed(self) -> None:
        if self._fh is not None:
            return
        if self.max_requests_per_file > 0:
            path = self._shard_path(self._cur_req_start, self._cur_req_start + self.max_requests_per_file)
        else:
            path = self.out_dir / self.filename
        self._fh = open(path, "a", encoding="utf-8")

    def _bg_writer(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            line, req_id = item
            self._open_if_needed()
            self._fh.write(line)

            if req_id is not None and req_id not in self._seen_requests:
                self._seen_requests.add(req_id)
                self._requests_in_file += 1
                self._cur_req_end = int(req_id) + 1

            if self.max_requests_per_file > 0 and self._requests_in_file >= self.max_requests_per_file:
                self._rotate_file()

    def write(self, rec: Dict[str, Any]) -> None:
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        req_id = rec.get("request_id")
        self._total_records += 1
        self._queue.put((line, req_id))

    def _rotate_file(self) -> None:
        if self._fh is not None:
            old_path = self._shard_path(self._cur_req_start, self._cur_req_start + self.max_requests_per_file)
            new_path = self._shard_path(self._cur_req_start, self._cur_req_end)
            self._fh.close()
            self._fh = None
            if old_path.exists() and old_path != new_path:
                old_path.rename(new_path)
            self._cur_req_start = self._cur_req_end
            self._requests_in_file = 0

    @property
    def total_records(self) -> int:
        return self._total_records

    def close(self) -> None:
        self._queue.put(None)
        self._writer_thread.join(timeout=60)
        if self._fh is not None:
            if self.max_requests_per_file > 0 and self._requests_in_file > 0:
                old_path = self._shard_path(self._cur_req_start, self._cur_req_start + self.max_requests_per_file)
                new_path = self._shard_path(self._cur_req_start, self._cur_req_end)
                self._fh.close()
                self._fh = None
                if old_path.exists() and old_path != new_path:
                    old_path.rename(new_path)
            else:
                self._fh.close()
                self._fh = None


@dataclass
class TraceContext:
    run_name: str = ""
    dataset: str = ""
    request_ids: Optional[List[int]] = None
    prefix_infos: Optional[List[RequestPrefixInfo]] = None
    step_idx: Optional[int] = None
    step_wall_us: Optional[int] = None


class _DistributionSummary:
    def __init__(self) -> None:
        self.values: List[int] = []

    def add(self, v: int) -> None:
        self.values.append(int(v))

    def summary(self) -> Dict[str, Any]:
        if not self.values:
            return {"count": 0}
        s = sorted(self.values)
        return {
            "count": len(s),
            "min": int(s[0]),
            "p50": int(_quantile(s, 0.5)),
            "p95": int(_quantile(s, 0.95)),
            "max": int(s[-1]),
            "mean": float(sum(s)) / float(len(s)),
        }


class Tracer:
    def __init__(self, cfg: TraceConfig):
        self.cfg = cfg
        trace_dir = str(Path(cfg.out_dir) / f"block{cfg.kv_block_size_tokens}")
        self.trace_dir = trace_dir
        self.writer = TraceWriter(trace_dir, max_requests_per_file=cfg.max_requests_per_file)
        self.ctx = TraceContext()
        self._run_meta_written = False
        self._unique_blocks = _DistributionSummary()
        self._tokens_per_block = _DistributionSummary()
        self._offsets = _DistributionSummary()
        self._touched_ratios: List[float] = []
        self._intersection_ratio = []  # float values
        self._prefix_hot_blocks: Dict[int, Dict[int, int]] = {}  # request_id -> {block_id: count}
        self._pending_by_step: Dict[int, List[Dict[str, Any]]] = {}

        if self.enabled:
            self._write_run_meta()

    @property
    def enabled(self) -> bool:
        if not self.cfg.enabled:
            return False
        if self.cfg.rank0_only and _rank() != 0:
            return False
        return True

    def set_run_meta(self, run_name: str = "", dataset: str = "") -> None:
        self.ctx.run_name = run_name
        self.ctx.dataset = dataset
        if self.enabled:
            self._write_run_meta()

    def _write_run_meta(self) -> None:
        try:
            p = Path(self.trace_dir) / "run_meta.json"
            _ensure_dir(Path(self.trace_dir))
            meta = {
                "schema_version": 2,
                "created_at_ms": _now_ms(),
                "run_name": self.ctx.run_name,
                "dataset": self.ctx.dataset,
                "rank": _rank(),
                "world_size": _world_size(),
                "block_size_tokens": int(self.cfg.kv_block_size_tokens),
                "max_requests_per_file": int(self.cfg.max_requests_per_file),
                "config": {
                    "store_scores_topk": bool(self.cfg.store_scores_topk),
                    "store_selected_token_pos": bool(self.cfg.store_selected_token_pos),
                    "sample_rate": float(self.cfg.sample_rate),
                    "rank0_only": bool(self.cfg.rank0_only),
                    "sync_cuda_for_timing": bool(self.cfg.sync_cuda_for_timing),
                    "record_meta_per_record": bool(self.cfg.record_meta_per_record),
                    "record_block_ids": bool(self.cfg.record_block_ids),
                    "record_kv_fetch": bool(self.cfg.record_kv_fetch),
                    "record_kv_fetch_latency_us": bool(self.cfg.record_kv_fetch_latency_us),
                    "record_kv_fetch_read_ops": bool(self.cfg.record_kv_fetch_read_ops),
                    "record_empty_tiers": bool(self.cfg.record_empty_tiers),
                    "enable_prefix_analysis": bool(self.cfg.enable_prefix_analysis),
                    "prefix_cache_key_tokens": int(self.cfg.prefix_cache_key_tokens),
                },
            }
            with open(p, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            self._run_meta_written = True
        except Exception:
            # Best-effort; tracing must not crash on meta write errors.
            pass

    def set_batch(self, request_ids: Sequence[int], prefix_infos: Optional[Sequence[RequestPrefixInfo]] = None) -> None:
        self.ctx.request_ids = [int(x) for x in request_ids]
        if prefix_infos is None:
            self.ctx.prefix_infos = None
        else:
            self.ctx.prefix_infos = list(prefix_infos)

    def set_step_timing(self, step_idx: int, step_wall_us: Optional[int]) -> None:
        self.ctx.step_idx = int(step_idx)
        self.ctx.step_wall_us = None if step_wall_us is None else int(step_wall_us)
        if not self.enabled:
            return
        if step_wall_us is None:
            return
        if not self.cfg.record_kv_fetch_latency_us:
            return
        pending = self._pending_by_step.pop(int(step_idx), [])
        for rec in pending:
            try:
                rec["kv_fetch"]["hbm"]["latency_us"] = int(step_wall_us)
            except Exception:
                pass
            self.writer.write(rec)

    def _should_sample(self) -> bool:
        if self.cfg.sample_rate >= 1.0:
            return True
        return torch.rand(()) < self.cfg.sample_rate

    def record_dsa_topk(
        self,
        *,
        layer_id: int,
        start_pos: int,
        end_pos: int,
        topk_indices: torch.Tensor,
        topk_scores: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.enabled:
            return
        if not self._should_sample():
            return

        bsz = int(topk_indices.size(0))
        request_ids = self.ctx.request_ids or list(range(bsz))
        if len(request_ids) != bsz:
            request_ids = list(range(bsz))

        prefix_infos = self.ctx.prefix_infos
        if prefix_infos is not None and len(prefix_infos) != bsz:
            prefix_infos = None

        # Shape is expected to be (bsz, seqlen, k). In decode seqlen==1.
        indices = topk_indices
        if indices.dim() == 2:
            indices = indices.unsqueeze(1)

        scores = topk_scores
        if scores is not None and scores.dim() == 2:
            scores = scores.unsqueeze(1)

        for bi in range(bsz):
            req_id = int(request_ids[bi])
            selected = indices[bi, -1].to(torch.int64).tolist()
            selected_unique = sorted(set(int(x) for x in selected))

            offsets = [int((end_pos - 1) - p) for p in selected_unique]
            offsets_sorted = sorted(offsets)
            offset_min = int(offsets_sorted[0]) if offsets_sorted else 0
            offset_p50 = int(_quantile(offsets_sorted, 0.5)) if offsets_sorted else 0
            offset_max = int(offsets_sorted[-1]) if offsets_sorted else 0

            block_size = int(self.cfg.kv_block_size_tokens)
            block_ids = [int(p // block_size) for p in selected_unique]
            unique_block_ids = sorted(set(block_ids))
            total_blocks_in_use = (end_pos + block_size - 1) // block_size
            touched_ratio = (len(unique_block_ids) / total_blocks_in_use) if total_blocks_in_use > 0 else 0.0

            per_block_counts: Dict[int, int] = {}
            for bid in block_ids:
                per_block_counts[bid] = per_block_counts.get(bid, 0) + 1
            tpb = list(per_block_counts.values())
            tpb_mean = _mean(tpb)
            tpb_p50 = int(_p50(tpb))
            tpb_p95 = int(_p95(tpb))

            prefix_hit = False
            prefix_cached_blocks = 0
            prefix_key = ""
            intersection_ratio = 0.0
            intersection_blocks: List[int] = []
            if self.cfg.enable_prefix_analysis and prefix_infos is not None:
                pi = prefix_infos[bi]
                prefix_hit = bool(pi.prefix_cache_hit)
                prefix_cached_blocks = int(pi.prefix_cached_blocks)
                prefix_key = str(pi.prefix_key)
                intersection_blocks = [bid for bid in unique_block_ids if bid < prefix_cached_blocks]
                intersection_ratio = (len(intersection_blocks) / len(unique_block_ids)) if unique_block_ids else 0.0
                if intersection_blocks:
                    hot = self._prefix_hot_blocks.setdefault(req_id, {})
                    for bid in intersection_blocks:
                        hot[bid] = hot.get(bid, 0) + 1
                self._intersection_ratio.append(float(intersection_ratio))

            rec: Dict[str, Any] = {
                "request_id": req_id,
                "layer_id": int(layer_id),
                "step_idx": int(start_pos),
                "seq_len_current": int(end_pos),
                "unique_token_pos_count": int(len(selected_unique)),
                "offset_min": offset_min,
                "offset_p50": offset_p50,
                "offset_max": offset_max,
                "unique_blocks": int(len(unique_block_ids)),
                "total_blocks_in_use": int(total_blocks_in_use),
                "touched_block_ratio": round(float(touched_ratio), 4),
                "tokens_per_touched_block": {"mean": round(float(tpb_mean), 2), "p50": tpb_p50, "p95": tpb_p95},
            }
            if self.cfg.record_meta_per_record:
                rec.update(
                    {
                        "event": "dsa_topk",
                        "run_name": self.ctx.run_name,
                        "dataset": self.ctx.dataset,
                        "rank": _rank(),
                        "world_size": _world_size(),
                        "block_size_tokens": block_size,
                    }
                )

            if self.cfg.record_block_ids:
                rec["selected_block_ids"] = unique_block_ids

            # C-point: HBM-only logical fetch stats (placeholder for future tiered fetch).
            if self.cfg.record_kv_fetch:
                bytes_per_token = 0
                env_bpt = os.getenv("DS_TRACE_KV_BYTES_PER_TOKEN")
                if env_bpt:
                    try:
                        bytes_per_token = int(env_bpt)
                    except ValueError:
                        bytes_per_token = 0
                hbm_bytes_read = int(bytes_per_token * (len(unique_block_ids) * block_size))
                hbm_read_ops = 1 if unique_block_ids else 0

                hbm: Dict[str, Any] = {
                    "bytes_read": hbm_bytes_read,
                    "batch_size": int(len(unique_block_ids)),
                }
                if self.cfg.record_block_ids:
                    hbm["hit_blocks"] = unique_block_ids
                if self.cfg.record_kv_fetch_read_ops:
                    hbm["read_ops"] = int(hbm_read_ops)
                if self.cfg.record_kv_fetch_latency_us:
                    hbm["latency_us"] = self.ctx.step_wall_us

                kv_fetch: Dict[str, Any] = {"hbm": hbm}

                # Only include empty tiers if explicitly requested.
                if self.cfg.record_empty_tiers:
                    kv_fetch["local_pool"] = {"bytes_read": 0, "batch_size": 0}
                    kv_fetch["remote_pool"] = {"bytes_read": 0, "batch_size": 0}
                    if self.cfg.record_block_ids:
                        kv_fetch["local_pool"]["hit_blocks"] = []
                        kv_fetch["remote_pool"]["hit_blocks"] = []
                    if self.cfg.record_kv_fetch_read_ops:
                        kv_fetch["local_pool"]["read_ops"] = 0
                        kv_fetch["remote_pool"]["read_ops"] = 0
                    if self.cfg.record_kv_fetch_latency_us:
                        kv_fetch["local_pool"]["latency_us"] = None
                        kv_fetch["remote_pool"]["latency_us"] = None

                rec["kv_fetch"] = kv_fetch
            if self.cfg.enable_prefix_analysis:
                rec["prefix"] = {
                    "prefix_cache_hit": prefix_hit,
                    "prefix_cached_blocks": prefix_cached_blocks,
                    "prefix_key": prefix_key,
                    "intersection_ratio": intersection_ratio,
                }
                if self.cfg.record_block_ids:
                    rec["prefix"]["intersection_blocks"] = intersection_blocks

            if self.cfg.store_selected_token_pos:
                rec["selected_token_pos"] = selected
            if self.cfg.store_scores_topk and scores is not None:
                rec["scores_topk"] = scores[bi, -1].detach().float().cpu().tolist()
            elif scores is not None:
                # Lightweight stats only.
                s = scores[bi, -1].detach().float()
                rec["scores_stats"] = {
                    "min": float(s.min().item()),
                    "mean": float(s.mean().item()),
                    "max": float(s.max().item()),
                }

            if self.cfg.record_kv_fetch and self.cfg.record_kv_fetch_latency_us and (self.ctx.step_wall_us is None) and (self.ctx.step_idx is not None):
                self._pending_by_step.setdefault(int(self.ctx.step_idx), []).append(rec)
            else:
                self.writer.write(rec)
            self._unique_blocks.add(len(unique_block_ids))
            self._touched_ratios.append(float(touched_ratio))
            for c in tpb:
                self._tokens_per_block.add(c)
            for off in offsets:
                self._offsets.add(off)

    def get_summary(self) -> Dict[str, Any]:
        inter = self._intersection_ratio
        inter_sorted = sorted(inter) if inter else []
        inter_sum = {
            "count": len(inter_sorted),
            "min": float(inter_sorted[0]) if inter_sorted else 0.0,
            "p50": float(_quantile([int(x * 1_000_000) for x in inter_sorted], 0.5) / 1_000_000.0) if inter_sorted else 0.0,
            "p95": float(_quantile([int(x * 1_000_000) for x in inter_sorted], 0.95) / 1_000_000.0) if inter_sorted else 0.0,
            "max": float(inter_sorted[-1]) if inter_sorted else 0.0,
            "mean": float(sum(inter_sorted) / len(inter_sorted)) if inter_sorted else 0.0,
        }
        # Hot blocks: top-20 per request_id.
        hot_blocks: Dict[str, Any] = {}
        for req_id, counter in self._prefix_hot_blocks.items():
            items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:20]
            hot_blocks[str(req_id)] = [{"block_id": int(b), "touch_count": int(c)} for b, c in items]

        tr = self._touched_ratios
        tr_sorted = sorted(tr) if tr else []
        tr_sum = {
            "count": len(tr_sorted),
            "min": round(float(tr_sorted[0]), 4) if tr_sorted else 0.0,
            "p50": round(float(_quantile([int(x * 1_000_000) for x in tr_sorted], 0.5) / 1_000_000.0), 4) if tr_sorted else 0.0,
            "p95": round(float(_quantile([int(x * 1_000_000) for x in tr_sorted], 0.95) / 1_000_000.0), 4) if tr_sorted else 0.0,
            "max": round(float(tr_sorted[-1]), 4) if tr_sorted else 0.0,
            "mean": round(float(sum(tr_sorted) / len(tr_sorted)), 4) if tr_sorted else 0.0,
        }

        return {
            "run_name": self.ctx.run_name,
            "dataset": self.ctx.dataset,
            "config": {
                "kv_block_size_tokens": self.cfg.kv_block_size_tokens,
                "store_scores_topk": self.cfg.store_scores_topk,
                "store_selected_token_pos": self.cfg.store_selected_token_pos,
                "sample_rate": self.cfg.sample_rate,
                "rank0_only": self.cfg.rank0_only,
                "sync_cuda_for_timing": self.cfg.sync_cuda_for_timing,
                "prefix_cache_key_tokens": self.cfg.prefix_cache_key_tokens,
            },
            "unique_blocks": self._unique_blocks.summary(),
            "touched_block_ratio": tr_sum,
            "tokens_per_touched_block": self._tokens_per_block.summary(),
            "offsets": self._offsets.summary(),
            "prefix_intersection_ratio": inter_sum,
            "prefix_hot_blocks": hot_blocks,
        }

    def close(self) -> None:
        self.writer.close()


_GLOBAL_TRACER: Optional[Tracer] = None


def init_tracer(cfg: Optional[TraceConfig] = None) -> Tracer:
    global _GLOBAL_TRACER
    if cfg is None:
        cfg = TraceConfig.from_env()
    else:
        cfg = apply_env_overrides(cfg)
    _GLOBAL_TRACER = Tracer(cfg)
    return _GLOBAL_TRACER


def get_tracer() -> Tracer:
    global _GLOBAL_TRACER
    if _GLOBAL_TRACER is None:
        _GLOBAL_TRACER = Tracer(TraceConfig.from_env())
    return _GLOBAL_TRACER

