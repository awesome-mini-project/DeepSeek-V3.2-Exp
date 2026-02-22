import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class RunMeta:
    trace_dir: Path
    block_size_tokens: Optional[int] = None
    run_name: str = ""
    dataset: str = ""
    world_size: Optional[int] = None
    rank: Optional[int] = None
    schema_version: Optional[int] = None


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_trace_dirs(root: Path) -> List[Path]:
    """
    Find leaf trace directories that contain trace_steps_*.jsonl.

    The expected layout is:
      outputs/<run>/block<bs>/{trace_steps_*.jsonl, run_meta.json, summary.json}
    """
    root = Path(root)
    out: List[Path] = []
    if root.is_file():
        return out

    # If root itself looks like a block dir, accept it.
    if list(root.glob("trace_steps_*.jsonl")):
        out.append(root)
        return out

    for p in root.rglob("trace_steps_*.jsonl"):
        out.append(p.parent)
    # de-dup, keep stable order
    uniq: List[Path] = []
    seen = set()
    for d in sorted(set(out)):
        if d in seen:
            continue
        seen.add(d)
        uniq.append(d)
    return uniq


def load_run_meta(trace_dir: Path) -> RunMeta:
    trace_dir = Path(trace_dir)
    meta_path = trace_dir / "run_meta.json"
    if not meta_path.exists():
        return RunMeta(trace_dir=trace_dir)
    j = _load_json(meta_path)
    return RunMeta(
        trace_dir=trace_dir,
        block_size_tokens=j.get("block_size_tokens"),
        run_name=str(j.get("run_name") or ""),
        dataset=str(j.get("dataset") or ""),
        world_size=j.get("world_size"),
        rank=j.get("rank"),
        schema_version=j.get("schema_version"),
    )


def iter_jsonl_records(trace_dir: Path) -> Iterator[Dict[str, Any]]:
    """
    Iterate all JSONL records under trace_dir, in filename order.

    This is streaming: it does not load entire files into memory.
    """
    trace_dir = Path(trace_dir)
    for fp in sorted(trace_dir.glob("trace_steps_*.jsonl")):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def get_env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return int(default)
    try:
        return int(v)
    except ValueError:
        return int(default)


def get_env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return float(default)
    try:
        return float(v)
    except ValueError:
        return float(default)


def compute_block_ids(token_positions: List[int], block_size_tokens: int) -> List[int]:
    bs = int(block_size_tokens)
    if bs <= 0:
        return []
    return [int(p) // bs for p in token_positions]


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(union) if union else 0.0

