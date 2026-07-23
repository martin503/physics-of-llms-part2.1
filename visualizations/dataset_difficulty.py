"""Measure the *realized* difficulty (op-count) distribution of a packed iGSM parquet
dataset, and how much difficulty is mixed inside each 768-token training window.

Why this exists
---------------
The parquet shards store only ``input_ids`` -- difficulty is not recorded. But it is
recoverable exactly: an iGSM problem's ``n_op`` equals the number of ``;`` tokens (id 26)
in its solution region ``[223] ... [224]`` (each operation is emitted as one ``;``-joined
clause). This was calibrated against ``problem.n_op`` on 1200 regenerated problems: 100%
exact match. Recovering op-counts this way needs no (expensive) regeneration -- only a
cheap token scan of the stored data.

This complements ``doc/data_generation.md`` (which explains the layout in theory) by
reporting what a given dataset directory *actually* contains.

Exact and parallel: every problem in every shard is parsed (a shard is
``concat(problems) -> chunk(768)``, so flattening its windows in row order rebuilds the
original stream and even boundary-straddling problems are counted). Shards are processed
**one per worker process**, so peak RAM is ~``workers x one decompressed shard``, never the
whole dataset -- fine for a 32 GB / 24-thread box on the ~32 GB corpus.

Usage
-----
    uv run python -m visualizations.dataset_difficulty \
        --data-dir "I:/.../data/igsm_train_117M" \
        --out visualizations/dataset_difficulty.png --json visualizations/dataset_difficulty.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Authoritative token layout / window size (shared with the generator).
from src.data.igsm import ANS_BOS, DEFAULT_CONTEXT_LENGTH, SOL_BOS

SEMI = 26  # GPT-2 BPE id for ';' -- one per operation in the solution (see module docstring)


def recover_op_counts(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover per-problem op-counts from a shard's flattened token stream.

    Args:
        flat: 1-D array of the shard's token ids (all windows concatenated in row order).

    Returns:
        ``(op_counts, start_positions)`` -- ``op_counts[k]`` is problem k's ``n_op`` (number
        of ``;`` in its ``[223]..[224]`` solution region); ``start_positions[k]`` is the flat
        index of its ``[223]`` (used to bin problems into 768-windows). Problems whose
        solution region is truncated at the very end of the shard are dropped.
    """
    sol = np.where(flat == SOL_BOS)[0]
    ans = np.where(flat == ANS_BOS)[0]
    semi = np.where(flat == SEMI)[0]
    j = np.searchsorted(ans, sol)  # match each [223] to the next [224]
    ok = j < len(ans)
    starts, ends = sol[ok], ans[j[ok]]
    op_counts = np.searchsorted(semi, ends) - np.searchsorted(semi, starts, side="right")
    return op_counts, starts


def _flatten_shard(path: str) -> tuple[np.ndarray, int]:
    """Read one shard and return ``(flat_token_stream, n_windows)`` without Python-list copies."""
    col = pq.read_table(path, columns=["input_ids"])["input_ids"]
    chunks = col.chunks
    list_arr = chunks[0] if len(chunks) == 1 else pa.concat_arrays(chunks)
    flat = np.asarray(list_arr.values.to_numpy(zero_copy_only=False))
    return flat, len(list_arr)


def scan_shard(path: str, ctx: int = DEFAULT_CONTEXT_LENGTH) -> dict:
    """Fully parse one shard: exact per-op problem counts + per-window difficulty mixing.

    Top-level (picklable) so it can be mapped across a process pool.
    """
    flat, n_windows = _flatten_shard(path)
    ops, starts = recover_op_counts(flat)
    if len(ops) == 0:
        return {"op_counts": {}, "distinct_per_window": {}, "n_windows": n_windows, "n_problems": 0}

    vals, cnts = np.unique(ops, return_counts=True)
    op_counts = {int(v): int(c) for v, c in zip(vals, cnts)}

    # distinct op-values per 768-window: sort by (window, op), then each (window, op) pair is a
    # contiguous run; count run-starts within each window's span.
    win = (starts // ctx).astype(np.int64)
    order = np.lexsort((ops, win))
    w, o = win[order], ops[order]
    win_start = np.r_[True, w[1:] != w[:-1]]
    pair_start = np.r_[True, (w[1:] != w[:-1]) | (o[1:] != o[:-1])]
    distinct = np.add.reduceat(pair_start.astype(np.int64), np.flatnonzero(win_start))
    dvals, dcnts = np.unique(distinct, return_counts=True)
    distinct_per_window = {int(v): int(c) for v, c in zip(dvals, dcnts)}

    return {
        "op_counts": op_counts,
        "distinct_per_window": distinct_per_window,
        "n_windows": n_windows,
        "n_problems": int(len(ops)),
    }


def scan_dataset(
    data_dir: str,
    *,
    workers: int | None = None,
    stride: int = 1,
    max_shards: int | None = None,
    ctx: int = DEFAULT_CONTEXT_LENGTH,
    verbose: bool = True,
) -> dict:
    """Exactly count problems per difficulty across a packed iGSM dataset, in parallel.

    Args:
        data_dir: Directory of ``batch_*.parquet`` shards.
        workers: Process-pool size (default: ``min(12, cpu_count)``). Each worker holds one
            decompressed shard at a time.
        stride: Take every ``stride``-th shard (1 = all shards).
        max_shards: Optionally cap the number of shards scanned.
        ctx: Window length (for binning problems into training contexts).
        verbose: Print progress.

    Returns:
        JSON-able result: exact ``op_problems`` (op -> problem count), the distribution of
        distinct op-values per 768-window, and totals.
    """
    shards = sorted(glob.glob(str(Path(data_dir) / "batch_*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no batch_*.parquet shards in {data_dir}")
    sel = shards[::stride][:max_shards]
    workers = workers or min(12, os.cpu_count() or 4)

    op_problems: Counter[int] = Counter()
    distinct_per_window: Counter[int] = Counter()
    n_problems = n_windows = 0
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_shard, p, ctx): p for p in sel}
        for fut in as_completed(futures):
            r = fut.result()
            for k, v in r["op_counts"].items():
                op_problems[int(k)] += v
            for k, v in r["distinct_per_window"].items():
                distinct_per_window[int(k)] += v
            n_problems += r["n_problems"]
            n_windows += r["n_windows"]
            done += 1
            if verbose and done % 50 == 0:
                print(f"  {done}/{len(sel)} shards", flush=True)

    total_windows = sum(distinct_per_window.values())
    return {
        "data_dir": data_dir,
        "n_shards_scanned": len(sel),
        "n_shards_total": len(shards),
        "workers": workers,
        "n_problems": n_problems,
        "n_windows": n_windows,
        "op_problems": dict(sorted(op_problems.items())),
        "distinct_ops_per_window": dict(sorted(distinct_per_window.items())),
        "single_op_window_fraction": distinct_per_window.get(1, 0) / total_windows if total_windows else None,
    }


def expected_min_pmf(max_op: int = 15) -> np.ndarray:
    """Paper target: pmf of ``min(a, b)`` with ``a, b ~ Uniform{1..max_op}`` (op sampled
    per problem). ``P(min=k) = ((M-k+1)^2 - (M-k)^2) / M^2``."""
    m = max_op
    return np.array([((m - k + 1) ** 2 - (m - k) ** 2) / m**2 for k in range(1, m + 1)])


def _op_val(d: dict, o: int) -> int:
    """Fetch op ``o`` from a dict whose keys may be int (in-process) or str (loaded JSON)."""
    return d.get(o, d.get(str(o), 0))


def plot_difficulty(result: dict, out: str | None = None, *, max_op: int = 15, show: bool = False):
    """Bar chart: exact problems-per-difficulty vs the paper's per-problem ``min`` target."""
    import matplotlib.pyplot as plt

    ops = np.arange(1, max_op + 1)
    counts = np.array([_op_val(result["op_problems"], o) for o in ops], float)
    measured = counts / counts.sum()
    expected = expected_min_pmf(max_op)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    w = 0.42
    ax.bar(ops - w / 2, measured, w, label="measured dataset", color="#4C78A8")
    ax.bar(ops + w / 2, expected, w, label="paper target: min(2 uniform) per problem",
           color="#E45756")
    ax.set_xlabel("difficulty (op count)")
    ax.set_ylabel("fraction of problems")
    ax.set_xticks(ops)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    frac = result.get("single_op_window_fraction")
    sub = f"{frac:.1%} of 768-token windows contain a single difficulty" if frac is not None else ""
    ax.set_title(f"iGSM difficulty distribution  ({result['n_shards_scanned']} shards, "
                 f"{result['n_problems']:,} problems)\n{sub}", fontsize=10)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140)
        print(f"saved {out}")
    if show:
        plt.show()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="directory of batch_*.parquet shards")
    p.add_argument("--workers", type=int, default=None, help="process-pool size (default min(12, cpus))")
    p.add_argument("--stride", type=int, default=1, help="scan every Nth shard (1 = all)")
    p.add_argument("--max-shards", type=int, default=None, help="cap number of shards scanned")
    p.add_argument("--out", default="visualizations/dataset_difficulty.png", help="output PNG path")
    p.add_argument("--json", default=None, help="optional path to also dump the raw result dict")
    p.add_argument("--show", action="store_true", help="show the figure interactively")
    args = p.parse_args()

    result = scan_dataset(
        args.data_dir, workers=args.workers, stride=args.stride, max_shards=args.max_shards,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "op_problems"}, indent=2))
    print("exact problems per op:", result["op_problems"])
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    plot_difficulty(result, out=args.out, show=args.show)


if __name__ == "__main__":
    main()
