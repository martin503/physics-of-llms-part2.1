#!/usr/bin/env python3
"""Eval a list of checkpoints on a list of slices (generation-based true_correct).

Generalized form: ``run_eval(targets, slices, results_path, ...)`` drives everything;
the module-level TARGETS/SLICES/RESULTS below are just the defaults for the original
6-run sweep (so ``python scripts/debug/sweep_eval.py`` still works unchanged).

Resumable: keeps non-error results in ``results_path`` and re-evals the rest.
Concurrency: 2 evals at a time, ONE per GPU via a GPU-pool semaphore (avoids the
OOM from two eval subprocesses landing on the same GPU). Per-slice true_correct
accuracy parsed from each inspect_ai JSON; inv_freq absmax recorded as a guard.

Each result record carries the full ``mean`` (slice->acc) and ``n`` (slice->count)
dicts, plus short top-level keys (``le15``/``eq15``/``eq20`` ...) for any slice whose
name ends in that suffix -- so legacy readers (sweep_report.py) keep working.
"""
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "sweeps" / "results.json"
SLICES = ["med_pq_op_le15", "med_pq_op_eq15", "med_pq_op_eq20"]
LR = {"gbs256_lr4e4": 4e-4, "gbs256_lr8e4": 8e-4, "gbs256_lr1e3": 1e-3,
      "gbs64_lr2e4": 2e-4, "gbs64_lr6e4": 6e-4, "gbs64_lr1e3": 1e-3, "base_100k": None}

_cfg = REPO / "sweeps" / "sweep_config.json"
TOKENS_TOTAL = json.loads(_cfg.read_text())["tokens_total"] if _cfg.exists() else None

TARGETS = [
    ("base_100k", "models/100k_model/gpt-rope-igsm-fixed"),
    ("gbs256_lr4e4", "sweeps/gbs256_lr4e4/final"),
    ("gbs256_lr8e4", "sweeps/gbs256_lr8e4/final"),
    ("gbs256_lr1e3", "sweeps/gbs256_lr1e3/final"),
    ("gbs64_lr2e4",  "sweeps/gbs64_lr2e4/final"),
    ("gbs64_lr6e4",  "sweeps/gbs64_lr6e4/final"),
    ("gbs64_lr1e3",  "sweeps/gbs64_lr1e3/final"),
]

# GPU pool: guarantees at most one eval subprocess per GPU at a time.
_POOL, _LOCK = [0, 1], threading.Condition()


def _get_gpu() -> int:
    with _LOCK:
        while not _POOL:
            _LOCK.wait()
        return _POOL.pop(0)


def _put_gpu(g: int) -> None:
    with _LOCK:
        _POOL.append(g)
        _LOCK.notify()


def inv_freq_absmax(model_dir: str) -> float:
    code = ("from src.model.gpt2_rope import GPT2LMHeadModelWithRoPE;"
            f"m=GPT2LMHeadModelWithRoPE.from_pretrained({model_dir!r});"
            "print(max(b.attn.rotary_emb.inv_freq.abs().max().item() for b in m.transformer.h))")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")  # CPU only
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO))
    try:
        return float(p.stdout.strip().splitlines()[-1])
    except Exception:
        return float("nan")


def parse_acc(json_path: str) -> tuple[dict, dict]:
    d = json.loads(Path(json_path).read_text())
    acc: dict[str, list[int]] = {}
    for s in d.get("samples", []):
        sl = s.get("metadata", {}).get("slice")
        val = s.get("scores", {}).get("igsm_correctness", {}).get("value")
        if sl is None or val is None:
            continue
        acc.setdefault(sl, []).append(int(val))
    return ({sl: (sum(v) / len(v) if v else float("nan")) for sl, v in acc.items()},
            {sl: len(v) for sl, v in acc.items()})


def run_eval(targets, slices, results_path, *, tokens_map=None, lr_map=None,
             limit=128, batch_size=64, max_workers=2) -> list[dict]:
    """Eval each (tag, model_dir) on `slices`; merge resumably into `results_path`.

    tokens_map / lr_map: optional dict[tag]->value attached to each record.
    """
    targets = list(targets)
    results_path = Path(results_path)
    existing: dict[str, dict] = {}
    if results_path.exists():
        for r in json.loads(results_path.read_text()):
            existing[r["tag"]] = r
    done = {t for t, r in existing.items() if "error" not in r}
    todo = [(t, m) for (t, m) in targets if t not in done]
    print(f"done={sorted(done)}\ntodo={[t for t, _ in todo]}", flush=True)

    def eval_one(tag: str, model_dir: str) -> dict:
        mdir = str(REPO / model_dir)
        if not Path(mdir).exists():
            return {"tag": tag, "model": model_dir, "error": f"missing: {mdir}"}
        invmax = inv_freq_absmax(mdir)  # CPU; holds no GPU
        gpu = _get_gpu()
        try:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [sys.executable, "-m", "src.eval.run", "--model", mdir,
                   "--slices", ",".join(slices), "--limit", str(limit), "--batch-size", str(batch_size)]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO))
        finally:
            _put_gpu(gpu)
        out = proc.stdout + "\n" + proc.stderr
        m = re.search(r"log=(\S+)", out)
        if not m or not Path(m.group(1)).exists():
            return {"tag": tag, "model": model_dir, "gpu": gpu, "inv_freq_absmax": invmax,
                    "error": "no eval log", "tail": (proc.stderr or proc.stdout)[-800:]}
        mean, n = parse_acc(m.group(1))
        rec = {"tag": tag, "model": model_dir, "gpu": gpu, "inv_freq_absmax": invmax,
               "lr": (lr_map or {}).get(tag), "tokens": (tokens_map or {}).get(tag),
               "mean": mean, "n": n}
        for sl in slices:  # short top-level keys (le15/eq15/eq20 ...)
            if sl in mean:
                rec[sl.rsplit("_", 1)[-1]] = mean[sl]
        return rec

    results = dict(existing)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(eval_one, t, m): t for t, m in todo}
        for f in as_completed(futs):
            r = f.result()
            results[r["tag"]] = r
            print(json.dumps(r), flush=True)
    ordered = [results[t] for t, _ in targets if t in results]
    results_path.write_text(json.dumps(ordered, indent=2))
    print(f"\nWrote {results_path}")
    corrupt = [r["tag"] for r in ordered if r.get("inv_freq_absmax", 0) > 1e3]
    if corrupt:
        print(f"WARNING: corrupt inv_freq in {corrupt} -- numbers unreliable")
    return ordered


def main() -> None:
    run_eval(TARGETS, SLICES, RESULTS, tokens_map={t: (0 if t == "base_100k" else TOKENS_TOTAL)
                                                    for t, _ in TARGETS}, lr_map=LR)


if __name__ == "__main__":
    main()
