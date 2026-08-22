"""Generate the iGSM-med evaluation datasets for Figure 3 (Part 2.1 reproduction).

A deliberately plain, single-process, non-resumable generator. For every
(p_format, op-spec) slice it writes ONE parquet file (one row per problem) to
``--out``. Each row carries the generation prompt, the gold solution/answer, the
op count, and the **pickled gold ``Problem``** so the eval can reuse iGSM's
``true_correct`` verbatim (answer + arithmetic + parameter dependencies).

Slices (med family: ip<=20, max_edge=20, perm_level=5, detail_level=0), x2 for pq/qp:
    op_le15 (in-distribution), op_eq15, op_eq20, op_eq21, op_eq22, op_eq23 (OOD),
    plus a reask slice (iGSM-med^{op=20,reask}: re-asks an exactly-op=20 problem) when --reask.

Token layout per problem (iGSM / GPT-2 BPE):
    prompt = [50256, 222, <problem>, 223]                              (= IdGen.prob_id)
    gold   = [222, <problem>, 223, <solution>, 224, <answer>, 50256]  (= IdGen.token_id)

Consumer notes (for the downstream inspect_ai eval):
- Scoring contract: feed ``prompt_ids`` to the model; it should generate the
  ``[solution, 224, answer, 50256]`` tail. Score with
  ``true_correct(prompt_ids + generated_ids, problem=pickle.loads(problem))`` -- ``true_correct``
  splits on 222/223/224 (skip_222=True), so the leading 50256 is fine. Do NOT pass generated-only
  (no sentinels -> silently 0%), and note prompt_ids is NOT a clean prefix of gold_token_id.
- The ``problem`` column is a pickled iGSM ``Problem``; call ``ensure_igsm_submodule()`` (puts iGSM
  on sys.path) BEFORE ``pickle.loads``, or it raises ``ModuleNotFoundError: math_gen``.
- ``reask`` slice: the base problem has exactly 20 ops (iGSM-med^{op=20}); ``op`` records the
  *re-asked* op count, not 20 -- reask changes the needed ops by design (footnote 10) -- and
  reask bypasses hash-bin validation (it is an OOD construction, footnote 10).
- Wrap ``true_correct`` in try/except in the scorer: it ``raise ValueError`` on the
  ``sol_op < n_op`` branch, which a malformed generation could trip.

Usage::

    uv run python -m src.data.eval --seed 0 --out data/igsm_eval
    uv run python -m src.data.eval --seed 0 --out data/igsm_eval --num-problems 8 --no-reask
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Annotated, Any

import pyarrow as pa
import pyarrow.parquet as pq
import typer
from tqdm import tqdm

from src.data.igsm import ANS_BOS, EOS, PROB_BOS, SOL_BOS, ensure_igsm_submodule, get_bins

# med-family base config (paper: perm_level=5 full shuffle, detail_level=0 most verbose).
MED_BASE: dict[str, Any] = dict(max_edge=20, perm_level=5, detail_level=0)

# Figure-3 op specs (name -> IdGen knobs). op=None + max_op=N => op<=N; op=K pins exact K ops.
OP_SPECS: list[tuple[str, dict[str, Any]]] = [
    ('op_le15', dict(max_op=15, op=None)),
    ('op_eq15', dict(max_op=15, op=15)),
    ('op_eq20', dict(max_op=20, op=20)),
    ('op_eq21', dict(max_op=21, op=21)),
    ('op_eq22', dict(max_op=22, op=22)),
    ('op_eq23', dict(max_op=23, op=23)),
]
# reask: base problem from iGSM-med^{op=20} (exactly 20 ops -- first OOD level), then resample the
# query (paper Section 2.4, Figure 3's "op=20 (reask)" column; footnote 10).
REASK_CFG: dict[str, Any] = dict(max_op=20, op=20)
FORMATS = ('pq', 'qp')
# Max base-problem regenerations when iGSM's buggy re_ask rejects a problem (see _generate_slice).
_REASK_MAX_ATTEMPTS = 100


def _generate_slice(
    name: str,
    cfg: dict[str, Any],
    p_format: str,
    reask: bool,
    num_problems: int,
    seed: int,
    bins: list[int],
) -> list[dict[str, Any]]:
    """Generate ``num_problems`` records for one slice (single process, seeded by ``seed``)."""
    from data_gen.pretrain.id_gen import IdGen  # type: ignore[import-not-found]
    from tools.tools import fix_seed  # type: ignore[import-not-found]
    from tools.tools_test import re_ask  # type: ignore[import-not-found]

    fix_seed(seed)  # iGSM uses module-level RNG -> a per-slice seed makes each slice reproducible
    pinned_op = cfg.get('op')

    records: list[dict[str, Any]] = []
    for _ in tqdm(range(num_problems), desc=name, unit='prob', leave=False):
        # A FRESH IdGen is built per problem (per attempt) for two reasons:
        #  (1) IdGen samples `op_` exactly once in __init__ and gen_prob only accepts problems with
        #      n_op == op_, so one instance emits only a SINGLE op count. Re-instantiating per
        #      problem re-draws op_, reproducing the op<=N spread for op_le15 (the reask base is
        #      pinned at op=20, so its fresh IdGen re-draws the same count)
        #      (src/data/igsm.py now also re-instantiates IdGen per problem for the same reason).
        #  (2) re_ask overwrites gen.op_ with the re-asked op count, which can EXCEED max_op; if
        #      that leaked into the next gen_prob, its `while n_op != op_` loop could never
        #      terminate (n_op is capped at max_op) -> an infinite hang. Fresh = no leak.
        # iGSM's re_ask is also itself buggy (~20% crash even with the whole_template fix below),
        # so we retry on crash with a fresh base problem.
        gen: Any = None
        ok = False
        for _attempt in range(_REASK_MAX_ATTEMPTS):
            gen = IdGen(**cfg)
            gen.gen_prob(bins, p_format=p_format)
            if pinned_op is not None:
                # Check the BASE op pre-reask: re_ask later overwrites n_op with the re-asked count
                # (footnote 10), which is generally != the pinned op.
                assert gen.problem.n_op == pinned_op, f'{name}: op pinning failed'
            if not reask:
                ok = True
                break
            # re_ask restores `problem.template` to the solution subgraph then reads
            # `problem.template.predecessors`, which KeyErrors off-subgraph; scoring only ever uses
            # `problem.whole_template` (sol_parser.py), so point template at the full graph first.
            gen.problem.template = gen.problem.whole_template  # type: ignore[union-attr]
            try:
                re_ask(gen, ava_hash=bins, p_format=p_format)
                ok = True
                break
            except Exception:
                continue  # re_ask bug for this (problem, query); regenerate and retry
        assert ok, f'{name}: re_ask failed after {_REASK_MAX_ATTEMPTS} attempts (iGSM bug)'
        if reask:
            # re_ask updates id_gen.op_ (the re-asked op count) but NOT problem.n_op, which
            # true_correct compares against parser.sol_op -> sync it so the gold scores correct.
            gen.problem.n_op = gen.op_  # type: ignore[union-attr]

        prompt_ids = list(gen.prob_id)  # type: ignore[union-attr]
        gold_token_id = list(gen.token_id)  # type: ignore[union-attr]
        assert prompt_ids[0] == EOS and prompt_ids[-1] == SOL_BOS, f'{name}: bad prompt layout'
        assert gold_token_id[0] == PROB_BOS and gold_token_id[-1] == EOS, (
            f'{name}: bad gold layout'
        )
        assert SOL_BOS in gold_token_id and ANS_BOS in gold_token_id, f'{name}: missing sentinels'

        records.append(
            {
                'slice': name,
                'p_format': p_format,
                'reask': reask,
                'op': int(gen.problem.n_op),  # type: ignore[union-attr]
                'gold_answer': int(gen.problem.ans),  # type: ignore[union-attr]
                'prompt_ids': prompt_ids,
                'gold_token_id': gold_token_id,
                'problem': pickle.dumps(gen.problem),  # type: ignore[union-attr]
            }
        )
    return records


def _write_slice(records: list[dict[str, Any]], path: Path) -> None:
    """Write one slice's records to a single parquet file at ``path``."""
    table = pa.table(
        {
            'slice': [r['slice'] for r in records],
            'p_format': [r['p_format'] for r in records],
            'reask': [r['reask'] for r in records],
            'op': pa.array([r['op'] for r in records], pa.int64()),
            'gold_answer': pa.array([r['gold_answer'] for r in records], pa.int64()),
            'prompt_ids': pa.array([r['prompt_ids'] for r in records], pa.list_(pa.int64())),
            'gold_token_id': pa.array([r['gold_token_id'] for r in records], pa.list_(pa.int64())),
            'problem': pa.array([r['problem'] for r in records], pa.binary()),
        }
    )
    pq.write_table(table, str(path))


app = typer.Typer(add_completion=False, help='Generate iGSM-med Figure-3 eval datasets (plain).')


@app.command()
def generate(
    seed: Annotated[int, typer.Option('--seed', help='Base RNG seed; slice i uses seed+i.')] = 0,
    out: Annotated[
        Path, typer.Option('--out', help='Output dir; one <slice>.parquet written per slice.')
    ] = Path('data/igsm_eval'),
    num_problems: Annotated[
        int, typer.Option('--num-problems', help='Problems per slice (paper uses 4096).')
    ] = 4096,
    reask: Annotated[
        bool, typer.Option('--reask/--no-reask', help='Also generate the reask slice(s).')
    ] = True,
) -> None:
    """Generate all iGSM-med Figure-3 eval slices (pq & qp) as one parquet file each."""
    ensure_igsm_submodule()
    out.mkdir(parents=True, exist_ok=True)
    bins = get_bins('test')  # eval bins [16..22] -> no template overlap with train bins [0..15]

    slices: list[tuple[str, dict[str, Any], str, bool]] = []
    for p_format in FORMATS:
        for spec_name, spec_cfg in OP_SPECS:
            slices.append(
                (f'med_{p_format}_{spec_name}', {**MED_BASE, **spec_cfg}, p_format, False)
            )
        if reask:
            slices.append((f'med_{p_format}_reask', {**MED_BASE, **REASK_CFG}, p_format, True))

    for i, (name, cfg, p_format, is_reask) in enumerate(slices):
        records = _generate_slice(name, cfg, p_format, is_reask, num_problems, seed + i, bins)
        _write_slice(records, out / f'{name}.parquet')
        typer.echo(f'  wrote {len(records)} problems -> {out / f"{name}.parquet"}')


if __name__ == '__main__':
    app()
