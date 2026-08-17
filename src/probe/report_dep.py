"""Interactive HTML report for dep(A, B) probe predictions (see ``run.py report-dep``).

Two files in one directory, no server, no CDN::

    <out>/index.html        the UI, copied verbatim from src/probe/report/dep_report.html
    <out>/report_data.js    this module's only product: `window.REPORT = {...}`

They must travel together; opening ``index.html`` without its data file shows a message
saying so. The data is a plain ``.js`` assignment rather than ``.json`` because a page
opened from disk (``file://``) may not ``fetch`` a sibling file, while it may load one as
a script -- so the report works by double-clicking it, with no web server.

Layout: problem text on the left (with the probe's marker tokens rendered as chips), the
problem's dependency graph on the right -- every parameter as a node on a circle, every
tested (A, B) pair as a directed edge A -> B ("A depends on B"). The nodes are the problem's
candidate parameters (``src.probe.labels.named_params``), not iGSM's raw ``all_param``, so a
variable the problem never mentions is not drawn.

Encoding: **line style carries the true label** (solid = dependency exists, dashed = none),
**colour carries prediction correctness** (green = correct, red = wrong):

    true positive   solid  green      false negative  solid  red
    true negative   dashed green      false positive  dashed red

Wrong predictions additionally get an 'x' at the edge midpoint so correctness never rides
on the red/green channel alone (the worst colour-vision-deficiency pair). True negatives
dominate the natural distribution (~85-90%) and are hidden by default. Node fill encodes
the parameter's owner layer in iGSM's category hierarchy; a dashed ink ring marks
parameters necessary for the answer.

Interactions: hover a node to isolate its pairs; **click** fills the query slots -- first
click sets A (its mentions highlight in the problem text), second click sets B, matching
the probe's ``[START] A [MID] B [END]`` input. A grid of radio buttons selects the problem
(columns = difficulty as iGSM ``n_op``, rows = alternative seeds -- generate the dataset with
``gen-data --uniform-difficulty`` to populate the grid). Confusion matrices (per problem or
whole test set) and an n x n dependency-matrix view sit below the graph.

Data comes from the ``test`` command's saved predictions for *both* runs (pretrained +
random control) on the *same* ``--dep-all-pairs`` dataset. Problems are regenerated from
their seed and requested op for text and parameter names. Predictions are never recomputed here.

If the pretrained run also holds solutions for the dataset (``run.py solve``), each problem
carries what the model itself wrote and whether that was right, and the page can show it in
place of iGSM's reference solution. Without them the page shows the reference solution alone.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from src.probe.evaluate import classification_metrics, load_predictions

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).resolve().parent / 'report' / 'dep_report.html'
INDEX_NAME = 'index.html'
DATA_NAME = 'report_data.js'


def _short_name(problem: Any, param: tuple[int, int, int, int]) -> str:
    """Compact node label, e.g. 'Danc Stud→Back' (words clipped to 4 chars).

    Mirrors `Problem.get_param` naming: instance params (kind 0) point at a concrete child
    node `N[i+1][k]` (single arrow), abstract params (kind 1) at a whole category `ln[k]`
    (double arrow) -- the legend in the page explains the two arrow types.
    """
    kind, i, j, k = param
    clip = lambda name: ' '.join(w[:4] for w in str(name).split())  # noqa: E731
    owner = clip(problem.N[i][j])
    attr = clip(problem.N[i + 1][k]) if kind == 0 else clip(problem.ln[k])
    return f'{owner}{"→" if kind == 0 else "⇒"}{attr}'


def _n_op_by_seed(data_dir: Path) -> dict[int, int]:
    """Map problem seed (`group`) -> reasoning-step count, from the dataset's own shards.

    `n_op` is constant within a group, so the last value per group wins.
    """
    from src.probe.data import SHARD_GLOB

    n_op: dict[int, int] = {}
    for shard in sorted(data_dir.glob(SHARD_GLOB)):
        table = pq.read_table(str(shard), columns=['group', 'n_op'])
        n_op.update(zip(table.column('group').to_pylist(), table.column('n_op').to_pylist()))
    return n_op


def _problem_payload(
    seed: int,
    split: str,
    med_cfg: dict[str, Any] | None,
    edges: list[list[float]],
    n_op: int,
    op: int | None,
) -> dict[str, Any]:
    """Regenerate problem `seed` and package text + parameter names + its edge list.

    Requires `op` to match the value the dataset requested for this problem (None if it
    requested none). `n_op` is the recorded step count, checked against the regenerated one.
    """
    from src.probe.labels import regenerate_problem

    # the edge list addresses parameters by their index in pp.all_param
    pp = regenerate_problem(seed, split=split, med_cfg=med_cfg, op=op)
    problem = pp.problem
    if int(problem.n_op) != n_op:
        raise ValueError(
            f'problem {seed} regenerates with n_op={problem.n_op} but the dataset recorded '
            f'{n_op}. The graph would not match the predictions. Check that metadata.json '
            f'(uniform_difficulty, med_cfg) matches how the dataset was generated.'
        )
    nece = pp.nece
    params = [
        {
            's': _short_name(problem, param),
            'f': problem.get_param(param),
            'nece': int(nece[p_idx]),
            'layer': int(param[1]),  # owner layer i in the category hierarchy -> node colour
            'kind': int(param[0]),  # 0 instance (→), 1 abstract (⇒)
        }
        for p_idx, param in enumerate(pp.all_param)
    ]
    n = len(pp.all_param)
    return {
        'seed': seed,
        'nOp': int(problem.n_op),
        'layers': [str(name) for name in problem.ln],  # layer names, index = params[].layer
        'desc': [str(s) for s in problem.problem[:-1]],
        'question': str(problem.problem[-1]),
        'solution': [str(s) for s in problem.solution],
        'params': params,
        'nPairsExpected': n * (n - 1),
        'edges': edges,
    }


def _attach_solutions(
    problems: list[dict[str, Any]], run_dir: Path, data_dir: Path
) -> dict[str, Any] | None:
    """Add each problem's model solution (`p['model']`) in place; return the solving metadata.

    Returns None when the run has no solutions for this dataset, which the page treats as
    "reference solution only". Problems `solve` did not cover keep `p['model'] = None`.
    """
    from src.probe.solve import load_solutions

    rows, meta = load_solutions(run_dir, data_dir)
    if not rows:
        return None
    for p in problems:
        row = rows.get(p['seed'])
        p['model'] = row and {
            'correct': bool(row['correct']),
            'answerCorrect': bool(row['answer_correct']),
            'solution': row['solution'],
            'answer': row['answer'],
            'goldAnswer': int(row['gold_answer']),
        }
    shown = [p['model'] for p in problems if p.get('model')]
    return {
        'model': meta.get('model_path'),
        'solveRate': meta.get('solve_rate'),
        'nProblems': meta.get('n_problems'),
        'nShown': len(shown),
        'nShownCorrect': sum(m['correct'] for m in shown),
    }


def build_report(
    pretrained_run: Path | str,
    random_run: Path | str,
    data_dir: Path | str,
    *,
    n_problems: int = 48,
    out: Path | str | None = None,
) -> Path:
    """Assemble predictions from both runs into a report directory; return that directory.

    `out` defaults to a timestamped `results/probes/<date>_<time>_dep`, mirroring how `vprobe`
    names run directories, so rebuilding never overwrites an earlier report. A given `out` is
    written into, replacing files of the same name.

    Both runs must have been `test`ed on `data_dir` already (their saved prediction files
    must align row-for-row -- same dataset, same order -- which is asserted, not assumed).
    The report carries the first `n_problems` problems of the dataset, sorted by difficulty
    (`n_op`) for the selection grid; overall confusion matrices/metrics are computed over
    ALL rows, not just those problems.
    """
    data_dir = Path(data_dir)
    pretrained_run, random_run = Path(pretrained_run), Path(random_run)
    pre = load_predictions(pretrained_run, data_dir)
    rnd = load_predictions(random_run, data_dir)
    for col in ('group', 'param_a', 'param_b', 'label'):
        if not np.array_equal(pre[col], rnd[col]):
            raise ValueError(
                f'prediction files disagree on column {col!r} -- the two runs were not '
                f'tested on the same dataset/order ({pretrained_run} vs {random_run})'
            )
    if (pre['param_a'] < 0).any():
        raise ValueError(
            f'{data_dir} has no pair-identity columns (param_a=-1) -- regenerate it with '
            'a current `gen-data --target dep --dep-all-pairs`'
        )

    meta = json.loads((data_dir / 'metadata.json').read_text(encoding='utf-8'))
    split, med_cfg = meta.get('split', 'test'), meta.get('med_cfg')
    # A problem only comes back the same when the same op request goes in. Datasets generated
    # without a requested op must be regenerated without one, even at the observed n_op.
    n_op_by_seed = _n_op_by_seed(data_dir)
    request_op = bool(meta.get('uniform_difficulty'))
    if not meta.get('dep_all_pairs'):
        logger.warning(
            '%s was generated without --dep-all-pairs: the graph will only show the '
            'balanced training subsample of pairs, not the full dependency structure',
            data_dir,
        )

    overall = {
        'pre': classification_metrics(pre['label'], pre['pred']),
        'rand': classification_metrics(rnd['label'], rnd['pred']),
    }

    # model identity for the header, from the runs' own configs
    pre_cfg = json.loads((pretrained_run / 'config.json').read_text(encoding='utf-8'))['params']
    rnd_cfg = json.loads((random_run / 'config.json').read_text(encoding='utf-8'))['params']

    seeds = list(dict.fromkeys(pre['group'].tolist()))[:n_problems]  # first N, dataset order
    problems = []
    for seed in seeds:
        m = pre['group'] == seed
        edges = [
            [
                int(a), int(b), int(y),
                int(pp_), round(float(q), 3),
                int(rp_), round(float(r), 3),
            ]
            for a, b, y, pp_, q, rp_, r in zip(
                pre['param_a'][m], pre['param_b'][m], pre['label'][m],
                pre['pred'][m], pre['p1'][m], rnd['pred'][m], rnd['p1'][m], strict=True,
            )
        ]
        n_op = int(n_op_by_seed[int(seed)])
        problems.append(
            _problem_payload(int(seed), split, med_cfg, edges, n_op, n_op if request_op else None)
        )
        logger.info(
            'problem %d: op=%d, %d params, %d pairs',
            seed, problems[-1]['nOp'], len(problems[-1]['params']), len(edges),
        )
    problems.sort(key=lambda p: (p['nOp'], p['seed']))  # grid columns = difficulty
    solutions_meta = _attach_solutions(problems, pretrained_run, data_dir)

    payload = {
        'meta': {
            'model': pre_cfg.get('model_path') or '(unknown model)',
            'randomSeed': rnd_cfg.get('seed'),
            'data': str(data_dir),
            'pretrainedRun': pretrained_run.name,
            'randomRun': random_run.name,
            'created': datetime.now().astimezone().isoformat(timespec='seconds'),
            'nRowsTotal': int(len(pre['label'])),
            'nProblemsShown': len(problems),
            'solutions': solutions_meta,
        },
        'overall': overall,
        'problems': problems,
    }
    out = Path(out) if out else Path('results/probes') / f'{datetime.now():%Y-%m-%d_%H%M%S}_dep'
    out.mkdir(parents=True, exist_ok=True)
    (out / DATA_NAME).write_text(
        f'window.REPORT = {json.dumps(payload, separators=(",", ":"))};\n', encoding='utf-8'
    )
    shutil.copyfile(TEMPLATE_PATH, out / INDEX_NAME)
    logger.info(
        'report: %d problems, %d total rows -> %s', len(problems), len(pre['label']),
        out / INDEX_NAME,
    )
    return out

