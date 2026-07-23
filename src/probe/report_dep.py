"""Interactive HTML report for dep(A, B) probe predictions (see ``run.py report-dep``).

One standalone file, no server, no external assets -- host it anywhere static (GitHub
Pages: copy to ``docs/index.html`` and enable Pages on the repo). Layout: problem text on
the left (with the probe's marker tokens rendered as chips), the problem's dependency
graph on the right -- every parameter as a node on a circle, every tested (A, B) pair as
a directed edge A -> B ("A depends on B").

Encoding (user spec): **line style carries the true label** (solid = dependency exists,
dashed = none), **colour carries prediction correctness** (green = correct, red = wrong):

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
(columns = difficulty as iGSM ``n_op``, rows = alternative seeds -- generate the dataset
from ``find-seeds`` output to populate the grid). Confusion matrices (per problem or whole
test set) and an n x n dependency-matrix view sit below the graph.

Data comes from the ``test`` command's saved predictions for *both* runs (pretrained +
random control) on the *same* ``--dep-all-pairs`` dataset. Problems are regenerated from
their seeds for text and parameter names -- predictions are never recomputed here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.probe.evaluate import classification_metrics, load_predictions

logger = logging.getLogger(__name__)


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


def _problem_payload(
    seed: int,
    split: str,
    med_cfg: dict[str, Any] | None,
    edges: list[list[float]],
) -> dict[str, Any]:
    """Regenerate problem `seed` and package text + parameter names + its edge list."""
    from src.probe.labels import regenerate_problem

    pp = regenerate_problem(seed, split=split, med_cfg=med_cfg)
    problem = pp.problem
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


def build_report(
    pretrained_run: Path | str,
    random_run: Path | str,
    data_dir: Path | str,
    *,
    n_problems: int = 48,
    out: Path | str = Path('visualizations/dep_probe_report.html'),
) -> Path:
    """Assemble predictions from both runs into the standalone HTML report at `out`.

    Both runs must have been `test`ed on `data_dir` already (their saved prediction files
    must align row-for-row -- same dataset, same order -- which is asserted, not assumed).
    The report embeds the first `n_problems` problems of the dataset, sorted by difficulty
    (`n_op`) for the selection grid; overall confusion matrices/metrics are computed over
    ALL rows, not just the embedded problems.
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
        problems.append(_problem_payload(int(seed), split, med_cfg, edges))
        logger.info(
            'problem %d: op=%d, %d params, %d pairs',
            seed, problems[-1]['nOp'], len(problems[-1]['params']), len(edges),
        )
    problems.sort(key=lambda p: (p['nOp'], p['seed']))  # grid columns = difficulty

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
        },
        'overall': overall,
        'problems': problems,
    }
    html = _TEMPLATE.replace(
        '__PAYLOAD__', json.dumps(payload, separators=(',', ':')).replace('</', '<\\/')
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding='utf-8')
    logger.info('report: %d problems, %d total rows -> %s', len(problems), len(pre['label']), out)
    return out


# --------------------------------------------------------------------------- #
# Template. Palette: status green/red (validated reference palette) for correct/
# incorrect; categorical slots (blue/aqua/yellow/violet/magenta) for hierarchy
# layers; line style carries the true label; 'x' markers back up the red/green
# channel. Light + dark via prefers-color-scheme.
# --------------------------------------------------------------------------- #

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dep(A, B) probe report</title>
<style>
  :root, :root[data-theme="light"] {
    color-scheme: light;
    --surface: #fcfcfb; --page: #f9f9f7;
    --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
    --good: #0ca30c; --bad: #d03b3b; --accent: #2a78d6; --accent-b: #4a3aa7;
    --chip-bg: #eceae4;
    --lay-0: #2a78d6; --lay-1: #1baf7a; --lay-2: #eda100; --lay-3: #4a3aa7; --lay-4: #e87ba4;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface: #1a1a19; --page: #0d0d0d;
      --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
      --good: #0ca30c; --bad: #e66767; --accent: #3987e5; --accent-b: #9085e9;
      --chip-bg: #2c2c2a;
      --lay-0: #3987e5; --lay-1: #199e70; --lay-2: #c98500; --lay-3: #9085e9; --lay-4: #d55181;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface: #1a1a19; --page: #0d0d0d;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    --good: #0ca30c; --bad: #e66767; --accent: #3987e5; --accent-b: #9085e9;
    --chip-bg: #2c2c2a;
    --lay-0: #3987e5; --lay-1: #199e70; --lay-2: #c98500; --lay-3: #9085e9; --lay-4: #d55181;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--page); color: var(--ink);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1440px; margin: 0 auto; padding: 16px 20px 40px; }
  .topbar { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
  h1 { font-size: 19px; margin: 0 0 10px; }
  .theme-btn {
    border: 1px solid var(--border); border-radius: 8px; background: var(--surface);
    color: var(--ink-2); padding: 5px 12px; font: 12px inherit; cursor: pointer;
  }
  .theme-btn:hover { color: var(--ink); }
  h2 { font-size: 15px; margin: 0 0 8px; }
  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 16px; margin-bottom: 14px;
  }
  /* labelled header meta */
  .meta { display: grid; grid-template-columns: auto 1fr; gap: 2px 14px; font-size: 13px; }
  .meta dt { color: var(--muted); }
  .meta dd { margin: 0; color: var(--ink-2); }
  .meta dd b { color: var(--ink); font-weight: 600; }
  /* controls */
  .controls { display: flex; flex-wrap: wrap; gap: 18px 36px; align-items: flex-start; }
  .control h2 { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
  .seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .seg button {
    border: 0; background: transparent; color: var(--ink-2); padding: 5px 12px;
    font: inherit; cursor: pointer;
  }
  .seg button.on { background: var(--accent); color: #fff; }
  .navgrid { display: flex; flex-direction: column; gap: 6px; }
  /* problem grid: columns = op count, rows = seed slots */
  .probgrid { border-collapse: collapse; }
  .probgrid th {
    font-weight: 400; color: var(--muted); font-size: 11px; padding: 2px 3px; text-align: center;
  }
  .probgrid th.rowlab { text-align: right; padding-right: 7px; }
  .probgrid td { padding: 2px 3px; text-align: center; }
  .probgrid button {
    width: 20px; height: 20px; border-radius: 50%; border: 1.5px solid var(--muted);
    background: transparent; cursor: pointer; padding: 0; vertical-align: middle;
  }
  .probgrid button:hover { border-color: var(--accent); }
  .probgrid button.on { background: var(--accent); border-color: var(--accent); }
  /* category toggles laid out like the confusion matrices below */
  .cattbl { border-collapse: collapse; }
  .cattbl th { font-weight: 400; color: var(--muted); font-size: 11px; padding: 2px 6px; }
  .cattbl td { padding: 2px; }
  .cattbl button {
    display: flex; flex-direction: column; align-items: center; gap: 2px;
    width: 74px; padding: 5px 4px; border: 1px solid var(--border); border-radius: 7px;
    background: transparent; color: var(--ink-2); font: 12px/1.2 inherit; cursor: pointer;
  }
  .cattbl button.on { background: var(--chip-bg); border-color: var(--muted); color: var(--ink); }
  .cattbl button svg { display: block; }
  /* main panels */
  .main { display: flex; gap: 14px; align-items: stretch; }
  .panel {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px; min-width: 0;
  }
  .left-col { display: flex; flex-direction: column; gap: 14px; flex: 1 1 40%; min-width: 0; }
  .text-panel { flex: 1 1 auto; min-height: 0; overflow-y: auto; }
  .cm-card { flex: 0 0 auto; margin-bottom: 0; padding: 12px 12px; }
  .graph-panel { flex: 1 1 60%; }
  .chip {
    display: inline-block; background: var(--chip-bg); color: var(--ink-2);
    border: 1px solid var(--border); border-radius: 5px; padding: 0 5px; margin: 0 2px;
    font: 11px/1.7 ui-monospace, Consolas, monospace; vertical-align: 1px; white-space: nowrap;
  }
  .chip.varA { color: var(--accent); font-weight: 700; }
  .chip.varB { color: var(--accent-b); font-weight: 700; }
  .desc { margin: 8px 0; }
  .desc mark { border-radius: 3px; color: inherit; padding: 0 1px; }
  .desc mark.hlA { background: color-mix(in srgb, var(--accent) 28%, transparent); }
  .desc mark.hlB { background: color-mix(in srgb, var(--accent-b) 30%, transparent); }
  .dropnote { color: var(--muted); font-size: 11px; border-top: 1px dashed var(--grid); margin-top: 10px; padding-top: 8px; }
  .question { color: var(--muted); font-style: italic; margin-top: 4px; }
  .stats { color: var(--muted); font-size: 12px; margin: 2px 0 10px; }
  .hint { color: var(--muted); font-size: 12px; margin-top: 8px; }
  svg.graph { width: 100%; height: auto; display: block; }
  .node circle { stroke-width: 1.4; cursor: pointer; }
  .node.selA circle { stroke: var(--accent); stroke-width: 3; }
  .node.selB circle { stroke: var(--accent-b); stroke-width: 3; }
  .node circle.necering { fill: none; stroke: var(--ink); stroke-width: 1; stroke-dasharray: 2 2; pointer-events: none; }
  .node text { font-size: 9px; fill: var(--ink-2); pointer-events: none; }
  .node.dim { opacity: 0.22; }
  .edge { fill: none; }
  .edgeg.dim { opacity: 0.06; }
  .xmark { stroke-width: 1.6; }
  /* legend table under the graph */
  .legend { margin-top: 10px; border-top: 1px solid var(--grid); padding-top: 8px; }
  .legend table { border-collapse: collapse; width: 100%; font-size: 12px; color: var(--ink-2); }
  .legend th {
    text-align: left; font-weight: 600; color: var(--muted); font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.05em; padding: 2px 12px 4px 0; vertical-align: top;
  }
  .legend td { padding: 2px 12px 2px 0; vertical-align: top; }
  .legend .lrow { display: flex; align-items: center; gap: 7px; margin: 2px 0; }
  .legend svg { flex: none; }
  #tooltip {
    position: fixed; display: none; pointer-events: none; z-index: 10;
    background: var(--surface); color: var(--ink); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 10px; font-size: 12px; max-width: 340px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
  }
  #tooltip .row { display: flex; gap: 8px; justify-content: space-between; }
  #tooltip .muted { color: var(--muted); }
  .cm-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
  .cm {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 12px;
  }
  .cm h3 { margin: 0 0 2px; font-size: 13px; }
  .cm .metrics { color: var(--ink-2); font-size: 12px; margin-bottom: 8px; }
  .cm table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
  .cm th { font-weight: 400; color: var(--muted); font-size: 11px; padding: 3px 5px; }
  .cm td {
    border: 1px solid var(--grid); text-align: right; padding: 5px 7px; min-width: 58px;
    font-size: 13px;
  }
  .cm td .pct { color: var(--muted); font-size: 11px; margin-left: 5px; }
  .cm td.correct { background: color-mix(in srgb, var(--good) 14%, var(--surface)); }
  .cm td.wrong   { background: color-mix(in srgb, var(--bad) 12%, var(--surface)); }
  /* dependency matrix view */
  details { margin-top: 14px; color: var(--ink-2); }
  details summary { cursor: pointer; }
  .mx-scroll { overflow-x: auto; margin-top: 10px; padding-top: 4px; }
  .mx { border-collapse: collapse; }
  .mx th.coltop { height: 96px; position: relative; padding: 0; min-width: 15px; }
  .mx th.coltop > span {
    position: absolute; bottom: 4px; left: 50%;
    transform: rotate(-45deg); transform-origin: bottom left;
    font-size: 9px; line-height: 1; font-weight: 400; color: var(--muted); white-space: nowrap;
  }
  .mx th.rowlab {
    text-align: right; font-size: 9px; line-height: 1; font-weight: 400; color: var(--muted);
    padding: 0 5px 0 0; white-space: nowrap;
  }
  .mx td { width: 15px; height: 15px; padding: 0; border: 1px solid var(--surface); line-height: 0; }
  .mx td.tp { background: var(--good); }
  .mx td.tn { background: color-mix(in srgb, var(--good) 22%, var(--surface)); }
  .mx td.fn { background: var(--bad); }
  .mx td.fp { background: color-mix(in srgb, var(--bad) 30%, var(--surface)); }
  .mx td.na { background: var(--chip-bg); }
  .mx-legend { display: flex; gap: 16px; font-size: 12px; margin-top: 8px; align-items: center; }
  .mx-legend .sw { width: 13px; height: 13px; display: inline-block; border-radius: 3px; margin-right: 5px; vertical-align: -2px; }
  @media (max-width: 1020px) { .main { flex-direction: column; } .text-panel { max-height: 480px; } }
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <h1>dep(A, B) V-probe — predictions vs ground truth</h1>
    <button class="theme-btn" id="themeToggle" title="toggle light/dark theme"></button>
  </div>

  <div class="card controls">
    <div class="control">
      <h2>probed model</h2>
      <span class="seg" id="modelSeg">
        <button data-model="pre" class="on">pretrained</button>
        <button data-model="rand">random control</button>
      </span>
    </div>
    <div class="control">
      <h2>problem</h2>
      <table class="probgrid" id="probGrid"></table>
    </div>
    <div class="control">
      <h2>navigate</h2>
      <div class="navgrid">
        <span class="seg">
          <button id="prevProb" title="previous problem at this difficulty">▲ problem</button>
          <button id="nextProb" title="next problem at this difficulty">problem ▼</button>
        </span>
        <span class="seg">
          <button id="prevDiff" title="previous difficulty">◀ difficulty</button>
          <button id="nextDiff" title="next difficulty">difficulty ▶</button>
        </span>
      </div>
    </div>
    <div class="control">
      <h2>shown outcomes</h2>
      <table class="cattbl">
        <tr><th></th><th>pred 1</th><th>pred 0</th></tr>
        <tr><th>dep&thinsp;=&thinsp;1</th><td><button data-cat="tp" class="on"></button></td><td><button data-cat="fn" class="on"></button></td></tr>
        <tr><th>dep&thinsp;=&thinsp;0</th><td><button data-cat="fp" class="on"></button></td><td><button data-cat="tn"></button></td></tr>
      </table>
    </div>
    <div class="control">
      <h2>hide variables</h2>
      <label class="hint" style="display:block"><input type="checkbox" id="hideIsolated" checked> hide unconnected</label>
      <label class="hint" style="display:block;margin-top:4px"><input type="checkbox" id="showOnlyNecessary"> show only necessary</label>
    </div>
  </div>

  <div class="main">
    <div class="left-col">
      <div class="panel text-panel">
        <h2>Probe input</h2>
        <div class="stats" id="probStats"></div>
        <div>
          <span class="chip">[EOS]</span><span class="chip">[BOS]</span>
          <span class="desc" id="desc"></span>
          <span class="chip">[START]</span><span class="chip varA" id="chipA">A</span><span class="chip">[MID]</span><span class="chip varB" id="chipB">B</span><span class="chip">[END]</span>
        </div>
        <div class="dropnote">the following problem sections are not passed to the model:</div>
        <div class="question" id="question"></div>
        <div class="question" id="solution"></div>
        <div class="hint">click a variable in the graph to set <b>A</b> (highlights its mentions),
          click a second one to set <b>B</b>; click A again to clear.</div>
      </div>
      <div class="card cm-card">
        <h2>Confusion matrices</h2>
        <span class="seg" id="scopeSeg">
          <button data-scope="problem" class="on">this problem</button>
          <button data-scope="all">whole test set</button>
        </span>
        <div class="cm-grid" id="cmGrid"></div>
      </div>
    </div>
    <div class="panel graph-panel">
      <svg id="graph" class="graph" viewBox="0 0 760 760" role="img" aria-label="dependency graph"></svg>
      <div class="legend" id="legend"></div>
    </div>
  </div>

  <div class="card" style="margin-top:14px">
    <details id="matrixView" open>
      <summary>full dependency matrix — rows: A, columns: B, cell: outcome of dep(A, B)</summary>
      <div class="mx-legend" id="mxLegend"></div>
      <div class="mx-scroll" id="mxWrap"></div>
    </details>
  </div>

  <div class="card" style="margin-top:14px">
    <dl class="meta" id="meta"></dl>
  </div>
</div>
<div id="tooltip"></div>

<script>
const DATA = __PAYLOAD__;

// show/hideIsolated are read from the buttons/checkbox at load so browser form-state
// restoration can never desync the controls from what is drawn
const state = { model: 'pre', prob: 0, show: {}, hideIsolated: true, showOnlyNecessary: false,
                selA: null, selB: null, hover: null, scope: 'problem' };
document.querySelectorAll('.cattbl button').forEach(b => {
  state.show[b.dataset.cat] = b.classList.contains('on');
});
state.hideIsolated = document.getElementById('hideIsolated').checked;
state.showOnlyNecessary = document.getElementById('showOnlyNecessary').checked;

const svg = document.getElementById('graph');
const tooltip = document.getElementById('tooltip');
const NS = 'http://www.w3.org/2000/svg';
const CX = 380, CY = 380, R = 272, RLAB = 286;
const CAT_NAMES = { tp: 'TP', fn: 'FN', fp: 'FP', tn: 'TN' };

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

/* ---------- theme toggle ---------- */
const themeBtn = document.getElementById('themeToggle');
function activeTheme() {
  return document.documentElement.getAttribute('data-theme') ||
    (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
}
function applyTheme(theme, rerender) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('depReportTheme', theme);
  themeBtn.textContent = theme === 'dark' ? '☀ light mode' : '🌙 dark mode';
  if (rerender) renderAll();
}
themeBtn.addEventListener('click', () => applyTheme(activeTheme() === 'dark' ? 'light' : 'dark', true));
const savedTheme = localStorage.getItem('depReportTheme');
applyTheme(savedTheme || activeTheme(), false);
function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;'); }
function prob() { return DATA.problems[state.prob]; }
// edge tuple: [a, b, label, predPre, p1Pre, predRand, p1Rand]
function edgePred(e) { return state.model === 'pre' ? e[3] : e[5]; }
function edgeP1(e)   { return state.model === 'pre' ? e[4] : e[6]; }
function edgeCat(e) {
  const y = e[2], p = edgePred(e);
  return y === 1 ? (p === 1 ? 'tp' : 'fn') : (p === 1 ? 'fp' : 'tn');
}
function catStyles() {
  const good = css('--good'), bad = css('--bad');
  return {
    tp: { stroke: good, dash: null,  width: 2.1, wrong: false },
    fn: { stroke: bad,  dash: null,  width: 2.1, wrong: true  },
    fp: { stroke: bad,  dash: '5 4', width: 1.5, wrong: true  },
    tn: { stroke: good, dash: '5 4', width: 1.2, wrong: false },
  };
}
function layerColor(i) { return css('--lay-' + Math.min(i, 4)); }

// small inline SVG line sample (for legends and toggle buttons)
function lineSample(cat, w = 46, h = 12) {
  const st = catStyles()[cat];
  const dash = st.dash ? ` stroke-dasharray="${st.dash}"` : '';
  const x = st.wrong
    ? `<path d="M ${w/2-3} ${h/2-3} L ${w/2+3} ${h/2+3} M ${w/2-3} ${h/2+3} L ${w/2+3} ${h/2-3}" stroke="${st.stroke}" stroke-width="1.5" fill="none"/>`
    : '';
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">` +
    `<line x1="1" y1="${h/2}" x2="${w-1}" y2="${h/2}" stroke="${st.stroke}" stroke-width="${st.width}"${dash}/>${x}</svg>`;
}

/* ---------- header meta ---------- */
document.getElementById('meta').innerHTML =
  `<dt>interpreted model</dt><dd><b>${esc(DATA.meta.model)}</b> (frozen GPT2-RoPE trained on iGSM-med) — its knowledge of “does A depend on B?” is read out by a V-probe at the end of the problem description</dd>` +
  `<dt>probe runs</dt><dd>pretrained: ${esc(DATA.meta.pretrainedRun)} · control: ${esc(DATA.meta.randomRun)} (same probe on a random-init model, seed ${DATA.meta.randomSeed})</dd>` +
  `<dt>test data</dt><dd>${esc(DATA.meta.data)} — ${DATA.meta.nRowsTotal.toLocaleString()} (A, B) pairs over held-out problems; ${DATA.meta.nProblemsShown} problems shown · generated ${DATA.meta.created}</dd>`;

/* ---------- problem selection grid ---------- */
function buildProbGrid() {
  const ops = [...new Set(DATA.problems.map(p => p.nOp))].sort((a, b) => a - b);
  const byOp = new Map(ops.map(op => [op, []]));
  DATA.problems.forEach((p, i) => byOp.get(p.nOp).push(i));
  const nRows = Math.max(...ops.map(op => byOp.get(op).length));
  let html = '<tr><th class="rowlab">ops:</th>' + ops.map(op => `<th>${op}</th>`).join('') + '</tr>';
  for (let r = 0; r < nRows; r++) {
    html += `<tr><th class="rowlab">${r === 0 ? 'problem' : ''}</th>`;
    for (const op of ops) {
      const idx = byOp.get(op)[r];
      html += idx === undefined
        ? '<td></td>'
        : `<td><button data-prob="${idx}" title="seed ${DATA.problems[idx].seed} · ${DATA.problems[idx].params.length} variables"></button></td>`;
    }
    html += '</tr>';
  }
  const grid = document.getElementById('probGrid');
  grid.innerHTML = html;
  grid.querySelectorAll('button').forEach(b =>
    b.addEventListener('click', () => {
      state.prob = +b.dataset.prob; state.selA = state.selB = null; state.hover = null;
      renderAll();
    }));
}
function syncProbGrid() {
  document.querySelectorAll('#probGrid button').forEach(b =>
    b.classList.toggle('on', +b.dataset.prob === state.prob));
}

/* ---------- controls ---------- */
document.querySelectorAll('#modelSeg button').forEach(b =>
  b.addEventListener('click', () => {
    state.model = b.dataset.model;
    document.querySelectorAll('#modelSeg button').forEach(x => x.classList.toggle('on', x === b));
    renderAll();
  }));
document.querySelectorAll('#scopeSeg button').forEach(b =>
  b.addEventListener('click', () => {
    state.scope = b.dataset.scope;
    document.querySelectorAll('#scopeSeg button').forEach(x => x.classList.toggle('on', x === b));
    renderMatrices();
  }));
document.querySelectorAll('.cattbl button').forEach(b => {
  b.innerHTML = `${lineSample(b.dataset.cat)}<span>${CAT_NAMES[b.dataset.cat]}</span>`;
  b.addEventListener('click', () => {
    state.show[b.dataset.cat] = !state.show[b.dataset.cat];
    b.classList.toggle('on', state.show[b.dataset.cat]);
    renderGraph(); renderMatrixView();
  });
});
document.getElementById('hideIsolated').addEventListener('change', e => {
  state.hideIsolated = e.target.checked; renderGraph();
});
document.getElementById('showOnlyNecessary').addEventListener('change', e => {
  state.showOnlyNecessary = e.target.checked; renderGraph();
});

/* ---------- problem/difficulty navigation ---------- */
function opsAndGroups() {
  const ops = [...new Set(DATA.problems.map(p => p.nOp))].sort((a, b) => a - b);
  const byOp = new Map(ops.map(op => [op, []]));
  DATA.problems.forEach((p, i) => byOp.get(p.nOp).push(i));
  return { ops, byOp };
}
function gotoProblem(delta) {
  const { ops, byOp } = opsAndGroups();
  const col = ops.indexOf(prob().nOp);
  const list = byOp.get(ops[col]);
  const row = list.indexOf(state.prob);
  state.prob = list[(row + delta + list.length) % list.length];
  state.selA = state.selB = null; state.hover = null;
  renderAll();
}
function gotoDifficulty(delta) {
  const { ops, byOp } = opsAndGroups();
  const col = ops.indexOf(prob().nOp);
  const list = byOp.get(ops[col]);
  const row = list.indexOf(state.prob);
  const nCol = (col + delta + ops.length) % ops.length;
  const nList = byOp.get(ops[nCol]);
  state.prob = nList[Math.min(row, nList.length - 1)];
  state.selA = state.selB = null; state.hover = null;
  renderAll();
}
document.getElementById('prevProb').addEventListener('click', () => gotoProblem(-1));
document.getElementById('nextProb').addEventListener('click', () => gotoProblem(1));
document.getElementById('prevDiff').addEventListener('click', () => gotoDifficulty(-1));
document.getElementById('nextDiff').addEventListener('click', () => gotoDifficulty(1));

/* ---------- graph ---------- */
let edgeEls = [], nodeEls = [];

function nodePos(i, n) {
  const a = -Math.PI / 2 + (2 * Math.PI * i) / n;
  return { x: CX + R * Math.cos(a), y: CY + R * Math.sin(a), a };
}

function renderGraph() {
  const p = prob();
  const n = p.params.length;
  svg.innerHTML = '';
  edgeEls = []; nodeEls = [];
  const nodeR = Math.max(3.5, Math.min(9, 260 / n));
  const styles = catStyles();

  const visible = p.edges.filter(e => state.show[edgeCat(e)] &&
    (!state.showOnlyNecessary || (p.params[e[0]].nece && p.params[e[1]].nece)));
  const connected = new Set();
  visible.forEach(e => { connected.add(e[0]); connected.add(e[1]); });

  const defs = document.createElementNS(NS, 'defs');
  for (const cat of ['tp', 'fn', 'fp', 'tn']) {
    const m = document.createElementNS(NS, 'marker');
    m.setAttribute('id', `arrow-${cat}`);
    m.setAttribute('viewBox', '0 0 10 10');
    m.setAttribute('refX', '8'); m.setAttribute('refY', '5');
    m.setAttribute('markerWidth', '5.5'); m.setAttribute('markerHeight', '5.5');
    m.setAttribute('orient', 'auto-start-reverse');
    const tri = document.createElementNS(NS, 'path');
    tri.setAttribute('d', 'M 0 1 L 9 5 L 0 9 z');
    tri.setAttribute('fill', styles[cat].stroke);
    m.appendChild(tri);
    defs.appendChild(m);
  }
  svg.appendChild(defs);

  const gEdges = document.createElementNS(NS, 'g');
  const gNodes = document.createElementNS(NS, 'g');
  svg.append(gEdges, gNodes);

  for (const e of visible) {
    const [a, b] = e;
    const cat = edgeCat(e), st = styles[cat];
    const pa = nodePos(a, n), pb = nodePos(b, n);
    // quadratic curve: control point pulled toward the centre (more for far-apart
    // nodes) and offset to one side so a->b and b->a don't overlap
    const mx = (pa.x + pb.x) / 2, my = (pa.y + pb.y) / 2;
    const dx = pb.x - pa.x, dy = pb.y - pa.y, len = Math.hypot(dx, dy) || 1;
    const pull = 0.45 * (len / (2 * R));
    const side = { x: -dy / len, y: dx / len };
    const cx = mx + (CX - mx) * pull + side.x * 10;
    const cy = my + (CY - my) * pull + side.y * 10;
    const sx = pa.x + dx * ((nodeR + 2) / len), sy = pa.y + dy * ((nodeR + 2) / len);
    const exq = pb.x - dx * ((nodeR + 6) / len), eyq = pb.y - dy * ((nodeR + 6) / len);
    const g = document.createElementNS(NS, 'g');
    g.setAttribute('class', 'edgeg');
    const path = document.createElementNS(NS, 'path');
    path.setAttribute('d', `M ${sx} ${sy} Q ${cx} ${cy} ${exq} ${eyq}`);
    path.setAttribute('class', 'edge');
    path.setAttribute('stroke', st.stroke);
    path.setAttribute('stroke-width', st.width);
    if (st.dash) path.setAttribute('stroke-dasharray', st.dash);
    path.setAttribute('marker-end', `url(#arrow-${cat})`);
    path.addEventListener('mousemove', ev => showEdgeTooltip(ev, e));
    path.addEventListener('mouseleave', hideTooltip);
    g.appendChild(path);
    if (st.wrong) {
      const qx = 0.25 * sx + 0.5 * cx + 0.25 * exq, qy = 0.25 * sy + 0.5 * cy + 0.25 * eyq;
      const s = 3.4;
      const x1 = document.createElementNS(NS, 'path');
      x1.setAttribute('d', `M ${qx - s} ${qy - s} L ${qx + s} ${qy + s} M ${qx - s} ${qy + s} L ${qx + s} ${qy - s}`);
      x1.setAttribute('class', 'xmark');
      x1.setAttribute('stroke', st.stroke);
      g.appendChild(x1);
    }
    gEdges.appendChild(g);
    edgeEls.push({ el: g, a, b });
  }

  for (let i = 0; i < n; i++) {
    if (state.hideIsolated && !connected.has(i)) continue;
    if (state.showOnlyNecessary && !p.params[i].nece) continue;
    const par = p.params[i];
    const pos = nodePos(i, n);
    const g = document.createElementNS(NS, 'g');
    g.setAttribute('class', 'node');
    const c = document.createElementNS(NS, 'circle');
    c.setAttribute('cx', pos.x); c.setAttribute('cy', pos.y); c.setAttribute('r', nodeR);
    const lc = layerColor(par.layer);
    c.setAttribute('fill', `color-mix(in srgb, ${lc} 30%, ${css('--surface')})`);
    c.setAttribute('stroke', lc);
    g.appendChild(c);
    if (par.nece) {  // dashed ink ring: necessary for the answer
      const ring = document.createElementNS(NS, 'circle');
      ring.setAttribute('cx', pos.x); ring.setAttribute('cy', pos.y);
      ring.setAttribute('r', nodeR + 2.6);
      ring.setAttribute('class', 'necering');
      g.appendChild(ring);
    }
    // two-line radial label: owner on top, attribute below
    const deg = (pos.a * 180) / Math.PI;
    const flip = deg > 90 || deg < -90;
    const [owner, attr] = splitLabel(par.s);
    const t = document.createElementNS(NS, 'text');
    const lx = CX + RLAB * Math.cos(pos.a), ly = CY + RLAB * Math.sin(pos.a);
    t.setAttribute('x', lx); t.setAttribute('y', ly);
    t.setAttribute('dominant-baseline', 'middle');
    t.setAttribute('text-anchor', flip ? 'end' : 'start');
    t.setAttribute('transform', `rotate(${flip ? deg + 180 : deg} ${lx} ${ly})`);
    const ts1 = document.createElementNS(NS, 'tspan');
    ts1.setAttribute('x', lx); ts1.setAttribute('dy', attr ? '-0.5em' : '0');
    ts1.textContent = owner;
    t.appendChild(ts1);
    if (attr) {
      const ts2 = document.createElementNS(NS, 'tspan');
      ts2.setAttribute('x', lx); ts2.setAttribute('dy', '1.05em');
      ts2.textContent = attr;
      t.appendChild(ts2);
    }
    g.appendChild(t);
    g.addEventListener('mouseenter', ev => { state.hover = i; showNodeTooltip(ev, i); applyFocus(); });
    g.addEventListener('mouseleave', () => { state.hover = null; hideTooltip(); applyFocus(); });
    g.addEventListener('click', () => clickNode(i));
    gNodes.appendChild(g);
    nodeEls.push({ el: g, i });
  }
  applyFocus();
}

function splitLabel(s) {
  // node labels are 'owner→attr' or 'owner⇒attr'; keep the arrow with the second line
  const m = s.match(/^(.*?)([→⇒])(.*)$/);
  return m ? [m[1], m[2] + m[3]] : [s, ''];
}

/* click cycle: none -> A; A set -> B; A+B set -> new A; clicking A clears both */
function clickNode(i) {
  if (state.selA === i) { state.selA = state.selB = null; }
  else if (state.selA === null) { state.selA = i; }
  else if (state.selB === null && state.selA !== i) { state.selB = i; }
  else { state.selA = i; state.selB = null; }
  applyFocus();
}

function applyFocus() {
  const focus = state.selA !== null ? state.selA : state.hover;
  const pair = state.selA !== null && state.selB !== null;
  const neighbours = new Set();
  for (const { a, b } of edgeEls) {
    if (a === focus) neighbours.add(b);
    if (b === focus) neighbours.add(a);
  }
  for (const { el, a, b } of edgeEls) {
    const on = pair
      ? (a === state.selA && b === state.selB) || (a === state.selB && b === state.selA)
      : focus === null || a === focus || b === focus;
    el.classList.toggle('dim', !on);
  }
  for (const { el, i } of nodeEls) {
    el.classList.toggle('selA', state.selA === i);
    el.classList.toggle('selB', state.selB === i);
    const on = pair
      ? i === state.selA || i === state.selB
      : focus === null || i === focus || neighbours.has(i);
    el.classList.toggle('dim', !on);
  }
  renderProbeInput();
}

/* ---------- tooltips ---------- */
function showEdgeTooltip(ev, e) {
  const p = prob();
  tooltip.innerHTML =
    `<div><b>A:</b> ${esc(p.params[e[0]].f)}</div>` +
    `<div><b>B:</b> ${esc(p.params[e[1]].f)}</div>` +
    `<div class="row"><span class="muted">true dep(A,B)</span><span>${e[2]}</span></div>` +
    `<div class="row"><span class="muted">pretrained</span><span>${e[3]} (p₁=${e[4].toFixed(3)})</span></div>` +
    `<div class="row"><span class="muted">random</span><span>${e[5]} (p₁=${e[6].toFixed(3)})</span></div>` +
    `<div class="row"><span class="muted">shown model</span><span>${CAT_NAMES[edgeCat(e)]}</span></div>`;
  placeTooltip(ev);
}
function showNodeTooltip(ev, i) {
  const p = prob();
  const pr = p.params[i];
  tooltip.innerHTML = `<div><b>${esc(pr.f)}</b></div>` +
    `<div class="muted">${pr.kind === 0 ? 'instance parameter (direct count)' : 'abstract parameter (category total)'}` +
    ` · layer: ${esc(p.layers[pr.layer])}</div>` +
    `<div class="muted">${pr.nece ? 'necessary for the answer' : 'not necessary for the answer'}</div>`;
  placeTooltip(ev);
}
function placeTooltip(ev) {
  tooltip.style.display = 'block';
  const pad = 14;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  const r = tooltip.getBoundingClientRect();
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
  tooltip.style.left = x + 'px'; tooltip.style.top = y + 'px';
}
function hideTooltip() { tooltip.style.display = 'none'; }

/* ---------- text panel: description + A/B substitution + highlighting ---------- */
function renderProbeInput() {
  const p = prob();
  const nameA = state.selA !== null ? p.params[state.selA].f : null;
  const nameB = state.selB !== null ? p.params[state.selB].f : null;
  document.getElementById('chipA').textContent = nameA || 'A';
  document.getElementById('chipB').textContent = nameB || 'B';
  let html = p.desc.map(esc).join('. ') + '.';
  // highlight B first, then A, so overlapping phrases prefer the A colour
  for (const [name, cls] of [[nameB, 'hlB'], [nameA, 'hlA']]) {
    if (!name) continue;
    const bare = esc(name.replace(/^each /, ''));
    if (bare) html = html.split(bare).join(`<mark class="${cls}">${bare}</mark>`);
  }
  document.getElementById('desc').innerHTML = html;
}

function renderText() {
  const p = prob();
  document.getElementById('question').textContent = p.question;
  document.getElementById('solution').textContent = p.solution.join('. ') + '.';
  const nSkip = p.nPairsExpected - p.edges.length;
  document.getElementById('probStats').textContent =
    `difficulty: ${p.nOp} ops · seed ${p.seed} · ${p.params.length} variables · ` +
    `${p.edges.length} ordered pairs tested` +
    (nSkip > 0 ? ` (${nSkip} skipped: input too long)` : '') +
    ` · ${p.edges.filter(e => e[2] === 1).length} true dependencies`;
  renderProbeInput();
}

/* ---------- legend (tabular, matplotlib-style samples) ---------- */
function renderLegend() {
  const p = prob();
  const arrow = (w) =>
    `<svg width="${w}" height="12" viewBox="0 0 ${w} 12"><line x1="1" y1="6" x2="${w-9}" y2="6" stroke="currentColor" stroke-width="1.6"/><path d="M ${w-9} 2 L ${w-1} 6 L ${w-9} 10 z" fill="currentColor"/></svg>`;
  const solid = `<svg width="46" height="12"><line x1="1" y1="6" x2="45" y2="6" stroke="currentColor" stroke-width="2"/></svg>`;
  const dashed = `<svg width="46" height="12"><line x1="1" y1="6" x2="45" y2="6" stroke="currentColor" stroke-width="2" stroke-dasharray="5 4"/></svg>`;
  const dot = (color, ring) =>
    `<svg width="16" height="16"><circle cx="8" cy="8" r="5" fill="color-mix(in srgb, ${color} 30%, var(--surface))" stroke="${color}" stroke-width="1.4"/>${ring ? '<circle cx="8" cy="8" r="7.4" fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="2 2"/>' : ''}</svg>`;
  const layers = p.layers.map((name, i) =>
    `<span class="lrow">${dot(layerColor(i), false)} ${esc(name)}</span>`).join('');
  document.getElementById('legend').innerHTML = `<table>
    <tr>
      <th>true label<br>(line style)</th>
      <th>prediction<br>(colour)</th>
      <th>how to read</th>
      <th>variables<br>(nodes)</th>
    </tr>
    <tr>
      <td>
        <div class="lrow">${solid} dep(A, B) = 1</div>
        <div class="lrow">${dashed} dep(A, B) = 0</div>
      </td>
      <td>
        <div class="lrow">${lineSample('tp')} correct</div>
        <div class="lrow">${lineSample('fn')} wrong (marked ×)</div>
      </td>
      <td>
        <div class="lrow">${arrow(46)} A → B: “A depends on B”</div>
        <div class="lrow">label “X→Y”: count of Y per X (instance)</div>
        <div class="lrow">label “X⇒Y”: total of category Y (abstract)</div>
        <div class="lrow">${dot('var(--muted)', true)} necessary for the answer</div>
      </td>
      <td>${layers}</td>
    </tr>
  </table>`;
}

/* ---------- confusion matrices ---------- */
function counts(edges, model) {
  const c = { tp: 0, tn: 0, fp: 0, fn: 0 };
  for (const e of edges) {
    const y = e[2], pr = model === 'pre' ? e[3] : e[5];
    c[y === 1 ? (pr === 1 ? 'tp' : 'fn') : (pr === 1 ? 'fp' : 'tn')]++;
  }
  return c;
}
function mcc({ tp, tn, fp, fn }) {
  const d = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  return d === 0 ? 0 : (tp * tn - fp * fn) / d;
}
function renderMatrices() {
  const grid = document.getElementById('cmGrid');
  grid.innerHTML = '';
  for (const model of ['pre', 'rand']) {
    let c, label = model === 'pre' ? 'pretrained' : 'random control';
    if (state.scope === 'problem') {
      c = counts(prob().edges, model);
    } else {
      const o = DATA.overall[model];
      c = { tp: o.tp, tn: o.tn, fp: o.fp, fn: o.fn };
    }
    const nTot = c.tp + c.tn + c.fp + c.fn;
    const acc = nTot ? (c.tp + c.tn) / nTot : 0;
    const cell = (v, cls) =>
      `<td class="${cls}">${v.toLocaleString()}<span class="pct">${nTot ? (100 * v / nTot).toFixed(1) : 0}%</span></td>`;
    const div = document.createElement('div');
    div.className = 'cm';
    div.innerHTML =
      `<h3>${label}${state.model === model ? ' (shown in graph)' : ''}</h3>` +
      `<div class="metrics">acc ${acc.toFixed(3)} · MCC ${mcc(c).toFixed(3)} · n=${nTot.toLocaleString()}</div>` +
      `<table><tr><th></th><th>pred 1</th><th>pred 0</th></tr>` +
      `<tr><th>dep = 1</th>${cell(c.tp, 'correct')}${cell(c.fn, 'wrong')}</tr>` +
      `<tr><th>dep = 0</th>${cell(c.fp, 'wrong')}${cell(c.tn, 'correct')}</tr></table>`;
    grid.appendChild(div);
  }
}

/* ---------- dependency matrix view ---------- */
function renderMatrixView() {
  const p = prob();
  const n = p.params.length;
  const cat = Array.from({ length: n }, () => new Array(n).fill(null));
  for (const e of p.edges) cat[e[0]][e[1]] = edgeCat(e);
  let html = '<table class="mx"><tr><th></th>' +
    p.params.map(par => `<th class="coltop"><span>${esc(par.s)}</span></th>`).join('') + '</tr>';
  for (let a = 0; a < n; a++) {
    html += `<tr><th class="rowlab">${esc(p.params[a].s)}</th>`;
    for (let b = 0; b < n; b++) {
      const c = cat[a][b];
      const title = c === null
        ? (a === b ? 'diagonal (untested)' : 'untested pair')
        : `${CAT_NAMES[c]} — dep(${p.params[a].s}, ${p.params[b].s})`;
      html += `<td class="${c || 'na'}" title="${esc(title)}"></td>`;
    }
    html += '</tr>';
  }
  html += '</table>';
  document.getElementById('mxWrap').innerHTML = html;
  document.getElementById('mxLegend').innerHTML =
    ['tp', 'tn', 'fp', 'fn'].map(c =>
      `<span><span class="sw" style="background:${mxColor(c)}"></span>${CAT_NAMES[c]}</span>`
    ).join('') + `<span><span class="sw" style="background:var(--chip-bg)"></span>untested / diagonal</span>` +
    `<span class="hint">(model: ${state.model === 'pre' ? 'pretrained' : 'random control'})</span>`;
}
function mxColor(c) {
  const good = css('--good'), bad = css('--bad');
  return { tp: good, tn: `color-mix(in srgb, ${good} 22%, var(--surface))`,
           fn: bad, fp: `color-mix(in srgb, ${bad} 30%, var(--surface))` }[c];
}

function renderAll() {
  syncProbGrid();
  renderText();
  renderGraph();
  renderLegend();
  renderMatrices();
  renderMatrixView();
}
buildProbGrid();
renderAll();
</script>
</body>
</html>
"""
