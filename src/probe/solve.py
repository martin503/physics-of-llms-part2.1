"""Let the model solve a probe dataset's problems, so the report can show its own answer.

A probe query hides the question and injects marker tokens (``[START] A [MID] B [END]``).
This module does the opposite: it feeds the plain eval prompt and reads what the model writes::

    prompt    = [50256] [222] <problem + question> [223]
    generated = <solution> [224] <answer> [50256]

Scoring reuses iGSM's ``true_correct`` on ``prompt + generated``, like ``src.eval.score``:
the answer, every intermediate calculation and every parameter dependency must be right, so a
right answer reached through a wrong step counts as wrong. ``answer_correct`` records the
weaker answer-only check next to it.

The solutions sit with the probe predictions the report draws them next to, and the model
that writes them is the one that probe run's ``config.json`` names::

    <run_dir>/test_<dataset>/generations.parquet   per problem: group, n_op, correct, solution, ...
    <run_dir>/test_<dataset>/generations.json      solve rate, model, when

Only a pretrained run can be solved; the random-init control writes nothing worth reading.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.data.igsm import ANS_BOS, EOS, REPO_ROOT

logger = logging.getLogger(__name__)

GENERATIONS_NAME = 'generations.parquet'
GENERATIONS_META_NAME = 'generations.json'


def dataset_problems(data_dir: Path | str) -> list[tuple[int, int]]:
    """`(seed, n_op)` per problem of the offline dataset, in dataset order.

    Same order the queries load in (sorted shards, insertion order within each), so taking
    the first N here selects the same problems `report_dep` embeds.
    """
    from src.probe.data import SHARD_GLOB

    seen: dict[int, int] = {}
    for shard in sorted(Path(data_dir).glob(SHARD_GLOB)):
        table = pq.read_table(str(shard), columns=['group', 'n_op'])
        for seed, n_op in zip(
            table.column('group').to_pylist(), table.column('n_op').to_pylist(), strict=True
        ):
            seen.setdefault(int(seed), int(n_op))
    return list(seen.items())


def _score(prompt_ids: list[int], generated_ids: list[int], problem: Any) -> tuple[bool, str]:
    """`(correct, note)` from iGSM's `true_correct` over the full token stream.

    A malformed generation makes iGSM's parser raise. That is an incorrect solution, not an
    error, so exceptions map to `False` with their type in `note`.
    """
    from tools.tools_test import true_correct  # type: ignore[import-not-found]

    try:
        correct, _my_print, _parser = true_correct(prompt_ids + generated_ids, problem=problem)
        return bool(correct), ''
    except Exception as exc:  # noqa: BLE001 -- true_correct raises on malformed output
        return False, f'{type(exc).__name__}: {exc}'


def _decode_generation(generated_ids: list[int]) -> tuple[str, str, bool]:
    """Split a generation at `[224]` into `(solution_text, answer_text, has_answer)`.

    Without a `[224]` the model never reached an answer, so everything it wrote is solution text.
    """
    from tools.tools import tokenizer  # type: ignore[import-not-found]

    if ANS_BOS not in generated_ids:
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), '', False
    cut = generated_ids.index(ANS_BOS)
    solution = tokenizer.decode(generated_ids[:cut], skip_special_tokens=True).strip()
    answer = tokenizer.decode(generated_ids[cut + 1 :], skip_special_tokens=True).strip()
    return solution, answer, True


def solve_run(
    run_dir: Path | str,
    data_dir: Path | str,
    *,
    n_problems: int | None = None,
    batch_size: int = 16,
    max_new_tokens: int = 1024,
    device: str = 'cuda',
    overwrite: bool = False,
) -> dict[str, Any]:
    """Solve `data_dir`'s problems with the model behind probe run `run_dir`; write both files.

    `n_problems` limits to the front of the dataset (None = all); pass the same number the
    report shows to solve exactly its problems. Existing solutions are kept unless `overwrite`.
    """
    from src.data.igsm import ensure_igsm_submodule
    from src.eval.generate import IgsmGenerator
    from src.probe.data import git_commit
    from src.probe.evaluate import test_output_dir
    from src.probe.labels import regenerate_problem

    run_dir, data_dir = Path(run_dir), Path(data_dir)
    params = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))['params']
    model_path = params.get('model_path')
    if params.get('random_model') or not model_path:
        raise ValueError(f'{run_dir} probes a random-init control; only a pretrained run solves')

    out_dir = test_output_dir(run_dir, data_dir)
    meta_path = out_dir / GENERATIONS_META_NAME
    if meta_path.exists() and not overwrite:
        raise ValueError(f'{meta_path} already holds solutions; pass --overwrite to replace')

    meta = json.loads((data_dir / 'metadata.json').read_text(encoding='utf-8'))
    split, med_cfg = meta.get('split', 'test'), meta.get('med_cfg')
    # An op-pinned problem differs from what the seed yields unpinned, so the request that
    # built the dataset has to be repeated here (same rule as report_dep).
    request_op = bool(meta.get('uniform_difficulty'))

    problems = dataset_problems(data_dir)[:n_problems]
    assert problems, f'no problems found in {data_dir}; run `gen-data` first'
    ensure_igsm_submodule()

    prompts: list[list[int]] = []
    golds: list[Any] = []
    for seed, n_op in tqdm(problems, desc='regenerate', unit='prob'):
        op = n_op if request_op else None
        pp = regenerate_problem(seed, split=split, med_cfg=med_cfg, op=op)
        if int(pp.problem.n_op) != n_op:
            raise ValueError(
                f'problem {seed} regenerates with n_op={pp.problem.n_op} but the dataset '
                f"recorded {n_op}. The solutions would not match the report's problems."
            )
        prompts.append([EOS, *pp.token_id[: pp.sol_bos_index + 1]])
        golds.append(pp.problem)

    generator = IgsmGenerator(model_path, device=device)
    # Length-sorted batches: the per-batch generation cap is `n_positions - longest prompt`,
    # so mixing a long prompt into a short batch would cut everyone else's budget.
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    generated: list[list[int]] = [[]] * len(prompts)
    for gen, i in zip(
        generator.batch_generate([prompts[i] for i in order], batch_size, max_new_tokens),
        order,
        strict=True,
    ):
        generated[i] = gen

    rows: dict[str, list[Any]] = {
        k: [] for k in ('group', 'n_op', 'correct', 'answer_correct', 'has_answer',
                        'solution', 'answer', 'gold_answer', 'n_new_tokens', 'note')
    }
    for (seed, n_op), prompt, gen, problem in tqdm(
        zip(problems, prompts, generated, golds, strict=True), total=len(problems), desc='score'
    ):
        solution, answer, has_answer = _decode_generation(gen)
        correct, note = _score(prompt, gen, problem)
        rows['group'].append(seed)
        rows['n_op'].append(n_op)
        rows['correct'].append(correct)
        rows['answer_correct'].append(answer == str(problem.ans))
        rows['has_answer'].append(has_answer)
        rows['solution'].append(solution)
        rows['answer'].append(answer)
        rows['gold_answer'].append(int(problem.ans))
        rows['n_new_tokens'].append(len(gen))
        rows['note'].append(note if has_answer else 'no [224]: generation cut off or malformed')

    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), str(out_dir / GENERATIONS_NAME))
    n_correct = sum(rows['correct'])
    metadata = {
        'kind': 'probe_solutions',
        'model_path': model_path,
        'run_dir': str(run_dir),
        'data_dir': str(data_dir),
        'n_problems': len(problems),
        'n_correct': n_correct,
        'solve_rate': n_correct / len(problems),
        'answer_solve_rate': sum(rows['answer_correct']) / len(problems),
        'max_new_tokens': max_new_tokens,
        'repo_commit': git_commit(REPO_ROOT),
        'command': ' '.join(sys.argv),
        'created': datetime.now(UTC).isoformat(timespec='seconds'),
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    logger.info(
        'solved %d/%d problems (%.3f; answer-only %.3f) with %s -> %s',
        n_correct, len(problems), metadata['solve_rate'], metadata['answer_solve_rate'],
        model_path, out_dir / GENERATIONS_NAME,
    )
    return metadata


def load_solutions(
    run_dir: Path | str, data_dir: Path | str
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Load a run's solutions for `data_dir` as `({seed: row}, metadata)`; `({}, {})` if none.

    Missing solutions are not an error. The report then omits the model-solution view.
    """
    from src.probe.evaluate import test_output_dir

    out_dir = test_output_dir(Path(run_dir), Path(data_dir))
    path = out_dir / GENERATIONS_NAME
    if not path.exists():
        return {}, {}
    table = pq.read_table(str(path)).to_pylist()
    meta_path = out_dir / GENERATIONS_META_NAME
    metadata = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    return {int(row['group']): row for row in table}, metadata
