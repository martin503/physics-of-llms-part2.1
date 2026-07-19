"""Build inspect_ai Tasks from the iGSM-med eval parquet slices.

Each parquet row (one problem) becomes a ``Sample`` whose metadata carries the fields the
scorer needs (``prompt_ids``, the precomputed ``generated_ids``, the pickled gold
``problem``) plus light aggregation keys (``slice``, ``op``, ``p_format``,
``gold_answer``). Generation happens in torch *before* the Task is built, so the solver is
a passthrough that only surfaces the decoded text for the eval log -- it never calls a
model (the inspect_ai ``eval()`` therefore needs no model).
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.solver import Solver, TaskState, solver

from src.eval.generate import IgsmGenerator
from src.eval.score import igsm_correctness


def load_slice(path: Path) -> list[dict[str, Any]]:
    """Read one ``med_*.parquet`` file into a list of row dicts."""
    return pq.read_table(path).to_pylist()


def load_slices(
    data_root: Path, slice_names: Sequence[str], limit: int | None = None
) -> list[dict[str, Any]]:
    """Load (optionally capped) rows from ``<data_root>/<name>.parquet`` for each slice."""
    rows: list[dict[str, Any]] = []
    for name in slice_names:
        path = data_root / f'{name}.parquet'
        if not path.exists():
            print(f'eval slice not found: {path}')
            continue
        slice_rows = load_slice(path)
        if limit is not None:
            slice_rows = slice_rows[:limit]
        rows.extend(slice_rows)
    return rows


def build_eval_task(generator: IgsmGenerator, rows: Sequence[dict[str, Any]]) -> Task:
    """Build an inspect_ai Task over ``rows`` (generation pre-attached) + correctness scorer."""
    samples = [_row_to_sample(r) for r in rows]
    return Task(
        dataset=MemoryDataset(samples),
        solver=[_attach_generation(generator)],
        scorer=[igsm_correctness()],
    )


def _attach_generation(generator: IgsmGenerator) -> Solver:
    """Passthrough solver: surface the precomputed generation text (no model call)."""

    @solver
    def attach() -> Solver:
        async def solve(state: TaskState, generate: Any) -> TaskState:  # noqa: ARG001
            gen = state.metadata['generated_ids']
            state.output = ModelOutput.from_content(
                model='igsm-local-rope', content=generator.decode(gen)
            )
            return state

        return solve

    return attach()


def _row_to_sample(row: dict[str, Any]) -> Sample:
    """Convert an eval parquet row to an inspect_ai Sample."""
    return Sample(
        input='igsm-med',  # vestigial: generation is precomputed, not text-driven
        target=str(row['gold_answer']),
        metadata={
            'slice': row['slice'],
            'p_format': row['p_format'],
            'op': int(row['op']),
            'gold_answer': int(row['gold_answer']),
            'prompt_ids': list(row['prompt_ids']),
            'generated_ids': list(row.get('generated_ids', [])),  # filled after torch gen
            # base64 (not raw bytes): inspect_ai's eval-log JSON is utf-8 and chokes on pickle.
            'problem': base64.b64encode(row['problem']).decode('ascii'),
        },
    )
