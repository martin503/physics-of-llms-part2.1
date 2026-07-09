"""The paper's "solution parser" metric, wrapped for inspect_ai.

Reuses iGSM's ``true_correct`` verbatim: it checks the final answer (mod 23) AND every
intermediate calculation AND every parameter-dependency edge (paper "Result 2",
footnote 13). Faithfulness requires the gold ``Problem`` (pickled into each eval row) and
the FULL token stream ``prompt_ids + generated_ids`` -- ``true_correct`` splits on the
sentinel ids 222/223/224 (``skip_222=True``), so the leading ``50256`` is harmless, but
passing generated-ids alone (no sentinels) silently scores 0%.
"""

from __future__ import annotations

import base64
import pickle
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState

from src.data.igsm import ensure_igsm_submodule


@lru_cache(maxsize=1)
def _true_correct() -> Any:
    """Lazily import iGSM's parser (needs the submodule on ``sys.path``)."""
    ensure_igsm_submodule()
    from tools.tools_test import true_correct

    return true_correct


def score_true_correct(
    prompt_ids: Sequence[int], generated_ids: Sequence[int], problem_bytes: bytes
) -> tuple[bool, str]:
    """Score ``generated_ids`` against the gold problem with iGSM's ``true_correct``.

    Returns ``(correct, explanation)``. Any failure is treated as incorrect rather than
    raised: ``true_correct`` itself raises ``ValueError`` on its ``sol_op < n_op`` branch,
    and a malformed generation can trip the ``Parser`` (``NotImplementedError`` etc.) --
    both must map to "incorrect", matching how the paper scores a non-solution.
    """
    ensure_igsm_submodule()
    problem = pickle.loads(problem_bytes)
    full = list(prompt_ids) + list(generated_ids)
    try:
        correct, _my_print, _parser = _true_correct()(full, problem=problem)
        return bool(correct), 'ok'
    except Exception as exc:  # noqa: BLE001 -- true_correct raises on malformed output
        return False, f'{type(exc).__name__}: {exc}'


@scorer(metrics=[accuracy()])
def igsm_correctness() -> Scorer:
    """inspect_ai scorer: run ``true_correct`` on the sample's precomputed generation."""

    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata
        # ``problem`` is base64 in the sample metadata (inspect_ai's log JSON is utf-8).
        correct, explanation = score_true_correct(
            md['prompt_ids'], md['generated_ids'], base64.b64decode(md['problem'])
        )
        return Score(
            value=int(correct),
            answer=str(md.get('gold_answer')),
            explanation=explanation,
            metadata={
                'slice': md['slice'],
                'op': md.get('op'),
                'p_format': md.get('p_format'),
            },
        )

    return score
