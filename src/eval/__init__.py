"""iGSM-med Figure-3 evaluation: torch generation + inspect_ai scoring.

See ``PLAN.md`` for the full handoff. ``src.data.eval`` generates the parquet slices;
this package loads them, generates solutions with the GPT2-RoPE model, and scores with
iGSM's ``true_correct`` via inspect_ai.
"""

from src.eval.generate import IgsmGenerator
from src.eval.score import igsm_correctness, score_true_correct
from src.eval.task import build_eval_task, load_slice, load_slices

__all__ = [
    'IgsmGenerator',
    'build_eval_task',
    'igsm_correctness',
    'load_slice',
    'load_slices',
    'score_true_correct',
]
