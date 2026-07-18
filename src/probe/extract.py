"""Stage A: run the *frozen* model over regenerated problems, cache (hidden states, labels).

Why cache instead of probing on-the-fly:
  * the transformer is frozen - extract once, then sweep many probes / layers / seeds
    without ever re-running the (expensive) forward pass;
  * "frozen" becomes structural: the model is never in the probe's optimizer, and every
    forward runs under `torch.inference_mode()` so no graph is even built.

Output: a single `.npz` (or parquet) with aligned rows
  `X: (N, d_model) float32` hidden states, `y: (N,) int64` nece labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from tqdm import tqdm

from src.probe.labels import ProbeProblem, param_read_positions, regenerate_problem


def load_frozen_model(model_path: str, device: str = 'cuda', dtype: torch.dtype = torch.float32):
    """Load the GPT2-RoPE checkpoint in eval mode with hidden-state output enabled.

    Mirrors `src.eval.generate.IgsmGenerator.__init__` but flips on `output_hidden_states`
    and disables the KV-cache (we do single full-sequence forward passes, not generation).
    """
    from src.model.gpt2_rope import GPT2LMHeadModelWithRoPE, verify_rope_buffers

    model = GPT2LMHeadModelWithRoPE.from_pretrained(model_path)
    verify_rope_buffers(model)  # reject checkpoints that load with garbage inv_freq
    model.config.use_cache = False
    model.config.output_hidden_states = True
    model.eval().to(device).to(dtype)
    return model


@torch.inference_mode()
def hidden_states_for_problem(
    model, pp: ProbeProblem, layer: int, device: str
) -> Float[torch.Tensor, 'seq d_model']:
    """Return layer-`layer` residual-stream states for one problem, shape `(seq, d_model)`.

    `output.hidden_states` is a tuple of length `n_layer + 1`: index 0 is the embedding
    output, index L is the residual stream *after* transformer block L. So `layer=6` reads
    the stream after the 6th block.
    """
    ids = torch.tensor([pp.token_id], device=device, dtype=torch.long)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    return out.hidden_states[layer][0]  # drop batch dim -> (seq, d_model)


def build_dataset(
    model_path: str,
    *,
    layer: int,
    n_problems: int,
    seed_start: int = 0,
    split: str = 'test',
    device: str = 'cuda',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble `(X, y)` over `n_problems` problems for a single `layer`.

    Each parameter contributes one row: hidden state at its read position (chosen in
    `labels.param_read_positions`) paired with its `nece` label.
    
    Args:
        model_path (str): Path to the pre-trained model.
        layer (int): Residual-stream layer to read (1..n_layer).
            0 = after embedding, before first transformer block
        n_problems (int): Number of problems to regenerate and extract.
        seed_start (int): Starting seed for problem regeneration.
        split (str): 'test' = held-out (bins 16-22).
        device (str): 'cuda' or 'cpu'.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: X (hidden states), y (nece labels per
            problem parameter), groups (problem seed per row, for group-wise splitting)
    """
    model = load_frozen_model(model_path, device=device)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    groups: list[int] = []  # problem seed per row -- rows within one problem are not independent

    for seed in tqdm(range(seed_start, seed_start + n_problems), desc='extract', unit='prob'):
        pp: ProbeProblem = regenerate_problem(seed, split=split)
        h = hidden_states_for_problem(model, pp, layer=layer, device=device)  # (seq, d_model)
        positions = param_read_positions(pp)  # {param: token_index}

        for p_idx, param in enumerate(pp.all_param):
            # move to CPU before numpy: np.stack cannot take CUDA tensors
            xs.append(h[positions[param]].cpu().numpy().astype(np.float32))
            ys.append(int(pp.nece[p_idx]))
            groups.append(seed)

    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
    )


def save_dataset(
    out: Path | str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, layer: int
) -> None:
    """Persist `(X, y, groups)` to `out` as a compressed `.npz` (records the probed layer).

    `groups[i]` is the problem seed row `i` came from; `train_probe` must split by group,
    not by row, or identical/correlated vectors leak between train and val.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, groups=groups, layer=np.int64(layer))
