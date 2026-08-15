"""
Ground-truth probe labels from iGSM's own `Problem` graph.

The big win (see doc/seb_notes.md): we do **not** hand-write `nece(A)` / `value(A)` /
`can_next(A)`. iGSM's `Problem` already computes them:

* `Problem.lora_label(keys)` -> `np.ndarray` of shape
  `(1 + n_steps, n_param, len(keys))` with `keys` in
  `{"known", "can_next", "nece_next", "nece", "val"}`.
* `Problem.lora_label2(key)`  -> pairwise matrix for `{"dep", "neighbor", ...}`.

  (source: `iGSM/math_gen/problem_gen.py`)

This module regenerates a problem *keeping the live `Problem` object* (the parquet
shards only store `token_id`, which is not enough to recover labels) and packages the
token stream + the full label tensor + alignment metadata into one :class:`ProbeProblem`.

What it does **not** take from iGSM unchanged is the candidate-parameter set: iGSM's
`Problem.all_param` includes parameters of nodes the problem text never names, which no
probe can answer. :func:`named_params` narrows it to the text-grounded ones.

Token layout (from `iGSM/data_gen/prototype/id_gen.py` and `src/data/igsm.py`)::

    token_id = [222] prob_token [223] sol_token [224] ans_token [50256]
                ^prob_bos        ^sol_bos        ^ans_bos        ^eos

Step alignment (the subtle part, verified against iGSM source): with `be_shortest=True`
(the `IdGen` default), solution sentences are emitted in `Problem.topological_order`
(`problem_gen.py` line 290) -- the *same* order `lora_label` iterates for its step axis.
So label row `i_` describes the world state after the first `i_` solution sentences, and
pairs with the token position where the model has read exactly `i_` complete sentences
(see :func:`_solution_step_positions`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.igsm import IGSM_MED, ensure_igsm_submodule, get_bins

PROB_BOS, SOL_BOS, ANS_BOS, EOS = 222, 223, 224, 50256
DOT = 13  # GPT-2 BPE id for '.', the solution-sentence terminator

# Everything Problem.lora_label supports; extracting all is ~free, so cache all of them.
LABEL_KEYS: tuple[str, ...] = ('nece', 'known', 'can_next', 'nece_next', 'val')

# A parameter identifier (l, i, j, k) in iGSM's layered category hierarchy
# (layer i category name = `ln[i]`; node j in layer i = `N[i][j]`):
#   l = 0: INSTANCE param -- "how many N[i+1][k] does each N[i][j] have?"
#          (direct edge between node j in layer i and node k in layer i+1)
#   l = 1: ABSTRACT param -- "how many ln[k] (in total) does N[i][j] have?"
#          (aggregate over all layer-k descendants of node (i, j); k in i+1 .. d-1)
Param = tuple[int, int, int, int]

# iGSM's sentinel node for "a literal constant"; never a real parameter.
RAND_PARAM: Param = (-1, 0, 0, 0)


@dataclass
class ProbeProblem:
    """
    One regenerated iGSM problem with everything a probe needs.

    Attributes:
        token_id: Full token stream `[222] prob [223] sol [224] ans [50256]`.
        all_param: The candidate parameters (see :data:`Param` for `(l, i, j, k)`
            semantics); this is axis 1 of `labels`. It is :func:`named_params`, *not*
            iGSM's raw `problem.all_param`.
        param_index: Position of each `all_param` entry inside `problem.all_param`, so
            labels that iGSM computes over its own indexing (`dep`) can be subset.
        keys: Label-key names for axis 2 of `labels` (see :data:`LABEL_KEYS`).
        labels: `(1 + n_steps, n_param, n_keys)` int array from `Problem.lora_label`,
            already narrowed to `all_param` along axis 1. Axis 0 is the reasoning step:
            row `i_` = state after the first `i_` solution sentences. Step-independent
            keys (`nece`) are constant along it.
        sol_bos_index: Index of the `[223]` token in `token_id` (the model has ingested
            the full problem statement here; also `step_positions[0]`).
        step_positions: `(1 + n_steps,)` token indices aligned to `labels` axis 0:
            `step_positions[i_]` is where the model has read exactly `i_` complete
            solution sentences (row 0 -> `[223]`, row i_ -> the '.' ending sentence i_).
        problem: The live iGSM `Problem` object -- escape hatch for labels not cached
            here, e.g. `problem.lora_label2('dep')` (pairwise dep(A, B)) or the node
            names `problem.N` / `problem.ln` needed for per-mention read positions.
    """

    token_id: list[int]
    all_param: list[Param]
    param_index: list[int]
    keys: tuple[str, ...]
    labels: np.ndarray
    sol_bos_index: int
    step_positions: list[int]
    problem: Any

    def label(self, key: str, step: int = 0) -> np.ndarray:
        """Return `(n_param,)` labels for `key` at reasoning step `step`."""
        return self.labels[step, :, self.keys.index(key)]

    @property
    def nece(self) -> np.ndarray:
        """`(n_param,)` 0/1 array; 1 iff the parameter is necessary for the answer."""
        return self.label('nece', step=0)

    def dep(self) -> np.ndarray:
        """`(n_param, n_param)` matrix; entry (a, b) = 1 iff param a depends on param b.

        Computed on demand (transitive closure is not free), from `Problem.lora_label2`,
        then narrowed to `all_param`. `dep` is reachability in the dependency DAG, so
        restricting it to a subset of the nodes is exactly "the dependencies among these
        parameters" -- dropping a node cannot change whether b is reachable from a.
        """
        labels, _nece_idx, _unnece_idx = self.problem.lora_label2('dep')
        index = np.asarray(self.param_index)
        return labels[np.ix_(index, index)]


def _named_nodes(problem: Any) -> set[tuple[int, int]]:
    """Hierarchy nodes `(layer, index)` that the problem *description* names.

    `to_problem` emits one sentence per instance parameter in `problem_order`, naming that
    parameter's two nodes plus the nodes of its operands (its predecessors in
    `problem.template`). Nothing else in the text names a node: this iGSM configuration
    states parameter equations only, it never verbalises the structure graph.

    The question sentence is excluded here; :func:`named_params` adds its parameter back
    directly, rather than treating its node as named.
    """
    nodes: set[tuple[int, int]] = set()
    for param in problem.problem_order:
        if param[0] != 0:
            continue  # not its own sentence; picked up below as some sentence's operand
        for kind, i, j, k in (param, *problem.template.predecessors(param)):
            if kind == -1:
                continue  # the literal-constant sentinel names nothing
            nodes.add((i, j))
            if kind == 0:
                nodes.add((i + 1, k))  # instance params also name their child node
    return nodes


def named_params(problem: Any) -> list[Param]:
    """Candidate parameters the problem points at, in `problem.all_param` order.

    A parameter is a candidate when it is:

    * an instance parameter `problem_order` states;
    * an abstract parameter of a node the description names ("each Moray Eel's Organs",
      askable even when this problem does not ask it);
    * the question's parameter -- in ~1% of problems it targets a node no sentence mentions
      ("How many Classroom does Green Field Elementary have?", answer 0), probed anyway.

    iGSM's `all_param` instead enumerates the full product over layer widths, consulting
    neither the structure graph `G` nor the sentences emitted: it offers "each Crab's Elbow
    Joint" in a text about Moray Eels, and never fewer than `2*2 + 2*1 = 6` entries. An
    unstated instance parameter stays out -- a pair the description never relates is not a
    fact about the text -- while a node's category totals follow from its sentences.
    """
    nodes = _named_nodes(problem)
    keep = {param for param in problem.problem_order if param[0] == 0}
    keep |= {(1, i, j, k) for i, j in nodes if i < problem.d - 1 for k in range(i + 1, problem.d)}
    keep.add(problem.ques_idx)
    assert keep.issuperset(problem.topological_order), (
        'a parameter necessary for the answer is not a candidate: '
        f'{sorted(set(problem.topological_order) - keep)} -- nece labels would lose positives'
    )
    return [param for param in problem.all_param if param in keep]


def _new_idgen(med_cfg: dict[str, Any] | None = None):
    """Construct a fresh iGSM `IdGen` (one per problem: `gen_prob` mutates it).

    iGSM modules are import*able* only after `ensure_igsm_submodule()` has put the
    submodule root on `sys.path` -- hence the function-level import (cached by Python
    after the first call, so there is no per-problem cost).
    """
    ensure_igsm_submodule()
    from data_gen.pretrain.id_gen import IdGen  # type: ignore[import-not-found]

    return IdGen(**(med_cfg or IGSM_MED))


def _solution_step_positions(token_id: list[int], sol_bos_index: int, n_steps: int) -> list[int]:
    """Token index at which the model has read exactly `i_` solution sentences, per step.

    Solution sentences are '. '-joined (`id_gen.py`: `" " + ". ".join(solution) + "."`),
    and GPT-2 BPE keeps the terminating '.' as its own token (id 13), so sentence `i_`
    ends at the `i_`-th DOT token inside the solution region `(sol_bos, ans_bos)`.

    Returns `[sol_bos_index, end_of_sentence_1, ..., end_of_sentence_n]` -- length
    `1 + n_steps`, aligned to `lora_label`'s axis 0. Asserts the sentence count matches
    `n_steps` so any tokenizer-alignment drift fails loudly instead of silently
    mislabelling.
    """
    ans_bos_index = token_id.index(ANS_BOS)
    dots = [i for i in range(sol_bos_index + 1, ans_bos_index) if token_id[i] == DOT]
    assert len(dots) == n_steps, (
        f'expected {n_steps} solution sentences (len(topological_order)) but found '
        f'{len(dots)} DOT tokens in the solution region -- token alignment is broken'
    )
    return [sol_bos_index, *dots]


def regenerate_problem(
    seed: int,
    split: str = 'test',
    med_cfg: dict[str, Any] | None = None,
    keys: tuple[str, ...] = LABEL_KEYS,
    op: int | None = None,
) -> ProbeProblem:
    """Regenerate one iGSM problem of given `op` for `seed` and extract its probe labels.

    Mirrors `src.data.igsm._generate_chunk` (fix_seed -> fresh IdGen -> gen_prob) so the
    same `(seed, split, op)` reproduces the same problem deterministically. Probe on the
    `test` split by default (bins 16-22) -- these problems were held out of training.

    If `op` is not specified, iGSM may generate any problem difficulty using the given seed.

    Candidates are :func:`named_params`, and every label array is narrowed to them, so
    `all_param`, `labels`, `nece` and `dep()` share one indexing -- the one a row's
    `param_a`/`param_b` refer to.
    """
    ensure_igsm_submodule()
    from tools.tools import fix_seed  # type: ignore[import-not-found]

    fix_seed(seed)
    cfg = IGSM_MED if med_cfg is None else med_cfg
    if op is not None:
        # max_op must be >= op for the target to be reachable (see src.data.igsm._generate_chunk)
        cfg = {**cfg, 'op': op, 'max_op': max(cfg.get('max_op', 0), op)}
    gen = _new_idgen(cfg)
    gen.gen_prob(get_bins(split), p_format='pq')
    problem = gen.problem

    # lora_label needs the "whole template" (all candidate params, necessary or not) and the
    # topological solution order. gen()/to_problem() normally build these; guard just in case.
    if not hasattr(problem, 'whole_template'):
        problem.set_whole_template()

    labels = problem.lora_label(list(keys))  # (1 + n_steps, n_all_param, n_keys)

    keep = set(named_params(problem))
    param_index = [n for n, param in enumerate(problem.all_param) if param in keep]

    token_id = list(gen.token_id)
    sol_bos_index = token_id.index(SOL_BOS)
    step_positions = _solution_step_positions(
        token_id, sol_bos_index, n_steps=len(problem.topological_order)
    )

    return ProbeProblem(
        token_id=token_id,
        all_param=[problem.all_param[n] for n in param_index],
        param_index=param_index,
        keys=tuple(keys),
        labels=labels[:, param_index, :],
        sol_bos_index=sol_bos_index,
        step_positions=step_positions,
        problem=problem,
    )


def param_read_positions(pp: ProbeProblem) -> dict[Param, int]:
    """
    Map each parameter to the token position where its probe should be read.

    For `nece(A)`, every parameter is read at the sol_bos token (the `[223]` index in
    `pp.token_id`), where the model has ingested the full problem statement.

    Args:
        pp (ProbeProblem): regenerated problem with `pp.token_id` and `pp.all_param`

    Returns:
        dict[Param, int]: Mapping from each parameter to its token index.
    """
    return {param: pp.sol_bos_index for param in pp.all_param}


def problem_desc_end_index(pp: ProbeProblem) -> int:
    """Token index of the last problem-*description* token in `pp.token_id` (before the question).

    The paper probes `dep(A, B)` at the end of the problem *description*, before the question is
    asked (Figure 13b / Appendix B) -- unlike `nece(A)`, which is probed at the end of the question
    (`sol_bos_index`). iGSM emits the question as the final problem sentence, so the description is
    `problem.problem[:-1]`, joined and tokenized exactly as `id_gen` builds `token_id`
    (`" " + ". ".join(sentences) + "."`, then `[222] + prob_token + ...`).

    The returned index points at the terminating '.' of the description; `token_id[index + 1]` is
    the first question token. Asserts the reconstructed description tokens align with `token_id`, so
    any BPE-boundary drift fails loudly rather than silently probing at the wrong position (the same
    guarantee `_solution_step_positions` gives for the solution steps).
    """
    ensure_igsm_submodule()
    from tools.tools import tokenizer  # iGSM's GPT-2 tokenizer

    desc_text = ' ' + '. '.join(pp.problem.problem[:-1]) + '.'
    desc_tokens = tokenizer.encode(desc_text)
    end_index = len(desc_tokens)  # token_id[0] is PROB_BOS (222); description is token_id[1:end+1]
    assert pp.token_id[1 : end_index + 1] == desc_tokens, (
        'problem-description tokens do not align with token_id -- tokenizer/BPE-boundary drift; '
        'the dep probe position cannot be trusted'
    )
    return end_index
