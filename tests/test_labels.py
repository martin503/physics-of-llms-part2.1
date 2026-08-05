"""Tests for the probe's candidate-parameter universe (``src.probe.labels``).

iGSM's ``Problem.all_param`` is a Cartesian product over the *layer widths*: it contains
parameters naming nodes the problem text never mentions, which no probe can answer (and it
is why a 3-variable problem used to be probed with 6 nodes). ``named_params`` narrows it to
the text-grounded set, and getting that rule wrong is silent: rows still generate, labels
still look plausible, the probe just trains on unanswerable questions.

The rule itself is pure graph/list logic, so it is unit-tested against a minimal fake
``Problem`` exposing only the attributes it reads -- fast, and it pins each clause of the
rule separately. The end-to-end invariants (label arrays stay in sync, no positive label is
ever dropped) need real iGSM problems and are marked ``slow``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.probe.labels import RAND_PARAM, _named_nodes, named_params, regenerate_problem


class FakeProblem:
    """Stand-in exposing exactly the attributes ``named_params`` reads.

    ``all_param`` is built the way ``Problem.set_whole_template`` builds it, so the fakes
    exercise the same "full product over layer widths" input the real filter faces.
    """

    def __init__(self, d, widths, problem_order, edges, ques_idx, topological_order=()):
        self.d = d
        self.l = widths
        self.problem_order = list(problem_order)
        self.ques_idx = ques_idx
        self.topological_order = list(topological_order)
        self.template = nx.DiGraph()
        self.template.add_nodes_from(self.problem_order)
        self.template.add_edges_from(edges)
        self.all_param = [
            (0, i, j, k)
            for i in range(d - 1)
            for j in range(widths[i])
            for k in range(widths[i + 1])
        ]
        self.all_param += [
            (1, i, j, k) for i in range(d - 1) for j in range(widths[i]) for k in range(i + 1, d)
        ]


def moray_eel():
    """The reported case: 2 creatures x 2 organs, but only Moray Eel is ever mentioned.

    Text: "each Moray Eel's Elbow Joint equals 14", "each Moray Eel's Biceps equals 13
    times as much as each Moray Eel's Elbow Joint", asking for Moray Eel's Organs.
    Node (0, 0) is Crab -- isolated in the structure graph and named nowhere.
    """
    elbow, biceps, organs = (0, 0, 1, 0), (0, 0, 1, 1), (1, 0, 1, 1)
    return FakeProblem(
        d=2,
        widths=[2, 2],
        problem_order=[elbow, biceps, RAND_PARAM, organs],
        edges=[(RAND_PARAM, elbow), (elbow, biceps), (elbow, organs), (biceps, organs)],
        ques_idx=organs,
        topological_order=[elbow, biceps, organs],
    )


def test_named_params_drops_parameters_of_unmentioned_nodes():
    """The regression: Crab is isolated in the structure graph and named in no sentence."""
    p = moray_eel()
    assert len(p.all_param) == 6  # what iGSM offers: the full 2x2 product plus 2 abstracts
    assert named_params(p) == [(0, 0, 1, 0), (0, 0, 1, 1), (1, 0, 1, 1)]  # all Moray Eel
    assert not [param for param in named_params(p) if param[2] == 0]  # nothing owned by Crab


def test_named_params_keeps_askable_parent_category():
    """Moray Eel's Organs is kept: a category total of a node the text names."""
    assert (1, 0, 1, 1) in named_params(moray_eel())


def test_named_params_preserves_igsm_ordering():
    """Row indices (``param_a``/``param_b``) address this list, so its order must be stable."""
    p = moray_eel()
    kept = named_params(p)
    assert kept == [param for param in p.all_param if param in set(kept)]


def test_rand_sentinel_names_no_node():
    """``(-1, 0, 0, 0)`` is iGSM's literal-constant node, not a parameter of node (0, 0).

    Read naively it looks like layer 0, node 0 -- which is Crab here, so mishandling it
    would silently re-admit exactly the parameters this filter exists to drop.
    """
    p = moray_eel()
    assert RAND_PARAM in p.problem_order
    assert (0, 0) not in _named_nodes(p)


def test_named_params_generates_unstated_categories_of_named_nodes():
    """A named node's category totals are askable even when no sentence mentions them.

    Depth 3: the text states "each Zoo0's Enclosure1 ..." only, but "how many Animals does
    Zoo0 have?" and "how many Animals does Enclosure1 have?" are both fair questions.
    """
    stated = (0, 0, 0, 1)  # Zoo0 -> Enclosure1
    p = FakeProblem(
        d=3,
        widths=[2, 2, 2],
        problem_order=[RAND_PARAM, stated],
        edges=[(RAND_PARAM, stated)],
        ques_idx=stated,
        topological_order=[stated],
    )
    kept = named_params(p)
    assert (1, 0, 0, 1) in kept and (1, 0, 0, 2) in kept  # Zoo0's Enclosures / Animals
    assert (1, 1, 1, 2) in kept  # Enclosure1's Animals -- named as the child of `stated`
    assert (1, 1, 0, 2) not in kept  # Enclosure0 is never named
    assert (0, 0, 1, 0) not in kept  # an unstated instance relation stays out


def test_named_params_does_not_invent_instance_relations():
    """Only *stated* instance parameters are kept; an unstated pair of named nodes is not.

    "each Crab's Elbow Joint" is a relation the description never establishes, even when
    both names occur -- unlike a category total, it does not follow from the sentences.
    """
    a, b = (0, 0, 0, 0), (0, 0, 1, 1)  # Node0->Child0 and Node1->Child1 both stated
    p = FakeProblem(
        d=2,
        widths=[2, 2],
        problem_order=[RAND_PARAM, a, b],
        edges=[(RAND_PARAM, a), (a, b)],
        ques_idx=b,
        topological_order=[a, b],
    )
    kept = named_params(p)
    assert a in kept and b in kept
    assert (0, 0, 0, 1) not in kept and (0, 0, 1, 0) not in kept  # the unstated cross pairs


def test_named_params_keeps_the_question_parameter_and_only_that_one():
    """The seed-57 shape: the question targets a node the description never names.

    That parameter is the answer (``nece = 1``) and worth probing regardless, but it is the
    only thing the question designates -- its node's other parameters stay out.
    """
    stated, ques = (0, 0, 1, 0), (1, 0, 0, 1)  # ques owner (0, 0) appears in no sentence
    p = FakeProblem(
        d=2,
        widths=[2, 2],
        problem_order=[RAND_PARAM, stated, ques],
        edges=[(RAND_PARAM, stated)],
        ques_idx=ques,
        topological_order=[ques],
    )
    kept = named_params(p)
    assert ques in kept and stated in kept
    assert (0, 0, 0, 0) not in kept and (0, 0, 0, 1) not in kept  # that node's own parameters


def test_named_params_rejects_dropping_a_necessary_parameter():
    """Guard: a ``nece = 1`` parameter left out would silently delete a positive label."""
    stated, orphan = (0, 0, 1, 0), (1, 0, 0, 1)
    p = FakeProblem(
        d=2,
        widths=[2, 2],
        problem_order=[RAND_PARAM, stated],
        edges=[(RAND_PARAM, stated)],
        ques_idx=stated,
        topological_order=[stated, orphan],  # orphan is necessary but grounded by nothing
    )
    with pytest.raises(AssertionError, match='necessary for the answer'):
        named_params(p)


# --------------------------------------------------------------------------- #
# End-to-end against real iGSM problems
# --------------------------------------------------------------------------- #

MED_23 = dict(max_op=23, max_edge=20, perm_level=5, detail_level=0)


@pytest.mark.slow
def test_reported_problem_has_only_its_three_real_variables():
    """Regression for the published report: seed 1515006390 showed 6 nodes, the text has 3.

    Reproduces only under the dataset's own ``med_cfg`` -- the default ``max_op`` yields a
    different problem for this seed.
    """
    pp = regenerate_problem(1515006390, split='test', med_cfg=MED_23)
    names = [pp.problem.get_param(param) for param in pp.all_param]
    assert names == [
        "each Moray Eel's Elbow Joint",
        "each Moray Eel's Biceps",
        "each Moray Eel's Organs",
    ]
    assert not any('Crab' in name for name in names)
    # Organs = Elbow Joint + Biceps, and Biceps is defined from Elbow Joint
    assert pp.dep().tolist() == [[0, 0, 0], [1, 0, 0], [1, 1, 0]]


@pytest.mark.slow
@pytest.mark.parametrize('seed', [3, 17, 57, 128, 501, 1234])
def test_every_label_array_matches_the_narrowed_parameter_list(seed):
    """``all_param``, ``labels`` and ``dep()`` must be narrowed identically.

    They are indexed together by a row's ``param_a``/``param_b``; if one is subset
    differently from another, every row is mislabelled while everything still runs.
    """
    pp = regenerate_problem(seed, split='test')
    index = pp.param_index
    raw_labels = pp.problem.lora_label(list(pp.keys))
    raw_dep, _nece_idx, _unnece_idx = pp.problem.lora_label2('dep')

    assert index == sorted(set(index))  # a strictly increasing subset, no duplicates
    assert pp.all_param == [pp.problem.all_param[i] for i in index]
    assert np.array_equal(pp.labels, raw_labels[:, index, :])
    assert np.array_equal(pp.dep(), raw_dep[np.ix_(index, index)])


@pytest.mark.slow
@pytest.mark.parametrize('seed', [3, 17, 57, 128, 501, 1234])
def test_filtering_drops_no_positive_label(seed):
    """Filtering may only remove unanswerable *negatives*, never a real answer.

    A dropped positive would be invisible in aggregate metrics (the classes are heavily
    imbalanced) while quietly making the task easier than it is.
    """
    pp = regenerate_problem(seed, split='test')
    raw_dep, _nece_idx, _unnece_idx = pp.problem.lora_label2('dep')

    # nece = 1 is exactly "in the solution's topological order"; none may be filtered out
    assert pp.nece.sum() == len(pp.problem.topological_order)
    assert pp.dep().sum() == raw_dep.sum()  # every dependency survives
