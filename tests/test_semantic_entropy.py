"""Unit tests for the semantic-entropy clustering and entropy math (no model)."""

import math

from semantic_entropy import cluster_by_equivalence, cluster_entropy


def test_cluster_all_distinct():
    ids = cluster_by_equivalence(4, lambda i, j: False)
    assert len(set(ids)) == 4


def test_cluster_all_same():
    ids = cluster_by_equivalence(4, lambda i, j: True)
    assert len(set(ids)) == 1


def test_cluster_transitive_merge():
    # 0~1 and 1~2 must merge 0,1,2 even though 0~2 is never tested directly.
    pairs = {(0, 1), (1, 2)}
    ids = cluster_by_equivalence(4, lambda i, j: (i, j) in pairs or (j, i) in pairs)
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] != ids[0]


def test_cluster_two_groups():
    group = {0: 0, 1: 0, 2: 1, 3: 1}
    ids = cluster_by_equivalence(4, lambda i, j: group[i] == group[j])
    assert ids[0] == ids[1]
    assert ids[2] == ids[3]
    assert ids[0] != ids[2]


def test_cluster_empty():
    assert cluster_by_equivalence(0, lambda i, j: True) == []


def test_entropy_uniform_is_log_k():
    assert math.isclose(cluster_entropy([0, 1, 2, 3]), math.log(4))
    assert math.isclose(cluster_entropy([0, 0, 1, 1]), math.log(2))


def test_entropy_single_cluster_is_zero():
    assert cluster_entropy([0, 0, 0, 0]) == 0.0


def test_entropy_empty_is_zero():
    assert cluster_entropy([]) == 0.0


def test_entropy_skewed_below_uniform():
    skewed = cluster_entropy([0, 0, 0, 1])
    uniform = cluster_entropy([0, 0, 1, 1])
    assert 0.0 < skewed < uniform
