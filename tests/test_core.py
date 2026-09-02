from __future__ import annotations

import numpy as np
import torch

from group_ood_gmmd.cnc_local import CNCLocal
from group_ood_gmmd.comparators import DN2Scorer, GEMScorer
from group_ood_gmmd.gmmd import (
    arithmetic_group_means,
    fit_gmmd_mean,
    fit_gmmd_reference,
    group_centroids,
    l2_rows,
    score_gmmd_central,
    score_gmmd_image,
    score_gmmd_mean,
)
from group_ood_gmmd.grouping import experiment1_split, experiment2_groups
from group_ood_gmmd.metrics import ood_metrics


def synthetic(classes: int = 3, rows_per_class: int = 60, dimension: int = 12):
    generator = np.random.default_rng(17)
    features = generator.normal(
        size=(classes * rows_per_class, dimension)
    ).astype(np.float32)
    logits = np.repeat(np.eye(classes, dtype=np.float32), rows_per_class, axis=0)
    logits += generator.normal(scale=0.01, size=logits.shape).astype(np.float32)
    queries = generator.normal(size=(37, dimension)).astype(np.float32)
    return features, logits, queries


def test_reference_and_basic_score_are_finite_and_deterministic():
    features, logits, queries = synthetic()
    state_a = fit_gmmd_reference(features, logits, rank=4, device="cpu")
    state_b = fit_gmmd_reference(features, logits, rank=4, device="cpu")
    first = score_gmmd_image(queries, state_a, nu=5, device="cpu", chunk_size=16)
    second = score_gmmd_image(queries, state_b, nu=5, device="cpu", chunk_size=64)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-6)


def test_central_uses_group_mean_without_renormalizing():
    features, logits, queries = synthetic()
    state = fit_gmmd_reference(features, logits, rank=4, device="cpu")
    normalized = l2_rows(queries[:24])
    groups = np.arange(24).reshape(6, 4)
    centroids = group_centroids(normalized, groups)
    assert np.any(np.linalg.norm(centroids, axis=1) < 0.99)
    scores = score_gmmd_central(centroids, state, 4, device="cpu")
    assert scores.shape == (6,)
    assert np.isfinite(scores).all()


def test_gmmd_mean_k_zero_is_the_basic_score():
    features, logits, queries = synthetic()
    classifier_weight = np.eye(3, 12, dtype=np.float32)
    mean_state = fit_gmmd_mean(
        features, logits, classifier_weight, rank=4, device="cpu"
    )
    assert mean_state.classifier_row_rank == 2
    assert mean_state.separated_directions == 0
    actual = score_gmmd_mean(queries, mean_state, nu=5, device="cpu")
    expected = score_gmmd_image(queries, mean_state.reference, nu=5, device="cpu")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_gmmd_mean_directional_branch_is_finite():
    features, logits, queries = synthetic(classes=4, rows_per_class=80, dimension=8)
    classifier_weight = np.eye(4, 8, dtype=np.float32)
    state = fit_gmmd_mean(
        features, logits, classifier_weight, rank=1, device="cpu"
    )
    assert state.classifier_row_rank == 3
    assert state.separated_directions == 2
    scores = score_gmmd_mean(queries, state, nu=5, device="cpu", chunk_size=11)
    assert scores.shape == (37,)
    assert np.isfinite(scores).all()


def test_comparison_scorers_and_group_mean():
    features, _, queries = synthetic()
    labels = np.repeat(np.arange(3), 60)
    gem = GEMScorer(device="cpu")
    gem.fit(features, labels)
    assert np.isfinite(gem.score(queries)).all()

    dn2 = DN2Scorer(device="cpu", neighbors=2, query_chunk=7, bank_chunk=9)
    dn2.fit(features[:30])
    assert np.isfinite(dn2.score(queries)).all()

    groups = np.arange(20).reshape(5, 4)
    means = arithmetic_group_means(np.arange(20, dtype=np.float64), groups)
    np.testing.assert_array_equal(means, np.asarray([1.5, 5.5, 9.5, 13.5, 17.5]))


def test_grouping_protocol_is_deterministic():
    first = experiment1_split(
        1000, excluded_class=2, class_index=4, group_size=8
    )
    second = experiment1_split(
        1000, excluded_class=2, class_index=4, group_size=8
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.intersect1d(first[0], first[1]).size

    paired_a = experiment2_groups(9000, 5640, 16, 7)
    paired_b = experiment2_groups(9000, 5640, 16, 7)
    np.testing.assert_array_equal(paired_a[0], paired_b[0])
    np.testing.assert_array_equal(paired_a[1], paired_b[1])


def test_cnc_fixed_view_count():
    scorer = object.__new__(CNCLocal)
    scorer.normalization_mean = torch.tensor(
        (0.4914, 0.4822, 0.4465), dtype=torch.float32
    ).view(3, 1, 1)
    scorer.normalization_std = torch.tensor(
        (0.2023, 0.1994, 0.2010), dtype=torch.float32
    ).view(3, 1, 1)
    image = torch.zeros(3, 32, 32)
    first = scorer.fixed_views(image)
    second = scorer.fixed_views(image)
    assert len(first) == 27
    assert all(torch.equal(left, right) for left, right in zip(first, second))


def test_metric_orientation():
    metrics = ood_metrics(
        np.asarray([0.0, 0.1, 0.2]),
        np.asarray([0.8, 0.9, 1.0]),
    )
    assert metrics["auroc"] == 100.0
