"""Deterministic group construction used by both paper experiments."""

from __future__ import annotations

import numpy as np


def partition_indices(values: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    count = len(values) // int(size)
    return values[: count * int(size)].reshape(count, int(size))


def experiment1_split(
    total: int,
    *,
    excluded_class: int,
    class_index: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint 500-image evaluation and validation group indices."""
    seed = excluded_class * 10_000 + class_index * 1_000 + group_size * 10
    permutation = np.random.RandomState(seed).permutation(total)
    return (
        partition_indices(permutation[:500], group_size),
        partition_indices(permutation[500:1000], group_size),
    )


def experiment2_groups(
    id_length: int,
    ood_length: int,
    group_size: int,
    trial: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use one seeded generator in the retained ID-then-OOD order."""
    generator = np.random.default_rng(trial * 100 + group_size)
    return (
        partition_indices(generator.permutation(id_length), group_size),
        partition_indices(generator.permutation(ood_length), group_size),
    )
