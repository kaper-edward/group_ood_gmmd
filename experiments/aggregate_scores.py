#!/usr/bin/env python3
"""Aggregate nine per-image OOD scores under the Experiment-2 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from group_ood_gmmd.grouping import experiment2_groups
from group_ood_gmmd.metrics import ood_metrics


DATASET_POOLS = {
    "cifar10": ("cifar100", "tin", "mnist", "svhn", "texture", "places365"),
    "cifar100": ("cifar10", "tin", "mnist", "svhn", "texture", "places365"),
    "imagenet200": ("ssb_hard", "ninco", "inaturalist", "texture", "openimage_o"),
}
METHODS = (
    "MSP",
    "MaxLogit",
    "Energy",
    "ReAct",
    "Mahalanobis",
    "RMDS",
    "ViM",
    "KNN-k50",
    "GMMD",
)
SEEDS = (0, 1, 2)
GROUP_SIZES = (1, 2, 4, 8, 16)
PARTITIONS = 30


def load_scores(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        id_scores = payload["test_id_scores"].astype(np.float64, copy=True)
        ood_scores = {
            key.removeprefix("ood_").removesuffix("_scores"): payload[key].astype(
                np.float64, copy=True
            )
            for key in payload.files
            if key.startswith("ood_") and key.endswith("_scores")
        }
    return id_scores, ood_scores


def align_id(
    dataset: str,
    scores: np.ndarray,
    indices: dict[str, np.ndarray],
) -> np.ndarray:
    if len(scores) == 9_000:
        return scores
    if dataset in {"cifar10", "cifar100"} and len(scores) == 10_000:
        return scores[indices[f"{dataset}_indices"]]
    raise ValueError(f"{dataset}: expected 9,000 or 10,000 ID scores, got {len(scores)}")


def one_partition(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    group_size: int,
    trial: int | None,
) -> dict[str, float]:
    if trial is None:
        return ood_metrics(id_scores, ood_scores)
    id_indices, ood_indices = experiment2_groups(
        len(id_scores), len(ood_scores), group_size, trial
    )
    return ood_metrics(
        id_scores[id_indices].mean(axis=1),
        ood_scores[ood_indices].mean(axis=1),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--official-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial-output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.official_index, allow_pickle=False) as payload:
        indices = {key: payload[key].copy() for key in payload.files}

    cells: dict[str, dict[str, dict[str, dict]]] = {}
    for dataset, pools in DATASET_POOLS.items():
        cells[dataset] = {}
        for seed in SEEDS:
            cells[dataset][f"s{seed}"] = {}
            for method in METHODS:
                path = args.score_root / f"scores_{dataset}_s{seed}_{method}.npz"
                id_scores, ood_scores = load_scores(path)
                id_scores = align_id(dataset, id_scores, indices)
                missing = sorted(set(pools) - set(ood_scores))
                if missing:
                    raise ValueError(f"{path}: missing OOD pools {missing}")
                trial_payload: dict[str, np.ndarray] = {}
                method_result: dict[str, dict] = {}
                for group_size in GROUP_SIZES:
                    trials: tuple[int | None, ...] | range = (
                        (None,) if group_size == 1 else range(PARTITIONS)
                    )
                    pool_result: dict[str, dict] = {}
                    for pool in pools:
                        rows = [
                            one_partition(
                                id_scores,
                                ood_scores[pool],
                                group_size,
                                trial,
                            )
                            for trial in trials
                        ]
                        for metric in ("auroc", "fpr95"):
                            trial_payload[f"{pool}_q{group_size}_{metric}"] = np.asarray(
                                [row[metric] for row in rows], dtype=np.float64
                            )
                        pool_result[pool] = {
                            "auroc_mean": float(np.mean([row["auroc"] for row in rows])),
                            "fpr95_mean": float(np.mean([row["fpr95"] for row in rows])),
                            "partitions": len(rows),
                        }
                    method_result[str(group_size)] = pool_result
                trial_path = args.trial_output / f"trials_{dataset}_s{seed}_{method}.npz"
                trial_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(trial_path, **trial_payload)
                cells[dataset][f"s{seed}"][method] = method_result

    public: dict[str, dict] = {}
    for dataset, pools in DATASET_POOLS.items():
        methods: dict[str, dict] = {}
        for method in METHODS:
            summary = {
                "overall_auroc": {},
                "overall_fpr95": {},
                "checkpoint_std": {},
            }
            for group_size in GROUP_SIZES:
                checkpoint_auroc = []
                checkpoint_fpr95 = []
                for seed in SEEDS:
                    row = cells[dataset][f"s{seed}"][method][str(group_size)]
                    checkpoint_auroc.append(
                        float(np.mean([row[pool]["auroc_mean"] for pool in pools]))
                    )
                    checkpoint_fpr95.append(
                        float(np.mean([row[pool]["fpr95_mean"] for pool in pools]))
                    )
                summary["overall_auroc"][str(group_size)] = float(
                    np.mean(checkpoint_auroc)
                )
                summary["overall_fpr95"][str(group_size)] = float(
                    np.mean(checkpoint_fpr95)
                )
                summary["checkpoint_std"][str(group_size)] = float(
                    np.std(checkpoint_auroc, ddof=0)
                )
            methods[method] = summary
        public[dataset] = {"methods": methods}

    output = {
        "schema": "gmmd-experiment2-reproduction-v1",
        "status": "PASS",
        "protocol": {
            "benchmarks": list(DATASET_POOLS),
            "checkpoints": list(SEEDS),
            "official_id_rows": 9_000,
            "group_sizes": list(GROUP_SIZES),
            "partitions": PARTITIONS,
            "partition_seed": "trial * 100 + N",
            "methods": list(METHODS),
        },
        "datasets": public,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
