#!/usr/bin/env python3
"""Evaluate the four Experiment-1 methods for one independent run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from group_ood_gmmd.cnc_local import CNCLocal
from group_ood_gmmd.comparators import DN2Scorer, GEMScorer, sampled_group_means
from group_ood_gmmd.gmmd import (
    fit_gmmd_reference,
    group_centroids,
    l2_rows,
    score_gmmd_central,
)
from group_ood_gmmd.grouping import experiment1_split
from group_ood_gmmd.metrics import ood_metrics, threshold_metrics
from group_ood_gmmd.resnet import resnet20_v1


CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
GROUP_SIZES = (2, 4, 8, 10)
METHODS = ("CNC-Local", "GEM", "DN2", "GMMD-Central")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def class_tensors(dataset: datasets.CIFAR10) -> dict[int, torch.Tensor]:
    grouped: dict[int, list[torch.Tensor]] = {index: [] for index in range(10)}
    for image, target in dataset:
        grouped[int(target)].append(image)
    return {index: torch.stack(grouped[index]) for index in range(10)}


def reference_split(
    images: dict[int, torch.Tensor],
    excluded: int,
) -> dict[int, torch.Tensor]:
    output: dict[int, torch.Tensor] = {}
    for class_index in range(10):
        if class_index == excluded:
            continue
        generator = np.random.RandomState(excluded * 10_000 + class_index * 1_000)
        indices = generator.permutation(len(images[class_index]))[:500]
        output[class_index] = images[class_index][torch.from_numpy(indices)]
    return output


@torch.no_grad()
def features(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach().flatten(1).cpu())

    handle = model.avgpool.register_forward_hook(hook)
    try:
        for start in range(0, len(images), batch_size):
            model(images[start : start + batch_size].to(device))
    finally:
        handle.remove()
    return torch.cat(captured).numpy().astype(np.float32, copy=False)


def load_model(path: Path, *, device: str) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = resnet20_v1(num_classes=9)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


def pseudo_logits(rows_by_class: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(rows_by_class)
    features = np.concatenate([rows_by_class[index] for index in ordered])
    logits = np.zeros((len(features), len(ordered)), dtype=np.float32)
    offset = 0
    for label, class_index in enumerate(ordered):
        count = len(rows_by_class[class_index])
        logits[offset : offset + count, label] = 1.0
        offset += count
    return features, logits


def dn2_bank(
    reference_features: dict[int, np.ndarray],
    *,
    training_seed: int,
    excluded: int,
    group_size: int,
) -> np.ndarray:
    rows = []
    for class_index in sorted(reference_features):
        rows.append(
            sampled_group_means(
                reference_features[class_index],
                group_size=group_size,
                groups=100,
                seed=(
                    20260821
                    + training_seed * 100_000
                    + excluded * 10_000
                    + group_size * 100
                    + class_index
                ),
            )
        )
    return np.concatenate(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    train_dataset = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=transform
    )
    train_images = class_tensors(train_dataset)
    test_images = class_tensors(test_dataset)
    result: dict[str, object] = {
        "schema": "gmmd-experiment1-run-v1",
        "training_seed": args.training_seed,
        "group_sizes": list(GROUP_SIZES),
        "methods": list(METHODS),
        "held_out_classes": {},
    }

    for excluded, class_name in enumerate(CLASSES):
        checkpoint = (
            args.checkpoint_root
            / f"seed_{args.training_seed}"
            / f"no_{class_name}"
            / "best.pth"
        )
        model = load_model(checkpoint, device=args.device)
        reference_images = reference_split(train_images, excluded)
        reference_features = {
            class_index: features(model, images, device=args.device)
            for class_index, images in reference_images.items()
        }

        cnc = CNCLocal(model, device=args.device, activation_ratio=0.5)
        cnc.fit(reference_images)

        retained_classes = sorted(reference_features)
        all_reference = np.concatenate(
            [reference_features[class_index] for class_index in retained_classes]
        )
        reference_labels = np.concatenate(
            [
                np.full(len(reference_features[class_index]), label, dtype=np.int64)
                for label, class_index in enumerate(retained_classes)
            ]
        )
        gem = GEMScorer(device=args.device, pinv_rtol=1e-12, query_chunk=8192)
        gem.fit(all_reference, reference_labels)
        gmmd_features, gmmd_logits = pseudo_logits(reference_features)
        gmmd = fit_gmmd_reference(
            gmmd_features, gmmd_logits, rank=128, device=args.device
        )

        test_features = {
            class_index: features(model, test_images[class_index], device=args.device)
            for class_index in range(10)
        }
        normalized_test = {
            class_index: l2_rows(values)
            for class_index, values in test_features.items()
        }
        per_size: dict[str, dict] = {}
        for group_size in GROUP_SIZES:
            dn2 = DN2Scorer(neighbors=2, device=args.device)
            dn2.fit(
                dn2_bank(
                    reference_features,
                    training_seed=args.training_seed,
                    excluded=excluded,
                    group_size=group_size,
                )
            )
            evaluation: dict[str, dict[int, np.ndarray]] = {
                method: {} for method in METHODS
            }
            validation: dict[str, dict[int, np.ndarray]] = {
                method: {} for method in METHODS
            }
            for class_index in range(10):
                evaluation_groups, validation_groups = experiment1_split(
                    len(test_images[class_index]),
                    excluded_class=excluded,
                    class_index=class_index,
                    group_size=group_size,
                )
                evaluation["CNC-Local"][class_index] = np.asarray(
                    [
                        cnc.score_group(test_images[class_index][indices])
                        for indices in evaluation_groups
                    ],
                    dtype=np.float64,
                )
                raw_means = test_features[class_index][evaluation_groups].mean(
                    axis=1, dtype=np.float32
                )
                evaluation["GEM"][class_index] = gem.score(raw_means)
                evaluation["DN2"][class_index] = dn2.score(raw_means)
                centroids = group_centroids(
                    normalized_test[class_index], evaluation_groups
                )
                evaluation["GMMD-Central"][class_index] = score_gmmd_central(
                    centroids,
                    gmmd,
                    group_size,
                    device=args.device,
                )

                if class_index == excluded:
                    continue
                validation["CNC-Local"][class_index] = np.asarray(
                    [
                        cnc.score_group(test_images[class_index][indices])
                        for indices in validation_groups
                    ],
                    dtype=np.float64,
                )
                raw_means = test_features[class_index][validation_groups].mean(
                    axis=1, dtype=np.float32
                )
                validation["GEM"][class_index] = gem.score(raw_means)
                validation["DN2"][class_index] = dn2.score(raw_means)
                centroids = group_centroids(
                    normalized_test[class_index], validation_groups
                )
                validation["GMMD-Central"][class_index] = score_gmmd_central(
                    centroids,
                    gmmd,
                    group_size,
                    device=args.device,
                )

            size_result: dict[str, dict] = {}
            for method in METHODS:
                id_scores = np.concatenate(
                    [evaluation[method][index] for index in retained_classes]
                )
                ood_scores = evaluation[method][excluded]
                validation_scores = np.concatenate(
                    [validation[method][index] for index in retained_classes]
                )
                size_result[method] = {
                    **ood_metrics(id_scores, ood_scores),
                    **threshold_metrics(
                        id_scores,
                        ood_scores,
                        validation_scores,
                        quantile=0.9,
                    ),
                }
            per_size[str(group_size)] = size_result
        result["held_out_classes"][class_name] = per_size

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
