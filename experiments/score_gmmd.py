#!/usr/bin/env python3
"""Fit GMMD-Mean and export OOD-oriented per-image reproduction inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from group_ood_gmmd.gmmd import fit_gmmd_mean, score_gmmd_mean


def load_features(path: Path, *, require_logits: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    if path.suffix == ".npy":
        if require_logits:
            raise ValueError("a training NPY requires --train-logits")
        return np.load(path, mmap_mode="r"), None
    with np.load(path, allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32, copy=True)
        logits = (
            payload["logits"].astype(np.float32, copy=True)
            if "logits" in payload
            else None
        )
    if require_logits and logits is None:
        raise ValueError(f"{path} does not contain logits")
    return features, logits


def parse_ood(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"--ood requires NAME=PATH, got {value!r}")
        output[name] = Path(path)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-logits", type=Path)
    parser.add_argument("--classifier-weight", type=Path, required=True)
    parser.add_argument("--test-id", type=Path, required=True)
    parser.add_argument("--ood", action="append", default=[], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--nu", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()

    train_features, train_logits = load_features(
        args.train, require_logits=args.train_logits is None
    )
    if args.train_logits is not None:
        train_logits = np.load(args.train_logits, mmap_mode="r")
    if train_logits is None:
        raise ValueError("training logits are required")
    classifier_weight = np.load(args.classifier_weight, allow_pickle=False)
    state = fit_gmmd_mean(
        train_features,
        train_logits,
        classifier_weight,
        rank=args.rank,
        device=args.device,
    )

    test_features, _ = load_features(args.test_id)
    output = {
        "test_id_scores": score_gmmd_mean(
            test_features,
            state,
            nu=args.nu,
            device=args.device,
            chunk_size=args.chunk_size,
        )
    }
    for name, path in parse_ood(args.ood).items():
        features, _ = load_features(path)
        output[f"ood_{name}_scores"] = score_gmmd_mean(
            features,
            state,
            nu=args.nu,
            device=args.device,
            chunk_size=args.chunk_size,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **output)
    temporary.replace(args.output)
    print(
        f"{args.output} row_rank={state.classifier_row_rank} "
        f"separated={state.separated_directions}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
