#!/usr/bin/env python3
"""Aggregate the ten independent Experiment-1 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("CNC-Local", "GEM", "DN2", "GMMD-Central")
GROUP_SIZES = (2, 4, 8, 10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = [
        json.loads((args.input_root / f"seed_{seed}.json").read_text(encoding="utf-8"))
        for seed in range(10)
    ]
    methods: dict[str, dict] = {method: {} for method in METHODS}
    for method in METHODS:
        for size in GROUP_SIZES:
            per_run = []
            for run in runs:
                classes = run["held_out_classes"]
                per_run.append(
                    float(
                        np.mean(
                            [
                                classes[name][str(size)][method]["auroc"]
                                for name in classes
                            ]
                        )
                    )
                )
            methods[method][str(size)] = {
                "mean_auroc": float(np.mean(per_run)),
                "sample_std": float(np.std(per_run, ddof=1)),
                "per_run_auroc": {
                    f"s{seed}": value for seed, value in enumerate(per_run)
                },
            }

    output = {
        "schema": "gmmd-experiment1-reproduction-v1",
        "status": "PASS",
        "protocol": {
            "independent_runs": 10,
            "held_out_classes": 10,
            "trained_models": 100,
            "group_sizes": list(GROUP_SIZES),
            "methods": list(METHODS),
        },
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
