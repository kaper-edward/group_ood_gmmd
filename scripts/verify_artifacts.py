#!/usr/bin/env python3
"""Verify retained measurements, paper assets, checksums, and release scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
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
PUBLIC_NAMES = {
    "MSP": "MSP-Mean",
    "MaxLogit": "MaxLogit-Mean",
    "Energy": "Energy-Mean",
    "ReAct": "ReAct-Mean",
    "Mahalanobis": "Mahalanobis-Mean",
    "RMDS": "RMDS-Mean",
    "ViM": "ViM-Mean",
    "KNN-k50": "KNN-Mean (k=50)",
    "GMMD": "GMMD-Mean",
}
TABLE_NAMES = {
    "MSP": "MSP",
    "MaxLogit": "MaxLogit",
    "Energy": "Energy",
    "ReAct": "ReAct",
    "Mahalanobis": "Mahalanobis",
    "RMDS": "RMDS",
    "ViM": "ViM",
    "KNN-k50": "KNN (k=50)",
    "GMMD": "GMMD",
}
GROUP_SIZES = (1, 2, 4, 8, 16)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def require_round(
    actual: float,
    expected: float,
    label: str,
    failures: list[str],
    tolerance: float = 0.001,
) -> None:
    """Match three-decimal paper values from four-decimal public measurements."""
    if abs(float(actual) - float(expected)) > tolerance + 1e-12:
        failures.append(
            f"{label}: {actual} differs from {expected} by more than {tolerance}"
        )


def verify_manifest(failures: list[str]) -> None:
    listed: set[str] = set()
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        listed.add(relative)
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"manifest file missing: {relative}")
        elif digest(path) != expected:
            failures.append(f"manifest checksum mismatch: {relative}")
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.name != "SHA256SUMS"
    }
    for relative in sorted(actual - listed):
        failures.append(f"file missing from manifest: {relative}")
    for relative in sorted(listed - actual):
        failures.append(f"manifest entry has no file: {relative}")


def verify_metadata(failures: list[str]) -> None:
    if not (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License\n"):
        failures.append("LICENSE is not the MIT License")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for value in ("license: MIT", "Taeyeon", "Heung-Seok", "group_ood_gmmd"):
        if value not in citation:
            failures.append(f"CITATION.cff missing {value!r}")
    exp1 = load("configs/experiment1.json")
    exp2 = load("configs/experiment2.json")
    if exp1["methods"] != ["CNC-Local", "GEM", "DN2", "GMMD-Central"]:
        failures.append("Experiment 1 method roster changed")
    if exp1["gmmd_central"]["rank"] != 128 or exp1["trained_models"] != 100:
        failures.append("Experiment 1 rank or model count changed")
    if exp2["gmmd_mean"]["rank"] != 128 or exp2["gmmd_mean"]["degrees_of_freedom"] != 5:
        failures.append("Experiment 2 GMMD settings changed")
    if exp2["official_id_rows"] != 9000 or exp2["partitions"] != 30:
        failures.append("Experiment 2 row count or partition count changed")
    inventory = load("data/input_checksums.json")
    rows = inventory.get("files", [])
    if len(rows) != 82:
        failures.append(f"expected 82 input checksum records, found {len(rows)}")
    if any(Path(row["path"]).is_absolute() for row in rows):
        failures.append("input checksum inventory contains an absolute path")
    if len({row["path"] for row in rows}) != len(rows):
        failures.append("input checksum inventory contains duplicate paths")
    index = ROOT / "data/indices/official_id_indices.npz"
    if digest(index) != "da901499e710d6e295e3088a3e6002810f24265cab9d74750ac210bbea908651":
        failures.append("official ID alignment hash changed")


def table_rows(table: dict) -> dict[str, dict]:
    return {row.get("label_contains", ""): row for row in table["rows"]}


def verify_experiment1(failures: list[str]) -> None:
    aggregate = load("results/experiment1/aggregate.json")
    if aggregate.get("status") != "PASS":
        failures.append("Experiment 1 aggregate status is not PASS")
    table2 = table_rows(load("results/paper/tables/table_II.json"))
    table4 = table_rows(load("results/paper/tables/table_IV.json"))
    sizes = (2, 4, 8, 10)
    for method in ("CNC-Local", "GEM", "DN2", "GMMD-Central"):
        for column, size in enumerate(sizes, start=1):
            row = aggregate["methods"][method][str(size)]
            require_round(
                row["mean_auroc"],
                table2[method]["cells"][str(column)][0],
                f"Table II {method} N={size} mean",
                failures,
            )
            require_round(
                row["sample_std_auroc"],
                table2[method]["cells"][str(column)][1],
                f"Table II {method} N={size} std",
                failures,
            )
            if method != "CNC-Local":
                require_round(
                    row["mean_sensitivity"],
                    table4[method]["cells"][str(column)][0],
                    f"Table IV {method} N={size} mean",
                    failures,
                )
                require_round(
                    row["sample_std_sensitivity"],
                    table4[method]["cells"][str(column)][1],
                    f"Table IV {method} N={size} std",
                    failures,
                )

    classwise = load("results/experiment1/classwise_n10.json")["classes"]
    table3 = load("results/paper/tables/table_III.json")["rows"]
    for row in table3[:-1]:
        name = row["label_contains"]
        source = classwise[name]
        values = row["cells"]
        require_round(source["cnc_native_mean"], values["1"][0], f"Table III {name} CNC", failures)
        require_round(source["gmmd_centroid_mean"], values["2"][0], f"Table III {name} GMMD", failures)
        require_round(source["mean_delta"], values["3"][0], f"Table III {name} delta", failures)
        require_round(source["seed_sd_of_delta"], values["3"][1], f"Table III {name} std", failures)

    figure = load("results/paper/figure_data/exp1_sweep.json")["series"]
    for method in ("CNC-Local", "GEM", "DN2", "GMMD-Central"):
        for index, size in enumerate(sizes):
            row = aggregate["methods"][method][str(size)]
            require_round(row["mean_auroc"], figure[f"{method}:mean"][index], f"Figure 2 {method} N={size}", failures)
            require_round(row["sample_std_auroc"], figure[f"{method}:seed_sd"][index], f"Figure 2 {method} std N={size}", failures)


def compute_experiment2(failures: list[str]) -> dict:
    directory = ROOT / "results/experiment2/partition_metrics"
    files = sorted(directory.glob("*.npz"))
    if len(files) != 81:
        failures.append(f"expected 81 partition-metric files, found {len(files)}")
    computed: dict[str, dict] = {}
    for dataset, pools in DATASETS.items():
        computed[dataset] = {}
        for method in METHODS:
            method_rows = {
                "overall_auroc": {},
                "overall_fpr95": {},
                "checkpoint_std": {},
            }
            for size in GROUP_SIZES:
                checkpoint_auroc = []
                checkpoint_fpr95 = []
                for seed in range(3):
                    path = directory / f"trials_{dataset}_s{seed}_{method}.npz"
                    if not path.is_file():
                        failures.append(f"missing partition metrics: {path.name}")
                        continue
                    with np.load(path, allow_pickle=False) as payload:
                        expected_keys = {
                            f"{pool}_q{current}_{metric}"
                            for pool in pools
                            for current in GROUP_SIZES
                            for metric in ("auroc", "fpr95")
                        }
                        if set(payload.files) != expected_keys:
                            failures.append(f"partition metric keys changed: {path.name}")
                        for key in payload.files:
                            current = int(key.split("_q", 1)[1].split("_", 1)[0])
                            expected_shape = (1,) if current == 1 else (30,)
                            if payload[key].shape != expected_shape:
                                failures.append(f"{path.name}:{key} shape {payload[key].shape}")
                            if not np.isfinite(payload[key]).all():
                                failures.append(f"{path.name}:{key} contains non-finite values")
                        checkpoint_auroc.append(
                            float(np.mean([payload[f"{pool}_q{size}_auroc"].mean() for pool in pools]))
                        )
                        checkpoint_fpr95.append(
                            float(np.mean([payload[f"{pool}_q{size}_fpr95"].mean() for pool in pools]))
                        )
                if len(checkpoint_auroc) == 3:
                    method_rows["overall_auroc"][str(size)] = float(np.mean(checkpoint_auroc))
                    method_rows["overall_fpr95"][str(size)] = float(np.mean(checkpoint_fpr95))
                    method_rows["checkpoint_std"][str(size)] = float(np.std(checkpoint_auroc, ddof=0))
            computed[dataset][method] = method_rows
    return computed


def verify_experiment2(computed: dict, failures: list[str]) -> None:
    aggregate = load("results/experiment2/aggregate.json")
    if aggregate.get("status") != "PASS":
        failures.append("Experiment 2 aggregate status is not PASS")
    for dataset in DATASETS:
        for method in METHODS:
            retained = aggregate["datasets"][dataset]["methods"][PUBLIC_NAMES[method]]
            actual = computed[dataset][method]
            for size in GROUP_SIZES:
                for metric in ("overall_auroc", "overall_fpr95"):
                    if abs(actual[metric][str(size)] - retained[metric][str(size)]) > 7e-4:
                        failures.append(f"Experiment 2 {dataset} {method} N={size} {metric} drift")
                if abs(actual["checkpoint_std"][str(size)] - retained["overall_checkpoint_std"][str(size)]) > 7e-4:
                    failures.append(f"Experiment 2 {dataset} {method} N={size} checkpoint std drift")

    table5 = table_rows(load("results/paper/tables/table_V.json"))
    table7 = table_rows(load("results/paper/tables/table_VII.json"))
    table8 = table_rows(load("results/paper/tables/table_VIII.json"))
    datasets = ("cifar10", "cifar100", "imagenet200")
    reported_sizes = (2, 4, 8, 16)
    for method in METHODS:
        table_name = TABLE_NAMES[method]
        public_name = PUBLIC_NAMES[method]
        for dataset_index, dataset in enumerate(datasets):
            require_round(
                computed[dataset][method]["overall_auroc"]["1"],
                table7[table_name]["cells"][str(dataset_index + 1)][0],
                f"Table VII {dataset} {method}",
                failures,
            )
            for size_index, size in enumerate(reported_sizes):
                column = dataset_index * 4 + size_index + 1
                require_round(
                    computed[dataset][method]["overall_auroc"][str(size)],
                    table5[table_name]["cells"][str(column)][0],
                    f"Table V {dataset} {method} N={size} mean",
                    failures,
                )
                require_round(
                    computed[dataset][method]["checkpoint_std"][str(size)],
                    table5[table_name]["cells"][str(column)][1],
                    f"Table V {dataset} {method} N={size} std",
                    failures,
                )
                require_round(
                    computed[dataset][method]["overall_fpr95"][str(size)],
                    table8[public_name]["cells"][str(dataset_index + 1)][size_index],
                    f"Table VIII {dataset} {method} N={size}",
                    failures,
                )

    table6 = table_rows(load("results/paper/tables/table_VI.json"))
    for column, method in enumerate(METHODS, start=1):
        require_round(computed["imagenet200"][method]["overall_auroc"]["8"], table6["AUROC"]["cells"][str(column)][0], f"Table VI {method} AUROC", failures)
        require_round(computed["imagenet200"][method]["overall_fpr95"]["8"], table6["FPR"]["cells"][str(column)][0], f"Table VI {method} FPR95", failures)

    figure3 = load("results/paper/figure_data/exp2_saturation.json")["series"]
    display_dataset = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "imagenet200": "ImageNet-200"}
    for dataset in DATASETS:
        for method in ("GMMD", "KNN-k50", "RMDS", "MSP", "Mahalanobis"):
            key = f"{display_dataset[dataset]}:{PUBLIC_NAMES[method]}"
            for index, size in enumerate((2, 4, 8, 16)):
                require_round(computed[dataset][method]["overall_auroc"][str(size)], figure3[key][index], f"Figure 3 {key} N={size}", failures)
                require_round(computed[dataset][method]["checkpoint_std"][str(size)], figure3[f"{key}:checkpoint_std"][index], f"Figure 3 {key} std N={size}", failures)

    figure4 = load("results/paper/figure_data/sqrtq_prediction.json")["series"]
    residuals = []
    directory = ROOT / "results/experiment2/partition_metrics"
    for dataset, pools in DATASETS.items():
        for method in METHODS:
            predictions = {size: [] for size in (2, 4, 8, 16)}
            for seed in range(3):
                with np.load(directory / f"trials_{dataset}_s{seed}_{method}.npz", allow_pickle=False) as payload:
                    for pool in pools:
                        baseline = float(payload[f"{pool}_q1_auroc"][0]) / 100.0
                        for size in predictions:
                            predictions[size].append(float(norm.cdf(np.sqrt(size) * norm.ppf(baseline)) * 100.0))
            key = f"{display_dataset[dataset]}:{PUBLIC_NAMES[method]}:residual"
            for index, size in enumerate((2, 4, 8, 16)):
                residual = computed[dataset][method]["overall_auroc"][str(size)] - float(np.mean(predictions[size]))
                residuals.append(residual)
                require_round(residual, figure4[key][index], f"Figure 4 {key} N={size}", failures)
    summary = figure4["agreement_summary_q_gt_1_reported"]
    require_round(float(np.median(np.abs(residuals))), summary["median_absolute_error_pp"], "Figure 4 median absolute residual", failures)
    if summary["cells"] != 108:
        failures.append("Figure 4 cell count changed")


def verify_assets(failures: list[str]) -> None:
    for number, stem in enumerate(("protocol", "exp1_sweep", "exp2_saturation", "sqrtq_prediction"), start=1):
        data = ROOT / "results/paper/figure_data" / f"{stem}.json"
        png = ROOT / "results/paper/figures" / f"{stem}.png"
        pdf = ROOT / "results/paper/figures" / f"{stem}.pdf"
        if load(str(data.relative_to(ROOT))).get("supporting_data_for") != f"Figure {number}":
            failures.append(f"Figure {number} supporting-data label changed")
        if not png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            failures.append(f"invalid PNG: {png.name}")
        if not pdf.read_bytes().startswith(b"%PDF"):
            failures.append(f"invalid PDF: {pdf.name}")
    for number in ("I", "II", "III", "IV", "V", "VI", "VII", "VIII"):
        if not (ROOT / "results/paper/tables" / f"table_{number}.md").is_file():
            failures.append(f"missing Table {number} Markdown")


def main() -> int:
    failures: list[str] = []
    verify_manifest(failures)
    verify_metadata(failures)
    verify_experiment1(failures)
    computed = compute_experiment2(failures)
    verify_experiment2(computed, failures)
    verify_assets(failures)
    scope = subprocess.run(
        ["python3", str(ROOT / "scripts/check_public_scope.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if scope.returncode != 0:
        failures.extend(scope.stdout.strip().splitlines())
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"ARTIFACT VERIFICATION: FAIL ({len(failures)})")
        return 1
    print("ARTIFACT VERIFICATION: PASS")
    print("experiments=2 methods=9 runs=10 partitions=30 tables=8 figures=4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
