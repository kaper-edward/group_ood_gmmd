# Reproducibility and artifact boundary

The release supports two levels of verification.

## Retained-artifact verification

```bash
python scripts/verify_artifacts.py
pytest
```

The verifier checks every retained file against `SHA256SUMS`, validates the
experiment protocols and method rosters, recomputes summary statistics from
the ten-run and 30-partition measurements, and cross-checks the values shown in
Tables I--VIII and Figures 1--4. The tests exercise the Gaussian mean law,
basic and directional GMMD-Mean branches, deterministic grouping, comparison
scorers, and metric orientation on synthetic inputs.

## Complete numerical reproduction

1. Acquire the datasets and OpenOOD v1.5 checkpoints listed in
   `docs/third_party.md`.
2. Prepare features and scores according to `docs/data.md`.
3. Execute all Experiment 1 cells and aggregate the ten independent runs.
4. Execute all Experiment 2 checkpoint cells and aggregate 30 paired
   partitions per group size.
5. Compare the rebuilt JSON and NPZ outputs with `results/` at the displayed
   precision.

The retained checksum inventory allows the external feature inputs to be
matched to those used for the paper even though those large or licensed files
are not redistributed.

## Included

- GMMD-Central and GMMD-Mean implementations at the paper settings;
- CNC-Local, GEM, and DN2 implementations used in Experiment 1;
- fixed protocol configurations and official CIFAR ID alignment indices;
- ten-independent-run Experiment 1 supporting measurements;
- Experiment 2 aggregates and all 30-partition AUROC/FPR95 measurements;
- supporting data and rendered assets for Tables I--VIII and Figures 1--4;
- deterministic tests, provenance checksums, and citation metadata.

## Excluded

- third-party datasets, checkpoints, and feature caches;
- per-image raw scores for GMMD or comparison methods;
- manuscript drafts and journal-submission files;
- exploratory experiments, reviews, conversations, and development history;
- machine-specific paths or host information.
