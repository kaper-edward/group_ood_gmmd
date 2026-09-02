# GMMD: Reproduction Code and Artifacts

This repository accompanies the article:

> **GMMD: Gaussian-Mixture Mahalanobis Scoring for Untrained-Class
> Identification and Group-Level OOD Detection**

Authors: Taeyeon Kim and Heung-Seok Chae (corresponding author).

The release implements the shared class-specific GMMD reference model and its
two group scorers:

- **GMMD-Central** averages normalized features from a same-class group and
  applies the Gaussian mean law with covariance `Sigma/N`.
- **GMMD-Mean** computes the current image-level GMMD score and then takes the
  arithmetic mean of the image scores. When the centered classifier row rank
  exceeds `R`, it evaluates the excess discriminative directions separately.

The paper settings are `R=128`, `nu=5`, and one component per class.

## Release contents

| Path | Contents |
|---|---|
| `src/group_ood_gmmd/` | GMMD-Central, GMMD-Mean, comparison scorers, grouping, metrics, and ResNet-20 code |
| `experiments/` | Reproduction entry points for Experiments 1 and 2 |
| `configs/` | Paper experiment settings |
| `data/indices/` | OpenOOD 9,000-row ID alignment indices |
| `data/input_checksums.json` | Checksums for the external feature inputs used in Experiment 2 |
| `results/experiment1/` | Ten-independent-run supporting measurements |
| `results/experiment2/` | Nine-method aggregates and 30-partition AUROC/FPR95 measurements |
| `results/paper/` | Supporting data for Tables I--VIII and Figures 1--4 |
| `docs/` | Data preparation and full reproduction instructions |

The repository intentionally excludes datasets, checkpoints, feature caches,
per-image raw scores, manuscript drafts, submission files, and research-process
records.

## Quick verification

The following checks use only retained artifacts and synthetic test data; they
do not download a dataset or run a backbone.

```bash
conda env create -f environment.yml
conda activate group-ood-gmmd
python scripts/verify_artifacts.py
pytest
```

`verify_artifacts.py` validates the checksum manifest and cross-checks the
retained experiment measurements against every released paper table and figure
data file.

## Full reproduction

1. Prepare the licensed data and frozen checkpoints described in
   [`docs/data.md`](docs/data.md).
2. Run the 100-model CIFAR-10 study described in
   [`docs/experiment1.md`](docs/experiment1.md).
3. Run the three-checkpoint OpenOOD evaluation described in
   [`docs/experiment2.md`](docs/experiment2.md).
4. Follow [`docs/reproducibility.md`](docs/reproducibility.md) to rebuild the
   aggregates and compare them with the retained artifacts.

Experiment 1 reports class-pure groups at `N={2,4,8,10}` over ten independent
runs. Experiment 2 reports `N={2,4,8,16}` over three independently trained
OpenOOD checkpoints; 30 paired partitions per checkpoint quantify grouping
sensitivity. Its retained `N=1` values are inputs to the square-root diagnostic,
not additional group-performance repetitions.

## License and third-party assets

The repository is released under the MIT License, following the license model
used by OpenOOD. Third-party datasets, checkpoints, and software remain under
their own licenses; see [`docs/third_party.md`](docs/third_party.md).
