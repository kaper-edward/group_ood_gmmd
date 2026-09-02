# Data and checkpoint preparation

This release does not redistribute datasets, checkpoints, feature caches, or
per-image raw scores. Obtain each third-party asset from its original source
and retain its license and split metadata.

## Experiment 1

Download CIFAR-10 through `torchvision`. For each training seed in `0..9`,
train ten ResNet-20 classifiers, withholding one class in turn. Each model is
trained on the other nine classes. The scripts select 500 training images from
each retained class for all four reference constructions.

The CIFAR-10 test split contains 1,000 images per class. At each group size,
the deterministic split function assigns 500 images to evaluation and 500 to
validation. The withheld class contributes only its evaluation subset.

## Experiment 2

Use OpenOOD v1.5 data lists and its three independently trained ResNet-18
checkpoints for each of CIFAR-10, CIFAR-100, and ImageNet-200. Do not retrain
these backbones. Extract the final pre-logit feature and class logits for the
ID training set, the official ID test set, and each OOD dataset.

For CIFAR-10 and CIFAR-100, the GMMD scoring entry point accepts NPZ files:

```text
train.npz:   features, logits, labels
test_id.npz: features, labels
ood_*.npz:   features, labels
```

For ImageNet-200, the same arrays may be supplied as NPY files to avoid an
additional in-memory copy. The centered linear-classifier weight matrix is
required to construct the 71 separated discriminative directions:

```text
train_features.npy
train_logits.npy
classifier_weight.npy   # shape [200, feature_dimension]
test_id_features.npy
ood_*.npy
```

All features and logits used by the retained run were stored as `float32`.
`data/input_checksums.json` records the logical paths, byte sizes, and SHA-256
digests of the external feature inputs used to produce the released numbers.
It contains no machine-specific path.

## Official 9,000-row ID evaluation set

OpenOOD v1.5 designates 9,000 CIFAR ID test rows for evaluation. Some feature
caches contain all 10,000 CIFAR test rows. In that case, apply the matching
array in `data/indices/official_id_indices.npz`; do not apply the index a second
time to a cache that already contains 9,000 rows. The aggregation script
enforces this rule.

## Comparison-method score interface

Run MSP, MaxLogit, Energy, ReAct, Mahalanobis, RMDS, ViM, and KNN (`k=50`)
through their OpenOOD definitions. Store one OOD-oriented score file per
benchmark and checkpoint:

```text
scores_<benchmark>_s<seed>_<method>.npz
```

Each file contains `test_id_scores` and one `ood_<dataset>_scores` array per
OOD dataset. Larger values must indicate more OOD-like inputs. These per-image
files are inputs to reproduction and are not part of this public release.
