# Experiment 2: group-level OOD detection

Experiment 2 applies the same arithmetic-mean group operator to nine
image-level OOD scores on three OpenOOD v1.5 benchmarks.

## Compute GMMD-Mean image scores

For CIFAR-10 and CIFAR-100, supply the feature NPZ files and the checkpoint's
linear-classifier weight matrix:

```bash
PYTHONPATH=src python experiments/score_gmmd.py \
  --train data/feature_cache/cifar10/resnet18_s0/train.npz \
  --classifier-weight data/feature_cache/cifar10/resnet18_s0/fc_weight.npy \
  --test-id data/feature_cache/cifar10/resnet18_s0/test_id.npz \
  --ood cifar100=data/feature_cache/cifar10/resnet18_s0/nearood_cifar100.npz \
  --ood tin=data/feature_cache/cifar10/resnet18_s0/nearood_tin.npz \
  --ood mnist=data/feature_cache/cifar10/resnet18_s0/farood_mnist.npz \
  --ood svhn=data/feature_cache/cifar10/resnet18_s0/farood_svhn.npz \
  --ood texture=data/feature_cache/cifar10/resnet18_s0/farood_texture.npz \
  --ood places365=data/feature_cache/cifar10/resnet18_s0/farood_places365.npz \
  --output data/score_cache/scores_cifar10_s0_GMMD.npz \
  --rank 128 --nu 5 --device cuda
```

Repeat the command for all three checkpoints and benchmarks. CIFAR-10 and
CIFAR-100 use the basic image score because their centered classifier row
ranks, 9 and 99, do not exceed `R=128`. ImageNet-200 has centered row rank 199;
GMMD-Mean therefore separates `k=71` discriminative directions and applies the
directional term before producing each image score.

Generate the other eight score files from the corresponding OpenOOD
postprocessors using the same checkpoints and evaluation rows.

## Form groups and compute metrics

```bash
python experiments/aggregate_scores.py \
  --score-root data/score_cache \
  --official-index data/indices/official_id_indices.npz \
  --output results_reproduced/experiment2/aggregate.json \
  --trial-output results_reproduced/experiment2/partition_metrics
```

For each checkpoint and `N` in `{2,4,8,16}`, the script seeds one generator
with `trial * 100 + N`, permutes ID first and then the current OOD pool, divides
both into nonoverlapping groups, and averages the scores within each group. It
repeats this paired construction 30 times and discards incomplete final groups.

AUROC and FPR95 are computed separately for each OOD dataset. The script first
uses an equal-weight OOD-dataset mean within a checkpoint and then reports the
mean and descriptive population standard deviation (`ddof=0`) over the three independently trained
checkpoints. `N=1` is evaluated once and is used only as the baseline of the
square-root prediction analysis.

The public repository retains the resulting partition-level metrics, but not
the per-image score files.
