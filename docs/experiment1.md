# Experiment 1: untrained-class identification

Experiment 1 compares CNC-Local, GEM, DN2, and GMMD-Central on the same
CIFAR-10 class-pure query groups. Ten independent training runs each contain
all ten leave-one-class-out conditions, for 100 trained ResNet-20 models.

## Train the classifiers

For every seed in `0..9` and every excluded class in `0..9`, run:

```bash
PYTHONPATH=src python experiments/train_resnet20.py \
  --data-root data/raw \
  --output-root checkpoints/experiment1 \
  --training-seed <seed> \
  --excluded-class <class-index>
```

The command uses 60 epochs of SGD, learning rate `0.1`, momentum `0.9`, weight
decay `1e-4`, and batch size `128`, with the rotation, translation, and
horizontal-flip augmentation recorded in `configs/experiment1.json`.

## Evaluate one independent run

```bash
PYTHONPATH=src python experiments/evaluate_experiment1.py \
  --data-root data/raw \
  --checkpoint-root checkpoints/experiment1 \
  --training-seed <seed> \
  --output outputs/experiment1/seed_<seed>.json \
  --device cuda
```

For each held-out-class model, the evaluator uses 500 training images per
retained class. At each `N` in `{2,4,8,10}`, it constructs disjoint 500-image
evaluation and validation subsets for every class and then forms
nonoverlapping same-class groups.

- CNC-Local uses one original and 26 deterministic transformed inputs per
  query image.
- GEM scores the mean of unnormalized features with a tied within-class
  covariance.
- DN2 scores the mean of unnormalized features against 100 reference-group
  means from each of the nine retained classes, using two neighbors.
- GMMD-Central averages normalized features without renormalizing the mean and
  applies the Gaussian mean law with covariance `Sigma/N` and `R=128`.

Each method receives its own 90th-percentile threshold from trained-class
validation groups. No withheld-class or evaluation data enters calibration.

Repeat the command for all ten training seeds, then run:

```bash
python experiments/aggregate_experiment1.py \
  --input-root outputs/experiment1 \
  --output results_reproduced/experiment1/aggregate.json
```

The aggregate first gives equal weight to the ten held-out classes within each
independent run and then reports the mean and sample standard deviation across
the ten runs.

## Query-time boundary

The paper's timing measurement uses one NVIDIA GeForce RTX 3090 Ti. It begins
when a preloaded CPU image group is transferred to the GPU and ends after the
final score is copied to the CPU. It excludes file access, image decoding,
model loading, reference construction, and metric computation. The first run's
ten models and four group sizes form 40 conditions; each condition has one
warm-up and four balanced-order measurements.
