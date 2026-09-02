#!/usr/bin/env python3
"""Train one CIFAR-10 leave-one-class-out ResNet-20 classifier."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

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
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remap(labels: torch.Tensor, excluded: int) -> torch.Tensor:
    return labels - (labels > excluded).long()


def loaders(
    root: Path,
    excluded: int,
    batch_size: int,
    workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    train = datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform
    )
    test = datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )
    train_indices = [
        index for index, target in enumerate(train.targets) if target != excluded
    ]
    test_indices = [
        index for index, target in enumerate(test.targets) if target != excluded
    ]
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(Subset(train, train_indices), shuffle=True, **common),
        DataLoader(Subset(test, test_indices), shuffle=False, **common),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    excluded: int,
    device: str,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_correct = 0
    total = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = remap(labels.to(device, non_blocking=True), excluded)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * len(images)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += len(images)
    return total_loss / total, 100.0 * total_correct / total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument(
        "--excluded-class", type=int, choices=range(10), required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_all(args.training_seed)
    train_loader, test_loader = loaders(
        args.data_root, args.excluded_class, batch_size=128, workers=args.workers
    )

    model = resnet20_v1(num_classes=9).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
    )

    output = (
        args.output_root
        / f"seed_{args.training_seed}"
        / f"no_{CLASSES[args.excluded_class]}"
    )
    output.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    epoch_metrics: list[dict[str, float | int]] = []
    for epoch in range(1, 61):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            excluded=args.excluded_class,
            device=args.device,
            optimizer=optimizer,
        )
        test_loss, test_accuracy = run_epoch(
            model,
            test_loader,
            criterion,
            excluded=args.excluded_class,
            device=args.device,
            optimizer=None,
        )
        epoch_metrics.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "excluded_class": args.excluded_class,
                    "training_seed": args.training_seed,
                    "test_accuracy": test_accuracy,
                },
                output / "best.pth",
            )

    (output / "epoch_metrics.json").write_text(
        json.dumps(epoch_metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(output / "best.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

