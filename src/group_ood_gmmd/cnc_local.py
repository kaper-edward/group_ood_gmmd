"""CNC-Local implementation used in Experiment 1."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torchvision.transforms import functional as vision_functional


class CNCLocal:
    """Build and compare class neuron clusters from ResNet-20 activations."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str,
        activation_ratio: float = 0.5,
        normalization_mean: tuple[float, float, float] = (
            0.4914,
            0.4822,
            0.4465,
        ),
        normalization_std: tuple[float, float, float] = (
            0.2023,
            0.1994,
            0.2010,
        ),
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.activation_ratio = float(activation_ratio)
        self.normalization_mean = torch.tensor(
            normalization_mean, dtype=torch.float32
        ).view(3, 1, 1)
        self.normalization_std = torch.tensor(
            normalization_std, dtype=torch.float32
        ).view(3, 1, 1)
        self.reference_clusters: dict[int, torch.Tensor] = {}
        self._reference_stack: torch.Tensor | None = None

    def fixed_views(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Return the paper's deterministic 27-view set for one image."""
        image = (
            image * self.normalization_std + self.normalization_mean
        ).clamp(0, 1)
        views = [image.clone()]

        for angle in range(-20, 25, 5):
            if angle:
                views.append(vision_functional.rotate(image, angle, fill=0))

        height, width = image.shape[-2:]
        for scale in (1.05, 1.10, 1.15, 1.20):
            new_height, new_width = int(height * scale), int(width * scale)
            resized = vision_functional.resize(
                image, [new_height, new_width], antialias=True
            )
            top = (new_height - height) // 2
            left = (new_width - width) // 2
            views.append(resized[:, top : top + height, left : left + width])

        for scale in (0.95, 0.90, 0.85, 0.80):
            new_height, new_width = int(height * scale), int(width * scale)
            resized = vision_functional.resize(
                image, [new_height, new_width], antialias=True
            )
            top = (height - new_height) // 2
            left = (width - new_width) // 2
            bottom = height - new_height - top
            right = width - new_width - left
            views.append(
                functional.pad(
                    resized, [left, right, top, bottom], value=0
                )
            )

        for brightness in (0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4):
            views.append(torch.clamp(image * brightness, 0, 1))

        views.append(torch.roll(image, shifts=width // 32, dims=2))
        views.append(torch.roll(image, shifts=height // 32, dims=1))
        if len(views) != 27:
            raise RuntimeError(f"expected 27 views, got {len(views)}")
        return [
            (view - self.normalization_mean) / self.normalization_std
            for view in views
        ]

    @torch.no_grad()
    def activations(self, images: torch.Tensor) -> torch.Tensor:
        """Return the six last-stage activation maps for every image."""
        _, layers = self.model(images.to(self.device), return_activations=True)
        if layers is None or len(layers) != 6:
            raise RuntimeError("ResNet-20 must return six last-stage activations")
        return torch.stack(
            [layer.detach().permute(0, 2, 3, 1) for layer in layers], dim=1
        )

    def cluster(self, activations: torch.Tensor) -> torch.Tensor:
        """Mark neurons active in at least the configured fraction of views."""
        return (activations > 0).float().mean(dim=0) >= self.activation_ratio

    @torch.no_grad()
    def fit(self, class_images: Mapping[int, torch.Tensor], chunk_size: int = 256) -> None:
        """Build one reference cluster per trained class."""
        self.reference_clusters = {}
        for class_index in sorted(class_images):
            images = class_images[class_index]
            positive: torch.Tensor | None = None
            count = 0
            for start in range(0, len(images), chunk_size):
                current = self.activations(images[start : start + chunk_size])
                current_positive = (current > 0).sum(dim=0)
                positive = (
                    current_positive
                    if positive is None
                    else positive + current_positive
                )
                count += int(current.shape[0])
            if positive is None or count == 0:
                raise ValueError(f"class {class_index} has no reference images")
            self.reference_clusters[class_index] = (
                positive.float() / float(count) >= self.activation_ratio
            ).cpu()
        self._reference_stack = torch.stack(
            [
                self.reference_clusters[index]
                for index in sorted(self.reference_clusters)
            ]
        ).to(self.device)

    @torch.no_grad()
    def score_group(self, images: torch.Tensor) -> float:
        """Return an OOD-aligned score for one image group."""
        if self._reference_stack is None:
            raise RuntimeError("fit must be called before score_group")
        views = torch.cat(
            [
                torch.stack(self.fixed_views(image.cpu()), dim=0)
                for image in images
            ],
            dim=0,
        )
        query = self.cluster(self.activations(views)).to(self.device)
        expanded = query.unsqueeze(0)
        dimensions = tuple(range(1, self._reference_stack.ndim))
        intersection = (self._reference_stack & expanded).sum(dim=dimensions).float()
        union = (self._reference_stack | expanded).sum(dim=dimensions).float()
        similarities = torch.where(
            union > 0, intersection / union, torch.zeros_like(union)
        )
        return -float(similarities.max().item())
