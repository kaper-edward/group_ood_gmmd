"""Experiment-1 GEM and DN2 comparison scorers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _tensor(
    value: np.ndarray | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    if not value.flags.writeable:
        value = np.array(value, copy=True)
    return torch.as_tensor(value, device=device, dtype=dtype)


@dataclass(frozen=True)
class GEMFitReceipt:
    rows: int
    dimension: int
    classes: int
    covariance_rank: int
    smallest_retained_eigenvalue: float
    largest_eigenvalue: float


class GEMScorer:
    """Tied-covariance GEM scorer with OOD-oriented output."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        pinv_rtol: float = 1e-12,
        query_chunk: int = 8192,
    ) -> None:
        self.device = torch.device(device)
        self.pinv_rtol = float(pinv_rtol)
        self.query_chunk = int(query_chunk)
        self.means: torch.Tensor | None = None
        self.precision: torch.Tensor | None = None

    @torch.no_grad()
    def fit(
        self,
        features: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
    ) -> GEMFitReceipt:
        values = _tensor(features, self.device, torch.float64)
        targets = _tensor(labels, self.device, torch.long)
        if values.ndim != 2 or targets.ndim != 1 or len(values) != len(targets):
            raise ValueError("features must be [rows, dimension] and labels [rows]")
        classes = torch.unique(targets, sorted=True)
        means = torch.stack([values[targets == current].mean(dim=0) for current in classes])
        centered = torch.cat(
            [values[targets == current] - means[index] for index, current in enumerate(classes)],
            dim=0,
        )
        centered -= centered.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / centered.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        cutoff = max(float(eigenvalues[-1]) * self.pinv_rtol, 0.0)
        keep = eigenvalues > cutoff
        if not bool(keep.any()):
            raise RuntimeError("GEM covariance has no retained eigenvalue")
        inverse = torch.where(
            keep, eigenvalues.reciprocal(), torch.zeros_like(eigenvalues)
        )
        self.means = means.float()
        self.precision = ((eigenvectors * inverse.unsqueeze(0)) @ eigenvectors.T).float()
        return GEMFitReceipt(
            rows=int(values.shape[0]),
            dimension=int(values.shape[1]),
            classes=int(len(classes)),
            covariance_rank=int(keep.sum()),
            smallest_retained_eigenvalue=float(eigenvalues[keep][0]),
            largest_eigenvalue=float(eigenvalues[-1]),
        )

    @torch.no_grad()
    def score(self, features: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.means is None or self.precision is None:
            raise RuntimeError("fit() must be called before score()")
        values = _tensor(features, self.device, torch.float32)
        chunks: list[torch.Tensor] = []
        for start in range(0, len(values), self.query_chunk):
            query = values[start : start + self.query_chunk]
            difference = query[:, None, :] - self.means[None, :, :]
            mahalanobis = torch.einsum(
                "bcd,de,bce->bc", difference, self.precision, difference
            )
            chunks.append((-torch.logsumexp(-0.5 * mahalanobis, dim=1)).cpu())
        return torch.cat(chunks).numpy().astype(np.float32, copy=False)


class DN2Scorer:
    """Exact blockwise two-nearest-neighbor scorer for group means."""

    def __init__(
        self,
        *,
        neighbors: int = 2,
        device: str = "cuda",
        query_chunk: int = 4096,
        bank_chunk: int = 1024,
    ) -> None:
        self.neighbors = int(neighbors)
        self.device = torch.device(device)
        self.query_chunk = int(query_chunk)
        self.bank_chunk = int(bank_chunk)
        self.bank: torch.Tensor | None = None

    def fit(self, reference_group_means: np.ndarray | torch.Tensor) -> None:
        bank = _tensor(reference_group_means, self.device, torch.float32).contiguous()
        if bank.ndim != 2 or len(bank) < self.neighbors:
            raise ValueError("reference bank is too small")
        self.bank = bank

    @torch.no_grad()
    def score(self, group_means: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.bank is None:
            raise RuntimeError("fit() must be called before score()")
        queries = _tensor(group_means, self.device, torch.float32)
        output: list[torch.Tensor] = []
        for query_start in range(0, len(queries), self.query_chunk):
            query = queries[query_start : query_start + self.query_chunk]
            best = torch.full(
                (len(query), self.neighbors),
                float("inf"),
                dtype=torch.float32,
                device=self.device,
            )
            query_norm = query.square().sum(dim=1, keepdim=True)
            for bank_start in range(0, len(self.bank), self.bank_chunk):
                bank = self.bank[bank_start : bank_start + self.bank_chunk]
                distances = (
                    query_norm
                    + bank.square().sum(dim=1).unsqueeze(0)
                    - 2.0 * (query @ bank.T)
                )
                distances.clamp_(min=0.0)
                local = torch.topk(
                    distances,
                    k=min(self.neighbors, len(bank)),
                    largest=False,
                ).values
                best = torch.topk(
                    torch.cat([best, local], dim=1),
                    k=self.neighbors,
                    largest=False,
                ).values
            output.append(best.mean(dim=1).cpu())
        return torch.cat(output).numpy().astype(np.float32, copy=False)


def sampled_group_means(
    features: np.ndarray,
    *,
    group_size: int,
    groups: int,
    seed: int,
) -> np.ndarray:
    """Sample each reference group without replacement; groups may overlap."""
    features = np.asarray(features, dtype=np.float32)
    if len(features) < int(group_size):
        raise ValueError("reference pool is smaller than group_size")
    generator = np.random.default_rng(seed)
    output = np.empty((groups, features.shape[1]), dtype=np.float32)
    for row in range(groups):
        indices = generator.choice(len(features), size=group_size, replace=False)
        output[row] = features[indices].mean(axis=0, dtype=np.float32)
    return output
