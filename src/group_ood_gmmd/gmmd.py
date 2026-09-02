"""GMMD-Central and GMMD-Mean as specified in the accompanying paper.

The public implementation uses one component per predicted ID class, a
low-rank-plus-isotropic covariance, rank cap R=128, and nu=5 by default.
All returned scores are oriented so that larger values are more OOD-like.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.special import ive
import torch


DEFAULT_RANK = 128
DEFAULT_NU = 5.0
MIN_SIGMA_SQ = 1e-6
COVARIANCE_CHUNK = 32_768


@dataclass
class _ComponentSpectrum:
    class_index: int
    count: int
    mean: np.ndarray
    eigenvalues: np.ndarray
    basis: np.ndarray
    feature_variance: float


@dataclass
class GMMDState:
    """Tensor state for one-component-per-class GMMD scoring."""

    dimension: int
    num_classes: int
    class_labels: list[int]
    requested_rank: int
    means: torch.Tensor
    basis: torch.Tensor
    eigenvalues: torch.Tensor
    sigma_sq: torch.Tensor
    log_det: torch.Tensor
    woodbury_weights: torch.Tensor
    effective_ranks: list[int]
    fallback_components: int


@dataclass
class GMMDMeanState:
    """Reference and optional directional state used by GMMD-Mean."""

    reference: GMMDState
    classifier_row_rank: int
    separated_directions: int
    density_basis: np.ndarray | None
    direction_basis: np.ndarray | None
    whitening: np.ndarray | None
    whitened_class_means: np.ndarray | None
    density_state: GMMDState | None


def l2_rows(values: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized float32 features."""
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / (norms + np.float32(1e-10))


def _covariance_spectrum(centered: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float]:
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular.square() / max(int(centered.shape[0]) - 1, 1)
    return (
        eigenvalues.detach().cpu().numpy().astype(np.float32, copy=False),
        vh.transpose(0, 1).detach().cpu().numpy().astype(np.float32, copy=False),
        float(eigenvalues.sum().item()),
    )


def _fit_spectra(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    max_rank: int,
    device: str,
) -> tuple[list[_ComponentSpectrum], list[int]]:
    values = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    class_labels = np.unique(labels).tolist()
    expected = list(range(len(class_labels)))
    if class_labels != expected:
        raise ValueError(
            "predicted class labels must be contiguous from zero; "
            f"observed {class_labels}"
        )

    components: list[_ComponentSpectrum] = []
    for class_index in class_labels:
        selected = torch.from_numpy(values[labels == class_index]).to(device)
        count = int(selected.shape[0])
        if count == 0:
            raise ValueError(f"class {class_index} has no reference features")
        mean = selected.mean(dim=0)
        centered = selected - mean
        if count > 2:
            eigenvalues, basis, total_variance = _covariance_spectrum(centered)
            keep = min(basis.shape[1], max(int(max_rank), 1) + 1)
            basis = basis[:, :keep]
            eigenvalues = eigenvalues[:keep]
            retained = float(np.sum(eigenvalues, dtype=np.float64))
            eigenvalues = np.concatenate(
                [eigenvalues, np.asarray([total_variance, retained], dtype=np.float32)]
            )
        else:
            basis = np.zeros((values.shape[1], 0), dtype=np.float32)
            eigenvalues = np.asarray([0.0, 0.0], dtype=np.float32)
        variance = float(selected.var().item()) if count > 1 else 1.0
        components.append(
            _ComponentSpectrum(
                class_index=int(class_index),
                count=count,
                mean=mean.detach().cpu().numpy().astype(np.float32, copy=False),
                eigenvalues=eigenvalues,
                basis=basis,
                feature_variance=variance,
            )
        )
    return components, [int(value) for value in class_labels]


def _build_state(
    components: list[_ComponentSpectrum],
    class_labels: list[int],
    *,
    dimension: int,
    rank: int,
    device: str,
    min_sigma_sq: float = MIN_SIGMA_SQ,
) -> GMMDState:
    effective_ranks: list[int] = []
    selected_basis: list[np.ndarray] = []
    selected_values: list[np.ndarray] = []
    sigma_values: list[float] = []
    fallback_components = 0

    for component in components:
        stored_values = component.eigenvalues[:-2]
        total_variance = float(component.eigenvalues[-2])
        if component.count > int(rank) + 1 and len(stored_values) > 1:
            actual_rank = min(
                int(rank), len(stored_values) - 1, component.count - 1
            )
            actual_rank = max(actual_rank, 1)
            values = stored_values[:actual_rank]
            basis = component.basis[:, :actual_rank]
            residual = total_variance - float(np.sum(values, dtype=np.float64))
            sigma = max(
                residual / max(dimension - actual_rank, 1), float(min_sigma_sq)
            )
        else:
            actual_rank = 1
            values = np.asarray([1.0], dtype=np.float32)
            basis = np.zeros((dimension, 1), dtype=np.float32)
            sigma = max(float(component.feature_variance), float(min_sigma_sq))
            fallback_components += 1
        effective_ranks.append(actual_rank)
        selected_basis.append(basis)
        selected_values.append(values)
        sigma_values.append(sigma)

    max_effective_rank = max(effective_ranks)
    count = len(components)
    means = np.stack([component.mean for component in components])
    basis = np.zeros((count, dimension, max_effective_rank), dtype=np.float32)
    eigenvalues = np.zeros((count, max_effective_rank), dtype=np.float32)
    for index, (vectors, values) in enumerate(zip(selected_basis, selected_values)):
        current_rank = int(values.shape[0])
        basis[index, :, :current_rank] = vectors
        eigenvalues[index, :current_rank] = values

    means_t = torch.from_numpy(means).to(device)
    basis_t = torch.from_numpy(basis).to(device)
    values_t = torch.from_numpy(eigenvalues).to(device)
    sigma_t = torch.from_numpy(np.asarray(sigma_values, dtype=np.float32)).to(device)
    log_det = (
        dimension * torch.log(sigma_t)
        + torch.log1p(values_t / (sigma_t.unsqueeze(1) + 1e-10)).sum(dim=1)
    )
    woodbury = values_t / (values_t + sigma_t.unsqueeze(1) + 1e-10)

    return GMMDState(
        dimension=dimension,
        num_classes=len(class_labels),
        class_labels=class_labels,
        requested_rank=int(rank),
        means=means_t,
        basis=basis_t,
        eigenvalues=values_t,
        sigma_sq=sigma_t,
        log_det=log_det,
        woodbury_weights=woodbury,
        effective_ranks=effective_ranks,
        fallback_components=fallback_components,
    )


def _fit_pre_normalized(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    rank: int,
    device: str,
) -> GMMDState:
    values = np.asarray(values, dtype=np.float32)
    components, class_labels = _fit_spectra(
        values, labels, max_rank=rank, device=device
    )
    return _build_state(
        components,
        class_labels,
        dimension=int(values.shape[1]),
        rank=rank,
        device=device,
    )


def fit_gmmd_reference(
    features: np.ndarray,
    logits: np.ndarray,
    *,
    rank: int = DEFAULT_RANK,
    device: str = "cuda",
) -> GMMDState:
    """Fit the shared one-component-per-predicted-class GMMD reference."""
    features = l2_rows(features)
    logits = np.asarray(logits)
    if len(features) != len(logits) or logits.ndim != 2:
        raise ValueError("features and two-dimensional logits must have equal rows")
    predicted = np.argmax(logits, axis=1).astype(np.int64, copy=False)
    expected = np.arange(logits.shape[1], dtype=np.int64)
    if not np.array_equal(np.unique(predicted), expected):
        raise ValueError("every classifier output class must occur in the ID reference")
    return _fit_pre_normalized(features, predicted, rank=rank, device=device)


def _mahalanobis(query: torch.Tensor, state: GMMDState) -> torch.Tensor:
    difference = query.unsqueeze(1) - state.means.unsqueeze(0)
    euclidean = difference.square().sum(dim=2) / state.sigma_sq.unsqueeze(0)
    projection = torch.einsum("bcd,cdr->bcr", difference, state.basis)
    correction = (
        (projection.square() * state.woodbury_weights.unsqueeze(0)).sum(dim=2)
        / state.sigma_sq.unsqueeze(0)
    )
    return torch.clamp(euclidean - correction, min=0.0)


def _class_log_density(
    points: np.ndarray,
    state: GMMDState,
    *,
    law: str,
    nu: float,
    device: str,
) -> np.ndarray:
    query = torch.from_numpy(np.asarray(points, dtype=np.float32)).to(device)
    mahalanobis = _mahalanobis(query, state)
    base = -0.5 * state.log_det.unsqueeze(0)
    if law == "student_t":
        penalty = 0.5 * (float(nu) + state.dimension) * torch.log1p(
            mahalanobis / float(nu)
        )
    elif law == "gaussian":
        penalty = 0.5 * mahalanobis
    else:
        raise ValueError(f"unsupported observation law: {law}")
    return (base - penalty).cpu().numpy()


@torch.no_grad()
def _score_image_pre_normalized(
    normalized: np.ndarray,
    state: GMMDState,
    *,
    nu: float = DEFAULT_NU,
    device: str = "cuda",
    chunk_size: int = 4096,
) -> np.ndarray:
    normalized = np.asarray(normalized, dtype=np.float32)
    output = np.empty(len(normalized), dtype=np.float32)
    for start in range(0, len(normalized), int(chunk_size)):
        end = min(start + int(chunk_size), len(normalized))
        class_logs = _class_log_density(
            normalized[start:end], state, law="student_t", nu=nu, device=device
        )
        maximum = class_logs.max(axis=1)
        output[start:end] = -(
            maximum + np.log(np.exp(class_logs - maximum[:, None]).sum(axis=1))
        )
    return output


def score_gmmd_image(
    features: np.ndarray,
    state: GMMDState,
    *,
    nu: float = DEFAULT_NU,
    device: str = "cuda",
    chunk_size: int = 4096,
) -> np.ndarray:
    """Return the basic image-level score in Equation (5)."""
    return _score_image_pre_normalized(
        l2_rows(features),
        state,
        nu=nu,
        device=device,
        chunk_size=chunk_size,
    )


@torch.no_grad()
def score_gmmd_central(
    normalized_centroids: np.ndarray,
    state: GMMDState,
    group_size: int,
    *,
    device: str = "cuda",
    chunk_size: int = 4096,
) -> np.ndarray:
    """Apply Equation (4) to means of already-normalized image features.

    The group mean is deliberately not normalized again. The omitted
    ``d*log(N)/2`` term is common to all groups evaluated at the same N and
    therefore does not alter their ranking.
    """
    centroids = np.asarray(normalized_centroids, dtype=np.float32)
    if int(group_size) < 1:
        raise ValueError("group_size must be positive")
    output = np.empty(len(centroids), dtype=np.float32)
    for start in range(0, len(centroids), int(chunk_size)):
        end = min(start + int(chunk_size), len(centroids))
        query = torch.from_numpy(centroids[start:end]).to(device)
        mahalanobis = _mahalanobis(query, state)
        class_logs = (
            -0.5 * state.log_det.unsqueeze(0)
            - 0.5 * float(group_size) * mahalanobis
        )
        output[start:end] = (-torch.logsumexp(class_logs, dim=1)).cpu().numpy()
    return output


def group_centroids(
    normalized_features: np.ndarray,
    groups: Sequence[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """Average normalized features by row index without renormalizing."""
    normalized_features = np.asarray(normalized_features, dtype=np.float32)
    groups = np.asarray(groups, dtype=np.int64)
    if groups.ndim != 2:
        raise ValueError("groups must have shape [groups, group_size]")
    return normalized_features[groups].mean(axis=1, dtype=np.float32)


def _pooled_within_class_covariance(
    coordinates: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    rank = coordinates.shape[1]
    totals = np.zeros((num_classes, rank), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.float64)
    for start in range(0, len(coordinates), COVARIANCE_CHUNK):
        block = coordinates[start : start + COVARIANCE_CHUNK].astype(np.float64)
        block_labels = labels[start : start + COVARIANCE_CHUNK]
        np.add.at(totals, block_labels, block)
        np.add.at(counts, block_labels, 1.0)
    means = totals / np.maximum(counts, 1.0)[:, None]
    scatter = np.zeros((rank, rank), dtype=np.float64)
    for start in range(0, len(coordinates), COVARIANCE_CHUNK):
        block_labels = labels[start : start + COVARIANCE_CHUNK]
        block = coordinates[start : start + COVARIANCE_CHUNK].astype(np.float64)
        block -= means[block_labels]
        scatter += block.T @ block
    return means, scatter / max(len(coordinates) - num_classes, 1)


def _inverse_square_root(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    if float(values.min()) <= 0.0:
        raise RuntimeError("selected-direction covariance is singular")
    return (vectors * values ** -0.5) @ vectors.T


def _log_bessel_i(order: float, argument: np.ndarray) -> np.ndarray:
    values = np.atleast_1d(np.asarray(argument, dtype=np.float64))
    if order <= 0.5:
        scaled = ive(order, values)
        return np.log(np.maximum(scaled, np.finfo(np.float64).tiny)) + values

    ratio = np.maximum(values, 1e-300) / order
    root = np.sqrt(1.0 + ratio * ratio)
    eta = root + np.log(ratio / (1.0 + root))
    t = 1.0 / root
    u1 = (3.0 * t - 5.0 * t**3) / 24.0
    u2 = (81.0 * t**2 - 462.0 * t**4 + 385.0 * t**6) / 1152.0
    u3 = (
        30375.0 * t**3
        - 369603.0 * t**5
        + 765765.0 * t**7
        - 425425.0 * t**9
    ) / 414720.0
    series = 1.0 + u1 / order + u2 / order**2 + u3 / order**3
    output = (
        order * eta
        - 0.5 * math.log(2.0 * math.pi * order)
        - 0.25 * np.log1p(ratio * ratio)
        + np.log(series)
    )
    if order < 60.0:
        scaled = ive(order, values)
        good = scaled > 0.0
        output = np.where(
            good, np.log(np.where(good, scaled, 1.0)) + values, output
        )
    return output


def fit_gmmd_mean(
    features: np.ndarray,
    logits: np.ndarray,
    classifier_weight: np.ndarray,
    *,
    rank: int = DEFAULT_RANK,
    device: str = "cuda",
) -> GMMDMeanState:
    """Fit the GMMD-Mean state, including Equation (6) when required."""
    normalized = l2_rows(features)
    logits = np.asarray(logits)
    predicted = np.argmax(logits, axis=1).astype(np.int64, copy=False)
    reference = fit_gmmd_reference(features, logits, rank=rank, device=device)

    weight = np.asarray(classifier_weight, dtype=np.float64)
    if weight.shape != (logits.shape[1], normalized.shape[1]):
        raise ValueError(
            "classifier_weight must have shape "
            f"({logits.shape[1]}, {normalized.shape[1]})"
        )
    centered = weight - weight.mean(axis=0, keepdims=True)
    _, singular, right = np.linalg.svd(centered, full_matrices=True)
    threshold = float(singular[0]) * 1e-6
    row_rank = int((singular > threshold).sum())
    separated = max(0, row_rank - int(rank))
    if separated == 0:
        return GMMDMeanState(
            reference=reference,
            classifier_row_rank=row_rank,
            separated_directions=0,
            density_basis=None,
            direction_basis=None,
            whitening=None,
            whitened_class_means=None,
            density_state=None,
        )

    row_basis = right[:row_rank].T.astype(np.float32)
    complement = right[row_rank:].T.astype(np.float32)
    row_coordinates = normalized @ row_basis
    class_means, within = _pooled_within_class_covariance(
        row_coordinates, predicted, logits.shape[1]
    )
    order = np.argsort(np.diag(within))
    selected, remaining = order[:separated], order[separated:]
    whitening = _inverse_square_root(within[np.ix_(selected, selected)])
    direction_basis = row_basis[:, selected]
    density_basis = np.concatenate(
        [complement, row_basis[:, remaining]], axis=1
    ).astype(np.float32)
    density_coordinates = normalized @ density_basis
    density_state = _fit_pre_normalized(
        density_coordinates, predicted, rank=rank, device=device
    )
    whitened_class_means = (
        class_means[:, selected].astype(np.float64) @ whitening.T
    )
    return GMMDMeanState(
        reference=reference,
        classifier_row_rank=row_rank,
        separated_directions=separated,
        density_basis=density_basis,
        direction_basis=direction_basis,
        whitening=whitening,
        whitened_class_means=whitened_class_means,
        density_state=density_state,
    )


def score_gmmd_mean(
    features: np.ndarray,
    state: GMMDMeanState,
    *,
    nu: float = DEFAULT_NU,
    device: str = "cuda",
    chunk_size: int = 4096,
) -> np.ndarray:
    """Return the current per-image GMMD-Mean score from Equations (5)--(6)."""
    if float(nu) <= 0.0:
        raise ValueError("nu must be positive")
    normalized = l2_rows(features)
    if state.separated_directions == 0:
        return _score_image_pre_normalized(
            normalized,
            state.reference,
            nu=nu,
            device=device,
            chunk_size=chunk_size,
        )

    required = (
        state.density_basis,
        state.direction_basis,
        state.whitening,
        state.whitened_class_means,
        state.density_state,
    )
    if any(value is None for value in required):
        raise RuntimeError("incomplete directional GMMD-Mean state")

    output = np.empty(len(normalized), dtype=np.float32)
    k = state.separated_directions
    bessel_order = k / 2.0 - 1.0
    concentration = np.linalg.norm(state.whitened_class_means, axis=1)
    for start in range(0, len(normalized), int(chunk_size)):
        end = min(start + int(chunk_size), len(normalized))
        block = normalized[start:end]
        density = block @ state.density_basis
        density_logs = _class_log_density(
            density,
            state.density_state,
            law="student_t",
            nu=nu,
            device=device,
        ).astype(np.float64)
        mapped = (block @ state.direction_basis).astype(np.float64) @ state.whitening.T
        radius = np.linalg.norm(mapped, axis=1)
        argument = np.maximum(np.outer(radius, concentration), 1e-9)
        exponent = density_logs + mapped @ state.whitened_class_means.T
        exponent += (
            bessel_order * np.log(argument)
            - (k / 2.0) * math.log(2.0 * math.pi)
            - _log_bessel_i(bessel_order, argument.ravel()).reshape(argument.shape)
        )
        maximum = exponent.max(axis=1)
        output[start:end] = -(
            maximum + np.log(np.exp(exponent - maximum[:, None]).sum(axis=1))
        )
    return output


def arithmetic_group_means(
    image_scores: np.ndarray,
    groups: Sequence[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """Apply Equation (7) to fixed groups of image-score row indices."""
    scores = np.asarray(image_scores, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    if groups.ndim != 2:
        raise ValueError("groups must have shape [groups, group_size]")
    return scores[groups].mean(axis=1)
