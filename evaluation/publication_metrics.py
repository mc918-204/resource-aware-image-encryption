"""Publication-facing traditional metrics with safe not-applicable handling."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.evaluation.security_metrics import (
    calculate_adjacent_correlations,
    calculate_entropy,
    safe_calculate_npcr_uaci,
)

ENCRYPTION_SPECIFIC_PUBLICATION_METRICS = (
    "entropy",
    "npcr",
    "uaci",
    "adjacent_correlation_horizontal",
    "adjacent_correlation_vertical",
    "adjacent_correlation_diagonal",
)


def _as_image_array(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    return np.asarray(image)


def _safe_psnr(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    try:
        if reference.shape != candidate.shape:
            return None
        if reference.ndim == 3:
            return float(peak_signal_noise_ratio(reference, candidate, data_range=255))
        return float(peak_signal_noise_ratio(reference, candidate, data_range=255))
    except Exception:
        return None


def _safe_ssim(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    try:
        if reference.shape != candidate.shape:
            return None
        if reference.ndim == 3:
            return float(structural_similarity(reference, candidate, channel_axis=2, data_range=255))
        return float(structural_similarity(reference, candidate, data_range=255))
    except Exception:
        return None


def compute_publication_metrics(
    *,
    plain_image: np.ndarray,
    protected_image: np.ndarray | None = None,
    protected_payload: np.ndarray | bytes | None = None,
    differential_pair: tuple[np.ndarray, np.ndarray] | None = None,
    is_encryption_baseline: bool = True,
) -> Dict[str, Any]:
    """
    Compute traditional security and distortion metrics without raising on single-image failures.

    `psnr` and `ssim` are computed as plain image vs protected image when a
    same-shape protected image is available. `npcr` and `uaci` are only
    computed when an explicit differential ciphertext pair is provided.
    """
    plain = np.asarray(plain_image)
    protected = _as_image_array(protected_image)
    payload_array: np.ndarray | None
    if protected_payload is None:
        payload_array = protected
    elif isinstance(protected_payload, bytes):
        payload_array = np.frombuffer(protected_payload, dtype=np.uint8)
    else:
        payload_array = np.asarray(protected_payload)

    metrics: Dict[str, Any] = {
        "entropy": None,
        "npcr": None,
        "uaci": None,
        "psnr": None,
        "ssim": None,
        "adjacent_correlation_horizontal": None,
        "adjacent_correlation_vertical": None,
        "adjacent_correlation_diagonal": None,
    }
    errors: Dict[str, str] = {}
    unsupported: list[str] = []

    try:
        if payload_array is not None:
            metrics["entropy"] = calculate_entropy(payload_array)
            correlations = calculate_adjacent_correlations(payload_array)
            metrics["adjacent_correlation_horizontal"] = correlations["horizontal"]
            metrics["adjacent_correlation_vertical"] = correlations["vertical"]
            metrics["adjacent_correlation_diagonal"] = correlations["diagonal"]
        else:
            unsupported.extend(
                [
                    "entropy",
                    "adjacent_correlation_horizontal",
                    "adjacent_correlation_vertical",
                    "adjacent_correlation_diagonal",
                ]
            )
    except Exception as exc:
        errors["entropy_or_correlation"] = str(exc)

    try:
        if protected is not None and protected.shape == plain.shape:
            metrics["psnr"] = _safe_psnr(plain, protected)
            metrics["ssim"] = _safe_ssim(plain, protected)
        else:
            unsupported.extend(["psnr", "ssim"])
    except Exception as exc:
        errors["psnr_ssim"] = str(exc)

    try:
        if differential_pair is not None:
            npcr, uaci = safe_calculate_npcr_uaci(differential_pair[0], differential_pair[1])
            metrics["npcr"] = npcr
            metrics["uaci"] = uaci
        else:
            unsupported.extend(["npcr", "uaci"])
    except Exception as exc:
        errors["npcr_uaci"] = str(exc)

    if not is_encryption_baseline:
        for metric_name in ENCRYPTION_SPECIFIC_PUBLICATION_METRICS:
            metrics[metric_name] = None
        unsupported.extend(ENCRYPTION_SPECIFIC_PUBLICATION_METRICS)

    metrics["metric_errors"] = errors
    metrics["unsupported_metrics"] = sorted(set(unsupported))
    metrics["metric_status"] = "metric_runtime_error" if errors else (
        "metric_not_applicable" if unsupported else "metric_success"
    )
    return metrics
