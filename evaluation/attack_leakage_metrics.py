"""Attack/leakage prototype metrics for M3-C validation.

This module implements minimal attack/leakage metrics that do NOT depend on
large external detectors. These are PROTOTYPE metrics for validation only
and must NOT be used as final TIFS security conclusions.

Implemented metrics:
1. reconstruction_leakage_score - legacy SSIM/MAE distortion proxy (compatibility name)
2. reconstruction_similarity_proxy - explicit similarity direction for leakage screening
3. edge_leakage_score - gradient-based structural leakage estimation
4. sensitive_region_exposure_attack - high-risk region exposure ratio
5. known_budget_attacker_score - risk quality estimation given budget knowledge
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


def _compute_ssim_simple(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 11,
    C1: float = 6.5025,
    C2: float = 58.5225,
) -> float:
    """Compute a simplified SSIM score using numpy only."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
    
    if img1.size == 0:
        return 1.0
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim)


def _compute_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * math.log10(max_val**2 / mse)
    return float(psnr)


def _compute_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(img1.astype(np.float64) - img2.astype(np.float64))))


def _compute_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using simple finite differences (Sobel-like)."""
    if img.ndim == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img.astype(np.float64)
    
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2.0
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2.0
    
    return np.sqrt(gx**2 + gy**2)


def compute_reconstruction_leakage_score(
    plain_image: np.ndarray,
    protected_image: np.ndarray,
    protection_levels: Optional[List[str]] = None,
    block_size: int = 16,
) -> float:
    """Compute the legacy reconstruction distortion proxy.

    Measures how much information is recoverable from the protected image
    compared to the plain image in unprotected/weakly-protected regions.

    Formula: weighted combination of (1 - SSIM) and normalized MAE
    over unprotected regions only.

    This compatibility function measures visual difference, not leakage:
    identical images return 0 and more different images tend toward 1. The
    historical field name is retained because it is present in old result
    schemas. Use :func:`compute_reconstruction_similarity_proxy` when a
    similarity-direction screening value is required.

    Args:
        plain_image: Original image array.
        protected_image: Protected/encrypted image array.
        protection_levels: List of protection levels per block.
        block_size: Size of each block in pixels.
    
    Returns:
        Leakage score in [0, 1].
    """
    if plain_image.shape != protected_image.shape:
        return float('nan')
    
    if plain_image.size == 0:
        return 0.0
    
    weak_levels = {"none", "low"}
    
    if protection_levels is not None and len(protection_levels) > 0:
        h, w = plain_image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        blocks_per_row = w // block_size
        blocks_per_col = h // block_size
        
        for idx, level in enumerate(protection_levels):
            if level in weak_levels:
                bi = idx // blocks_per_row
                bj = idx % blocks_per_row
                y_start = bi * block_size
                x_start = bj * block_size
                y_end = min(y_start + block_size, h)
                x_end = min(x_start + block_size, w)
                mask[y_start:y_end, x_start:x_end] = True
        
        if not np.any(mask):
            return 0.0
        
        if plain_image.ndim == 3:
            mask_3d = np.stack([mask] * plain_image.shape[2], axis=-1)
        else:
            mask_3d = mask
        
        plain_weak = plain_image[mask_3d].astype(np.float64)
        protected_weak = protected_image[mask_3d].astype(np.float64)
        
        ssim_weak = _compute_ssim_simple(plain_weak.reshape(-1, 1), protected_weak.reshape(-1, 1))
        mae_weak = _compute_mae(plain_weak, protected_weak)
        mae_norm = min(mae_weak / 255.0, 1.0)
        
        leakage = 0.5 * (1 - ssim_weak) + 0.5 * mae_norm
    else:
        ssim_full = _compute_ssim_simple(plain_image, protected_image)
        mae_full = _compute_mae(plain_image, protected_image)
        mae_norm = min(mae_full / 255.0, 1.0)
        leakage = 0.5 * (1 - ssim_full) + 0.5 * mae_norm
    
    return float(np.clip(leakage, 0.0, 1.0))


def compute_reconstruction_similarity_proxy(
    plain_image: np.ndarray,
    protected_image: np.ndarray,
    protection_levels: Optional[List[str]] = None,
    block_size: int = 16,
) -> float:
    """Return an explicit similarity-direction reconstruction proxy.

    The value is ``1 -`` the legacy distortion proxy, so 1 means the selected
    region is visually identical to the plaintext and therefore maximally
    exposed under this narrow proxy. It is not a calibrated estimate of
    mutual information or an attack success probability.
    """
    distortion = compute_reconstruction_leakage_score(
        plain_image,
        protected_image,
        protection_levels=protection_levels,
        block_size=block_size,
    )
    if not np.isfinite(distortion):
        return float("nan")
    return float(np.clip(1.0 - distortion, 0.0, 1.0))


def compute_edge_leakage_score(
    plain_image: np.ndarray,
    protected_image: np.ndarray,
    protection_levels: Optional[List[str]] = None,
    block_size: int = 16,
) -> float:
    """Compute edge/gradient-based structural leakage score.

    Estimates how much structural/contour information leaks through
    by comparing gradient magnitudes between plain and protected images.

    Lower is better (less structural leakage).
    Range: [0, 1].

    Args:
        plain_image: Original image array.
        protected_image: Protected/encrypted image array.
        protection_levels: List of protection levels per block.
        block_size: Size of each block in pixels.
    
    Returns:
        Edge leakage score in [0, 1].
    """
    if plain_image.shape != protected_image.shape:
        return float('nan')
    
    if plain_image.size == 0:
        return 0.0
    
    grad_plain = _compute_gradient_magnitude(plain_image)
    grad_protected = _compute_gradient_magnitude(protected_image)
    
    weak_levels = {"none", "low"}
    
    if protection_levels is not None and len(protection_levels) > 0:
        h, w = plain_image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        blocks_per_row = w // block_size
        blocks_per_col = h // block_size
        
        for idx, level in enumerate(protection_levels):
            if level in weak_levels:
                bi = idx // blocks_per_row
                bj = idx % blocks_per_row
                y_start = bi * block_size
                x_start = bj * block_size
                y_end = min(y_start + block_size, h)
                x_end = min(x_start + block_size, w)
                mask[y_start:y_end, x_start:x_end] = True
        
        if not np.any(mask):
            return 0.0
        
        grad_plain_weak = grad_plain[mask]
        grad_protected_weak = grad_protected[mask]
    else:
        grad_plain_weak = grad_plain.flatten()
        grad_protected_weak = grad_protected.flatten()
    
    max_grad = max(np.max(grad_plain_weak), np.max(grad_protected_weak), 1e-10)
    
    diff = np.abs(grad_plain_weak.astype(np.float64) - grad_protected_weak.astype(np.float64))
    edge_leakage = float(np.mean(diff) / max_grad)
    
    return float(np.clip(edge_leakage, 0.0, 1.0))


def compute_sensitive_region_exposure_attack(
    risk_scores: Optional[np.ndarray],
    protection_levels: Optional[List[str]],
    topk: int = 5,
) -> Dict[str, Any]:
    """Compute sensitive region exposure attack metric.

    Based on existing risk scores and protection levels, this estimates
    the proportion of high-risk regions that remain visible or weakly protected.

    This is an attack-oriented version of semantic_exposure_rate.

    Args:
        risk_scores: Array of per-block risk/sensitivity scores.
        protection_levels: List of protection levels per block.
        topk: Number of top sensitive blocks to check.
    
    Returns:
        Dictionary with attack results.
    """
    if risk_scores is None or len(risk_scores) == 0:
        return {
            "attack_name": "sensitive_region_exposure_attack",
            "attacker_knowledge": "budget_aware",
            "detector_name": "risk_score_based",
            "detector_version": "m3c_prototype_v1",
            "confidence_before": None,
            "confidence_after": None,
            "detection_count_before": None,
            "detection_count_after": None,
            "attack_success_rate": None,
            "leakage_score": None,
            "status": "skipped",
            "error_message": "No risk scores available for sensitive region exposure attack.",
        }
    
    if protection_levels is None or len(protection_levels) == 0:
        return {
            "attack_name": "sensitive_region_exposure_attack",
            "attacker_knowledge": "budget_aware",
            "detector_name": "risk_score_based",
            "detector_version": "m3c_prototype_v1",
            "confidence_before": None,
            "confidence_after": None,
            "detection_count_before": None,
            "detection_count_after": None,
            "attack_success_rate": None,
            "leakage_score": None,
            "status": "skipped",
            "error_message": "No protection levels available for sensitive region exposure attack.",
        }
    
    num_blocks = len(risk_scores)
    weak_levels = {"none", "low"}
    
    topk_indices = np.argsort(risk_scores)[-topk:]
    topk_count = len(topk_indices)
    
    exposed_count = 0
    for idx in topk_indices:
        if idx < len(protection_levels) and protection_levels[idx] in weak_levels:
            exposed_count += 1
    
    exposure_rate = exposed_count / topk_count if topk_count > 0 else 0.0
    
    return {
        "attack_name": "sensitive_region_exposure_attack",
        "attacker_knowledge": "budget_aware",
        "detector_name": "risk_score_based",
        "detector_version": "m3c_prototype_v1",
        "confidence_before": float(np.mean(risk_scores[topk_indices])) if topk_count > 0 else None,
        "confidence_after": None,
        "detection_count_before": int(topk_count),
        "detection_count_after": int(exposed_count),
        "attack_success_rate": float(exposure_rate),
        "leakage_score": float(exposure_rate),
        "status": "success",
        "error_message": "",
    }


def compute_known_budget_attacker_score(
    risk_scores: Optional[np.ndarray],
    protection_levels: Optional[List[str]],
    budget_ratio: float,
) -> Dict[str, Any]:
    """Compute known-budget attacker score.

    Assumes the attacker knows the budget ratio and algorithm family,
    and estimates the risk quality of unprotected high-risk blocks.

    Lower is better (less exploitable risk).
    Range: [0, 1].

    Args:
        risk_scores: Array of per-block risk/sensitivity scores.
        protection_levels: List of protection levels per block.
        budget_ratio: The budget ratio used for selective protection.
    
    Returns:
        Dictionary with attack results.
    """
    if risk_scores is None or len(risk_scores) == 0:
        return {
            "attack_name": "known_budget_attacker_score",
            "attacker_knowledge": "budget_aware",
            "detector_name": "budget_aware_risk_estimator",
            "detector_version": "m3c_prototype_v1",
            "confidence_before": None,
            "confidence_after": None,
            "detection_count_before": None,
            "detection_count_after": None,
            "attack_success_rate": None,
            "leakage_score": None,
            "status": "skipped",
            "error_message": "No risk scores available for known-budget attacker.",
        }
    
    if protection_levels is None or len(protection_levels) == 0:
        return {
            "attack_name": "known_budget_attacker_score",
            "attacker_knowledge": "budget_aware",
            "detector_name": "budget_aware_risk_estimator",
            "detector_version": "m3c_prototype_v1",
            "confidence_before": None,
            "confidence_after": None,
            "detection_count_before": None,
            "detection_count_after": None,
            "attack_success_rate": None,
            "leakage_score": None,
            "status": "skipped",
            "error_message": "No protection levels available for known-budget attacker.",
        }
    
    weak_levels = {"none", "low"}
    num_blocks = len(risk_scores)
    
    weak_mask = np.array([protection_levels[i] in weak_levels for i in range(num_blocks)])
    
    if not np.any(weak_mask):
        return {
            "attack_name": "known_budget_attacker_score",
            "attacker_knowledge": "budget_aware",
            "detector_name": "budget_aware_risk_estimator",
            "detector_version": "m3c_prototype_v1",
            "confidence_before": None,
            "confidence_after": None,
            "detection_count_before": None,
            "detection_count_after": None,
            "attack_success_rate": 0.0,
            "leakage_score": 0.0,
            "status": "success",
            "error_message": "",
        }
    
    weak_risks = risk_scores[weak_mask]
    max_risk = float(np.max(risk_scores)) if np.max(risk_scores) > 0 else 1.0
    
    attacker_score = float(np.mean(weak_risks) / max_risk)
    
    return {
        "attack_name": "known_budget_attacker_score",
        "attacker_knowledge": "budget_aware",
        "detector_name": "budget_aware_risk_estimator",
        "detector_version": "m3c_prototype_v1",
        "confidence_before": float(np.mean(risk_scores)),
        "confidence_after": float(np.mean(weak_risks)),
        "detection_count_before": int(num_blocks),
        "detection_count_after": int(np.sum(weak_mask)),
        "attack_success_rate": attacker_score,
        "leakage_score": attacker_score,
        "status": "success",
        "error_message": "",
    }
