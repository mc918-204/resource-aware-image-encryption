"""Core selective-protection metrics for the publication benchmark."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import numpy as np


DEFAULT_PROTECTION_WEIGHTS = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.25,
    "none": 0.0,
}

SELECTIVE_METRIC_OUTPUT_KEYS = (
    "semantic_exposure_rate",
    "topk_sensitive_exposure",
    "budget_normalized_protection_gain",
    "oracle_gap",
    "utility",
    "utility_per_cost",
    "protection_density",
    "effective_topk",
    "total_budget",
)

BUDGET_DEPENDENT_SELECTIVE_METRICS = {
    "budget_normalized_protection_gain",
    "utility_per_cost",
}

SELECTIVE_NOT_APPLICABLE_NOTES = {
    "reference_method": "not_applicable_for_reference_method",
    "zero_budget": "not_applicable_for_zero_budget",
}


class SelectiveMetricInputError(ValueError):
    """Raised when selective-metric inputs are structurally invalid."""


def _normalize_inputs(
    risk_scores: Sequence[float],
    protection_levels: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    scores = np.asarray(risk_scores, dtype=np.float64)
    levels = list(protection_levels)
    if scores.ndim != 1:
        raise SelectiveMetricInputError(f"risk_scores must be 1D, got shape {scores.shape}.")
    if len(levels) != int(scores.size):
        raise SelectiveMetricInputError(
            "Risk/protection length mismatch: "
            f"risk_scores={int(scores.size)}, protection_levels={len(levels)}."
        )
    return scores, levels


def _resolve_topk_count(num_blocks: int, topk: int | float | None) -> int:
    if num_blocks <= 0:
        return 0
    if topk is None:
        return max(1, int(np.ceil(num_blocks * 0.1)))
    if isinstance(topk, float):
        if topk <= 0:
            return 0
        if topk < 1:
            return max(1, int(np.ceil(num_blocks * topk)))
        return min(num_blocks, int(topk))
    return min(num_blocks, max(0, int(topk)))


def _level_weights(levels: Sequence[str], protection_weights: Dict[str, float]) -> np.ndarray:
    return np.asarray([protection_weights.get(level, 0.0) for level in levels], dtype=np.float64)


def semantic_exposure_rate(
    risk_scores: Sequence[float],
    protection_levels: Sequence[str],
    strong_levels: Iterable[str] = ("high",),
) -> float:
    """
    Risk-weighted exposure among sensitive blocks.

    Exposure is the fraction of sensitive-risk mass that is not protected by a
    strong protection level.
    """
    scores, levels = _normalize_inputs(risk_scores, protection_levels)
    if scores.size == 0:
        return 0.0
    strong = set(strong_levels)
    threshold = float(np.quantile(scores, 0.7))
    sensitive_mask = scores >= threshold
    denominator = float(np.sum(scores[sensitive_mask]))
    if denominator <= 0:
        return 0.0
    weak_mask = np.array([level not in strong for level in levels], dtype=bool)
    exposed = float(np.sum(scores[sensitive_mask & weak_mask]))
    return float(exposed / denominator)


def topk_sensitive_exposure(
    risk_scores: Sequence[float],
    protection_levels: Sequence[str],
    topk: int | float | None = None,
    strong_levels: Iterable[str] = ("high",),
) -> float:
    """Risk-weighted exposure inside the top-k highest-risk blocks."""
    scores, levels = _normalize_inputs(risk_scores, protection_levels)
    if scores.size == 0:
        return 0.0
    strong = set(strong_levels)
    k = _resolve_topk_count(scores.size, topk)
    if k <= 0:
        return 0.0
    top_indices = np.argsort(scores)[::-1][:k]
    denominator = float(np.sum(scores[top_indices]))
    if denominator <= 0:
        return 0.0
    top_levels = np.array(levels)[top_indices]
    exposed = float(np.sum(scores[top_indices][[lvl not in strong for lvl in top_levels]]))
    return exposed / denominator


def budget_normalized_protection_gain(
    risk_scores: Sequence[float],
    protection_levels: Sequence[str],
    budget_used: float,
    protection_weights: Dict[str, float] | None = None,
) -> float | None:
    """Risk-weighted protection gain per unit budget."""
    weights = protection_weights or DEFAULT_PROTECTION_WEIGHTS
    scores, levels = _normalize_inputs(risk_scores, protection_levels)
    if scores.size == 0:
        return 0.0
    weighted_gain = float(np.sum(scores * _level_weights(levels, weights)))
    if budget_used <= 0:
        return None
    return weighted_gain / float(budget_used)


def utility(
    risk_scores: Sequence[float],
    protection_levels: Sequence[str],
    protection_weights: Dict[str, float] | None = None,
) -> float:
    """Normalized protected-risk utility under the configured level weights."""
    weights = protection_weights or DEFAULT_PROTECTION_WEIGHTS
    scores, levels = _normalize_inputs(risk_scores, protection_levels)
    if scores.size == 0:
        return 0.0
    denominator = float(np.sum(scores))
    if denominator <= 0:
        return 0.0
    protected = float(np.sum(scores * _level_weights(levels, weights)))
    return protected / denominator


def utility_per_cost(
    risk_scores: Sequence[float],
    protection_levels: Sequence[str],
    budget_used: float,
    protection_weights: Dict[str, float] | None = None,
) -> float | None:
    """Normalized utility per unit budget."""
    value = utility(risk_scores, protection_levels, protection_weights=protection_weights)
    if budget_used <= 0:
        return None
    return value / float(budget_used)


def protection_density(protection_levels: Sequence[str]) -> float:
    """Fraction of blocks that receive any non-none protection."""
    lvls = np.asarray(protection_levels)
    if lvls.size == 0:
        return 0.0
    return float(np.sum(lvls != "none") / lvls.size)


def oracle_gap(
    risk_scores: Sequence[float],
    protection_levels: Sequence[str],
    level_costs: Sequence[float],
    budget_used: float,
    protection_weights: Dict[str, float] | None = None,
    oracle_allocation: Sequence[str] | None = None,
) -> float | None:
    """
    Utility gap to an oracle allocation under the same budget.

    If `oracle_allocation` is not provided, returns ``None`` rather than
    fabricating an oracle.
    """
    if oracle_allocation is None:
        return None
    del level_costs
    weights = protection_weights or DEFAULT_PROTECTION_WEIGHTS
    current = utility(risk_scores, protection_levels, protection_weights=weights)
    oracle_value = utility(risk_scores, oracle_allocation, protection_weights=weights)
    del budget_used
    return float(oracle_value - current)


def compute_selective_metrics(
    *,
    risk_scores: Sequence[float],
    protection_levels: Sequence[str],
    budget_used: float,
    total_budget: float,
    level_costs: Sequence[float],
    topk: int | float | None = None,
    strong_levels: Iterable[str] = ("high",),
    protection_weights: Dict[str, float] | None = None,
    oracle_allocation: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Compute the publication-facing selective protection metrics."""
    weights = protection_weights or DEFAULT_PROTECTION_WEIGHTS
    scores, levels = _normalize_inputs(risk_scores, protection_levels)
    return {
        "semantic_exposure_rate": semantic_exposure_rate(
            scores,
            levels,
            strong_levels=strong_levels,
        ),
        "topk_sensitive_exposure": topk_sensitive_exposure(
            scores,
            levels,
            topk=topk,
            strong_levels=strong_levels,
        ),
        "budget_normalized_protection_gain": budget_normalized_protection_gain(
            scores,
            levels,
            budget_used=budget_used,
            protection_weights=weights,
        ),
        "oracle_gap": oracle_gap(
            scores,
            levels,
            level_costs=level_costs,
            budget_used=budget_used,
            protection_weights=weights,
            oracle_allocation=oracle_allocation,
        ),
        "utility": utility(
            scores,
            levels,
            protection_weights=weights,
        ),
        "utility_per_cost": utility_per_cost(
            scores,
            levels,
            budget_used=budget_used,
            protection_weights=weights,
        ),
        "protection_density": protection_density(levels),
        "effective_topk": _resolve_topk_count(len(levels), topk),
        "total_budget": total_budget,
    }


def empty_selective_metrics(status: str = "metric_not_applicable") -> Dict[str, Any]:
    """Return a complete selective-metric payload for expected N/A cases."""
    payload = {key: None for key in SELECTIVE_METRIC_OUTPUT_KEYS}
    payload["metric_status"] = status
    payload["unsupported_metrics"] = list(SELECTIVE_METRIC_OUTPUT_KEYS)
    return payload
