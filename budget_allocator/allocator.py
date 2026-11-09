import numpy as np
from collections import Counter
from typing import List, Dict, Any
from .cost_model import CostModel


def _build_result(levels: List[str], cost: float, budget: float, allocator_type: str) -> Dict[str, Any]:
    """Shared result builder for all allocator variants."""
    counts = Counter(levels)
    stats = {
        "high": counts.get("high", 0),
        "medium": counts.get("medium", 0),
        "low": counts.get("low", 0),
        "none": counts.get("none", 0),
    }
    total = len(levels)
    if total > 0:
        stats["high_ratio"] = stats["high"] / total
        stats["medium_ratio"] = stats["medium"] / total
        stats["low_ratio"] = stats["low"] / total
        stats["none_ratio"] = stats["none"] / total
    else:
        stats.update({"high_ratio": 0, "medium_ratio": 0, "low_ratio": 0, "none_ratio": 0})

    stats["budget_limit"] = budget
    stats["budget_used"] = cost
    stats["budget_utilization"] = (cost / budget * 100.0) if budget > 0 else 0.0

    if stats["none"] == total and total > 0:
        protection_level = "DEGRADED"
    elif stats["none"] > 0:
        protection_level = "DEGRADED"
    elif stats["high"] == total:
        protection_level = "FULL"
    else:
        protection_level = "PARTIAL"

    stats["feasibility_status"] = protection_level
    stats["protection_level"] = protection_level
    stats["allocator_type"] = allocator_type

    return {
        "levels": levels,
        "total_cost": cost,
        "stats": stats
    }


class LegacyHardAllocator:
    """
    Original V2 allocator using fixed percentage thresholds.
    Phase 1: All blocks start as "none", upgrade top blocks to "low" if budget allows.
    Phase 2: Top 30% -> "high", next 40% -> "medium"
    Phase 3: Fill gaps by upgrading low->medium->high greedily.

    This is the BASELINE that must remain unchanged for reproducibility.
    """

    def __init__(self, cost_model: CostModel = None):
        self.cost_model = cost_model or CostModel()
        self.allocator_type = "legacy_hard"

    def allocate(self,
                 scores: np.ndarray,
                 budget: float,
                 budget_type: str = "time") -> Dict[str, Any]:
        n_blocks = len(scores)
        if n_blocks == 0:
            return _build_result([], 0.0, budget, self.allocator_type)

        c_low = self.cost_model.get_cost("low", budget_type)
        c_med = self.cost_model.get_cost("medium", budget_type)
        c_high = self.cost_model.get_cost("high", budget_type)

        sorted_indices = np.argsort(scores)[::-1]

        # V1-compatible baseline: all blocks receive at least low protection, even
        # when the requested budget is below the low-level floor.
        levels = ["low"] * n_blocks
        current_cost = n_blocks * c_low
        if budget < current_cost:
            return _build_result(levels, current_cost, budget, self.allocator_type)

        n_top = int(n_blocks * 0.3)
        top_indices = sorted_indices[:n_top]

        for idx in top_indices:
            upgrade_cost = c_high - c_low
            if current_cost + upgrade_cost <= budget:
                levels[idx] = "high"
                current_cost += upgrade_cost

        for idx in sorted_indices:
            if levels[idx] != "low":
                continue
            upgrade_cost = c_med - c_low
            if current_cost + upgrade_cost <= budget:
                levels[idx] = "medium"
                current_cost += upgrade_cost

        for idx in sorted_indices:
            lvl = levels[idx]
            if lvl == "low":
                if current_cost + (c_med - c_low) <= budget:
                    levels[idx] = "medium"
                    current_cost += (c_med - c_low)
                    lvl = "medium"

            if lvl == "medium":
                if current_cost + (c_high - c_med) <= budget:
                    levels[idx] = "high"
                    current_cost += (c_high - c_med)

        return _build_result(levels, current_cost, budget, self.allocator_type)


class ThresholdAllocator:
    """
    Improved allocator using score-based thresholds.
    Uses HIGH_THRESHOLD and MEDIUM_THRESHOLD to determine upgrade eligibility.

    This is a NEW VARIANT, not the original V2 behavior.
    """

    # Frozen defaults. These two constants were set from testing experience and
    # never selected on a validation split; the ablation in the paper shows they
    # are what causes this allocator to leave budget unspent at high budgets, so
    # they are now overridable per instance while the class defaults preserve
    # every previously recorded result.
    HIGH_THRESHOLD = 0.7
    MEDIUM_THRESHOLD = 0.4

    def __init__(
        self,
        cost_model: CostModel = None,
        high_threshold: float | None = None,
        medium_threshold: float | None = None,
    ):
        """Build a threshold allocator.

        Args:
            cost_model: Level cost model; the project default when omitted.
            high_threshold: Score at or above which a block may be raised to
                ``high``. Defaults to :data:`HIGH_THRESHOLD`.
            medium_threshold: Score at or above which a block may be raised to
                ``medium``. Defaults to :data:`MEDIUM_THRESHOLD`.

        Raises:
            ValueError: a threshold lies outside [0, 1], or the medium threshold
                exceeds the high threshold, which would make the medium pass
                unreachable.
        """
        self.cost_model = cost_model or CostModel()
        self.allocator_type = "threshold"
        self.high_threshold = (
            self.HIGH_THRESHOLD if high_threshold is None else float(high_threshold)
        )
        self.medium_threshold = (
            self.MEDIUM_THRESHOLD if medium_threshold is None else float(medium_threshold)
        )
        for name, value in (
            ("high_threshold", self.high_threshold),
            ("medium_threshold", self.medium_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")
        if self.medium_threshold > self.high_threshold:
            raise ValueError(
                "medium_threshold must not exceed high_threshold "
                f"({self.medium_threshold} > {self.high_threshold})"
            )

    def allocate(self,
                 scores: np.ndarray,
                 budget: float,
                 budget_type: str = "time") -> Dict[str, Any]:
        n_blocks = len(scores)
        if n_blocks == 0:
            return _build_result([], 0.0, budget, self.allocator_type)

        c_low = self.cost_model.get_cost("low", budget_type)
        c_med = self.cost_model.get_cost("medium", budget_type)
        c_high = self.cost_model.get_cost("high", budget_type)

        sorted_indices = np.argsort(scores)[::-1]

        levels = ["low"] * n_blocks
        current_cost = n_blocks * c_low
        if budget < current_cost:
            return _build_result(levels, current_cost, budget, self.allocator_type)

        high_upgrade_cost = c_high - c_low
        med_upgrade_cost = c_med - c_low

        for idx in sorted_indices:
            if levels[idx] != "low":
                continue
            if scores[idx] >= self.high_threshold:
                if current_cost + high_upgrade_cost <= budget:
                    levels[idx] = "high"
                    current_cost += high_upgrade_cost

        for idx in sorted_indices:
            if levels[idx] != "low":
                continue
            if scores[idx] >= self.medium_threshold:
                if current_cost + med_upgrade_cost <= budget:
                    levels[idx] = "medium"
                    current_cost += med_upgrade_cost

        return _build_result(levels, current_cost, budget, self.allocator_type)


BudgetAllocator = LegacyHardAllocator
