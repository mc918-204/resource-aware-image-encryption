"""Exact score-weighted authentication coverage under uniform block costs.

This additive research reference keeps the frozen V1 allocators unchanged.
All blocks retain at least low protection. High and medium have equal utility
for this objective, so the cheaper authenticated operation and a top-k ranking
give an exact solution. This is not an oracle for externally labelled recall.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np

from .allocator import _build_result
from .cost_model import CostModel


class AuthenticationCoverageAllocator:
    """Maximize sum(scores[i] * is_authenticated[i]) with a low-level floor."""

    allocator_type = "authentication_coverage_exact"

    def __init__(self, cost_model: CostModel | None = None) -> None:
        self.cost_model = cost_model or CostModel()

    def allocate(self, scores: np.ndarray, budget: float, budget_type: str = "time") -> dict[str, Any]:
        """Return an exact top-k allocation; reject an infeasible low floor."""
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("scores must be a finite nonnegative one-dimensional array")
        if not np.isfinite(budget) or budget < 0:
            raise ValueError("budget must be finite and nonnegative")
        costs = {level: Decimal(str(self.cost_model.get_cost(level, budget_type)))
                 for level in ("low", "medium", "high")}
        if any(not value.is_finite() or value < 0 for value in costs.values()):
            raise ValueError("costs must be finite and nonnegative")
        strong = min(("medium", "high"), key=lambda level: (costs[level], level))
        increment = costs[strong] - costs["low"]
        if increment <= 0:
            raise ValueError("this reference requires positive authentication upgrade cost")
        floor = len(values) * costs["low"]
        limit = Decimal(str(budget))
        if floor > limit:
            raise ValueError("budget does not cover the common all-low floor")
        count = min(len(values), int((limit - floor) // increment))
        order = np.argsort(-values, kind="stable")
        levels = ["low"] * len(values)
        for index in order[:count]:
            levels[int(index)] = strong
        cost = floor + count * increment
        result = _build_result(levels, float(cost), float(budget), self.allocator_type)
        result["stats"].update({
            "optimality_method": "uniform_increment_top_k_exchange_argument",
            "allocation_objective": float(np.sum(values[order[:count]])),
            "objective_definition": "score_weighted_authentication_coverage",
            "minimum_level": "low",
        })
        return result
