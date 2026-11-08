"""Explicit budget units for additive research experiments.

A normalized paper budget is an interpolation between the all-low and all-high
costs for the current image. It is not an absolute allocator budget. This module
does not change the frozen allocator's behavior for infeasible absolute budgets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, Literal

from src.budget_allocator.cost_model import CostModel

BudgetMode = Literal["absolute", "paper_low_high"]


def _finite_nonnegative(value: Any, field: str) -> float:
    """Reject nonnumeric, boolean, negative and nonfinite inputs explicitly."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a finite nonnegative real number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return number


@dataclass(frozen=True)
class ResolvedResearchBudget:
    """Input units and per-image absolute cost, suitable for artifact recording."""

    mode: BudgetMode
    requested_budget: float
    block_count: int
    budget_type: str
    low_cost_per_block: float
    high_cost_per_block: float
    all_low_cost: float
    all_high_cost: float
    budget_limit: float
    below_legacy_floor: bool

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible record of the conversion."""
        return asdict(self)


def resolve_research_budget(
    value: float,
    block_count: int,
    *,
    mode: BudgetMode = "absolute",
    cost_model: CostModel | None = None,
    budget_type: str = "time",
) -> ResolvedResearchBudget:
    """Resolve declared units before allocation, without inferring a mode.

    ``paper_low_high`` requires 0 <= value <= 1 and computes
    N*c_low + value*(N*c_high - N*c_low). ``absolute`` preserves the
    supplied cost, including values below the historical all-low floor.
    The latter are flagged, not silently raised to the minimum cost.
    Cost units are model units, not measured wall-clock milliseconds.
    """
    if mode not in ("absolute", "paper_low_high"):
        raise ValueError(f"unsupported budget mode: {mode!r}")
    if isinstance(block_count, bool) or not isinstance(block_count, Integral):
        raise TypeError("block_count must be a nonnegative integer")
    count = int(block_count)
    if count < 0:
        raise ValueError("block_count must be nonnegative")
    requested = _finite_nonnegative(value, "budget")
    if mode == "paper_low_high" and requested > 1:
        raise ValueError("paper_low_high budget must lie in [0, 1]")
    model = CostModel() if cost_model is None else cost_model
    if not isinstance(model, CostModel):
        raise TypeError("cost_model must be a CostModel")
    low = _finite_nonnegative(model.get_cost("low", budget_type), "low cost")
    high = _finite_nonnegative(model.get_cost("high", budget_type), "high cost")
    if high < low:
        raise ValueError("high cost must be at least low cost")
    try:
        floor, ceiling = count * low, count * high
    except OverflowError as exc:
        raise ValueError("per-image cost must be finite") from exc
    if not (math.isfinite(floor) and math.isfinite(ceiling)):
        raise ValueError("per-image cost must be finite")
    limit = requested if mode == "absolute" else floor + requested * (ceiling - floor)
    if not math.isfinite(limit):
        raise ValueError("resolved budget must be finite")
    return ResolvedResearchBudget(
        mode=mode, requested_budget=requested, block_count=count,
        budget_type=budget_type, low_cost_per_block=low,
        high_cost_per_block=high, all_low_cost=floor, all_high_cost=ceiling,
        budget_limit=limit, below_legacy_floor=limit < floor,
    )
