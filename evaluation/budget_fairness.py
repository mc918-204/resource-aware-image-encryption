"""Two independent fairness axes for comparing selective-protection methods.

The defect being fixed (plan item M6)
-------------------------------------
With ``c(none)=0, c(low)=1, c(medium)=3, c(high)=10`` and a declared
budget ``C_B = N + 9*B*N``:

* The proposed method starts from an all-``low`` floor (cost ``N``) and can
  upgrade ``B*N`` blocks to ``high`` (each upgrade costs ``9``).
* The external baselines start from all-``none`` (cost ``0``), so at the
  same ``C_B`` they can afford ``C_B/10 = (0.1 + 0.9*B)*N`` blocks at
  ``high``.

The historical runner nonetheless gave the baselines only
``floor(N*B)`` blocks. At ``B=0.05`` a baseline was **entitled** to
``0.145*N`` but received ``0.050*N`` -- 34.5% budget utilization -- so
"comparable protection ratio" in the manuscript's Table 3 was not
comparable at all, and the unfairness peaked exactly at the budget point
chosen as the headline scenario.

The fix is not one "correct" number of blocks but two clearly separated
questions, reported separately:

**Axis A -- equal coverage** (``equal_coverage``)
    Every method protects the *same* number of blocks ``k = floor(q*N)``
    with the *same* level. This isolates **ranking quality**: who picks
    the better blocks. Cost models are irrelevant here, which removes
    them as a confound.

**Axis B -- equal declared cost** (``equal_declared_cost``)
    Every method receives the same ``C_B`` and must **spend it all**
    under its own level repertoire. This measures allocation under the
    project's cost model, and reports true utilization so under-spending
    is visible instead of hidden.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np

from src.budget_allocator.cost_model import CostModel

__all__ = [
    "AXIS_EQUAL_COVERAGE",
    "AXIS_EQUAL_DECLARED_COST",
    "FAIRNESS_AXES",
    "UPGRADE_BREADTH_FIRST",
    "UPGRADE_DEPTH_FIRST",
    "UPGRADE_STRATEGIES",
    "affordable_high_blocks",
    "budget_greedy_upgrade",
    "equal_coverage_levels",
    "resolve_topk",
    "summarize_budget_usage",
]


AXIS_EQUAL_COVERAGE = "equal_coverage"
AXIS_EQUAL_DECLARED_COST = "equal_declared_cost"
FAIRNESS_AXES: tuple[str, ...] = (AXIS_EQUAL_COVERAGE, AXIS_EQUAL_DECLARED_COST)

_UPGRADE_ORDER: tuple[str, ...] = ("low", "medium", "high")


def resolve_topk(
    num_blocks: int,
    *,
    topk_mode: str,
    topk_value: float,
) -> int:
    """Resolve a top-k block count under one explicit mode.

    Two definitions previously coexisted in the repository -- an absolute
    ``topk: 5`` in the paper config and ``floor(N * budget_ratio)`` inside the
    runner (finding M7). This is the single resolver; callers must name the mode.

    Args:
        num_blocks: Block count ``N`` for the image.
        topk_mode: ``"absolute"`` (``topk_value`` is a block count) or
            ``"ratio"`` (``topk_value`` is a fraction of ``N``).
        topk_value: The count or fraction.

    Returns:
        A count in ``[0, num_blocks]``. Ratio mode rounds *up* so a nonzero
        fraction never resolves to zero blocks on small images.

    Raises:
        ValueError: unknown mode, or a negative/out-of-range value.
    """
    if num_blocks <= 0:
        return 0
    if topk_mode == "absolute":
        if topk_value < 0:
            raise ValueError("absolute topk must be nonnegative")
        return int(min(num_blocks, math.floor(topk_value)))
    if topk_mode == "ratio":
        if not 0.0 <= topk_value <= 1.0:
            raise ValueError("ratio topk must lie in [0, 1]")
        if topk_value == 0.0:
            return 0
        return int(min(num_blocks, max(1, math.ceil(num_blocks * topk_value))))
    raise ValueError(f"unsupported topk_mode: {topk_mode!r}")


def affordable_high_blocks(
    num_blocks: int,
    budget_limit: float,
    *,
    cost_model: CostModel | None = None,
    budget_type: str = "time",
    floor_level: str = "none",
) -> int:
    """How many blocks a method could raise to ``high`` at this budget.

    This is the entitlement figure that exposes M6: comparing it to the blocks
    a method actually protected yields the true utilization.

    Args:
        num_blocks: Block count ``N``.
        budget_limit: Absolute declared budget ``C_B``.
        cost_model: Cost model; the project default when omitted.
        budget_type: Cost axis, ``time`` or ``key``.
        floor_level: The level every unselected block sits at. Baselines start
            from ``none``; the proposed pipeline from ``low``.

    Returns:
        ``floor((C_B - N*c_floor) / (c_high - c_floor))``, clamped to
        ``[0, num_blocks]``.
    """
    model = cost_model or CostModel()
    floor_cost = model.get_cost(floor_level, budget_type)
    high_cost = model.get_cost("high", budget_type)
    if high_cost <= floor_cost:
        return num_blocks
    spare = budget_limit - num_blocks * floor_cost
    if spare <= 0:
        return 0
    return int(min(num_blocks, math.floor(spare / (high_cost - floor_cost))))


def equal_coverage_levels(
    scores: Sequence[float],
    *,
    coverage_ratio: float,
    level: str = "high",
) -> List[str]:
    """Assign ``level`` to the top ``floor(q*N)`` scoring blocks (axis A).

    All methods on this axis get the identical block count and identical
    cryptographic treatment, so the only difference between them is which blocks
    their scores rank highest.

    Args:
        scores: Per-block risk scores.
        coverage_ratio: ``q`` in ``[0, 1]``.
        level: Level applied to the selected blocks.

    Returns:
        A level list with ``level`` on selected blocks and ``none`` elsewhere.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not 0.0 <= coverage_ratio <= 1.0:
        raise ValueError("coverage_ratio must lie in [0, 1]")
    count = int(min(values.size, math.floor(values.size * coverage_ratio)))
    levels = ["none"] * values.size
    if count <= 0:
        return levels
    # Descending score; ties resolved by ascending index for determinism.
    order = np.lexsort((np.arange(values.size), -values))
    for index in order[:count]:
        levels[int(index)] = level
    return levels


UPGRADE_DEPTH_FIRST = "depth_first"
UPGRADE_BREADTH_FIRST = "breadth_first"
UPGRADE_STRATEGIES: tuple[str, ...] = (UPGRADE_DEPTH_FIRST, UPGRADE_BREADTH_FIRST)


def budget_greedy_upgrade(
    scores: Sequence[float],
    budget_limit: float,
    *,
    cost_model: CostModel | None = None,
    budget_type: str = "time",
    floor_level: str = "none",
    allowed_levels: Sequence[str] = _UPGRADE_ORDER,
    strategy: str = UPGRADE_DEPTH_FIRST,
) -> Dict[str, Any]:
    """Spend a declared budget greedily from a floor level (axis B).

    Baselines consume the ``C_B`` they were granted instead of stopping at
    ``floor(N*q)`` blocks, which is the M6 fix. The procedure is shared by all
    baselines, so none is advantaged by allocator sophistication; the exact
    optimum is available separately via
    :class:`~src.budget_allocator.research_allocator.CostAwareAllocator`, and
    the gap between them is a reported result (plan item C8).

    Two strategies are offered because the choice is not neutral and must not be
    made by accident:

    ``depth_first`` (default)
        Walk the score ranking and lift each block as high as the remaining
        budget permits before moving on. This concentrates authenticated
        encryption on the top-ranked blocks and yields the **largest**
        ``recall@strong`` a baseline can reach with this ranking, so it is the
        strongest -- i.e. most demanding -- comparator for the proposed method.
    ``breadth_first``
        Promote every block one rung, then the next, mirroring the frozen
        :class:`~src.budget_allocator.allocator.LegacyHardAllocator`'s
        floor-first behavior. It maximizes the count of *some* protection but
        spends the budget thin, so it reaches strong coverage on fewer blocks.

    The default is ``depth_first`` deliberately: choosing the weaker baseline
    strategy would inflate the proposed method's apparent margin.

    Args:
        scores: Per-block risk scores.
        budget_limit: Absolute declared budget.
        cost_model: Cost model; project default when omitted.
        budget_type: Cost axis.
        floor_level: Starting level for every block.
        allowed_levels: Ascending-cost levels available for promotion.
        strategy: One of :data:`UPGRADE_STRATEGIES`.

    Returns:
        ``{"levels", "budget_used", "budget_limit", "budget_feasible",
        "budget_utilization_true", "upgrade_strategy"}``. ``budget_feasible`` is
        ``False`` when even the floor exceeds the budget; the floor is still
        returned (matching the frozen allocator) but flagged.

    Raises:
        ValueError: unknown ``strategy``.
    """
    if strategy not in UPGRADE_STRATEGIES:
        raise ValueError(
            f"unsupported strategy {strategy!r}; expected one of {list(UPGRADE_STRATEGIES)}"
        )
    model = cost_model or CostModel()
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    count = values.size
    floor_cost = model.get_cost(floor_level, budget_type)
    ladder = sorted(
        (
            level
            for level in allowed_levels
            if model.get_cost(level, budget_type) > floor_cost
        ),
        key=lambda level: model.get_cost(level, budget_type),
    )

    levels = [floor_level] * count
    used = float(count * floor_cost)
    feasible = used <= budget_limit + 1e-9

    if count and feasible and ladder:
        # Descending score; ties by ascending index for determinism.
        order = [int(index) for index in np.lexsort((np.arange(count), -values))]
        if strategy == UPGRADE_DEPTH_FIRST:
            for index in order:
                for target_level in reversed(ladder):
                    delta = model.get_cost(target_level, budget_type) - model.get_cost(
                        levels[index], budget_type
                    )
                    if delta <= 0:
                        continue
                    if used + delta <= budget_limit + 1e-9:
                        used += delta
                        levels[index] = target_level
                        break
        else:
            for target_level in ladder:
                target_cost = model.get_cost(target_level, budget_type)
                for index in order:
                    delta = target_cost - model.get_cost(levels[index], budget_type)
                    if delta <= 0:
                        continue
                    if used + delta <= budget_limit + 1e-9:
                        used += delta
                        levels[index] = target_level
    return {
        "levels": levels,
        "budget_used": float(used),
        "budget_limit": float(budget_limit),
        "budget_feasible": bool(feasible),
        "budget_utilization_true": (
            float(used / budget_limit) if budget_limit > 0 else None
        ),
        "upgrade_strategy": strategy,
    }


def summarize_budget_usage(
    levels: Sequence[str],
    budget_limit: float,
    *,
    cost_model: CostModel | None = None,
    budget_type: str = "time",
    floor_level: str = "none",
) -> Dict[str, Any]:
    """Report true budget usage and the affordable-high entitlement.

    Recording both makes M6 auditable directly from ``raw_results.csv``: a row
    whose ``budget_utilization_true`` is far below 1.0 while
    ``budget_affordable_high_blocks`` exceeds its actual strong-block count was
    not competing on equal terms.
    """
    model = cost_model or CostModel()
    used = float(sum(model.get_cost(level, budget_type) for level in levels))
    return {
        "budget_used": used,
        "budget_utilization_true": (
            float(used / budget_limit) if budget_limit > 0 else None
        ),
        "budget_feasible": bool(used <= budget_limit + 1e-9),
        "budget_affordable_high_blocks": affordable_high_blocks(
            len(levels),
            budget_limit,
            cost_model=model,
            budget_type=budget_type,
            floor_level=floor_level,
        ),
    }
