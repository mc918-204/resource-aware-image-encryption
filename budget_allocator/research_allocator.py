"""Exact research-only allocation over configurable protection levels."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import reduce
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from src.budget_allocator.allocator import _build_result
from src.budget_allocator.cost_model import CostModel


LEVEL_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
_TIE_BREAK_RULE = "objective_desc,cost_asc,level_rank_input_order_desc"
_MAX_DECIMAL_PRECISION = 4096
_RANK_BITS_PER_BLOCK = 2

MAX_BLOCKS = 10_000
MAX_STATES = 2_048
MAX_TRANSITIONS = 12_000_000
MAX_BUDGET_UNITS = 10_000_000
MAX_BUDGET_SCALE_DIGITS = 12
MAX_INTEGER_BITS = 32_768


@dataclass(frozen=True)
class AllocationPolicy:
    """Configuration for exact multiple-choice-knapsack allocation."""

    allowed_levels: Sequence[str]
    protection_utility: Mapping[str, float | Decimal]
    budget_scale: int
    minimum_level: str = "none"


@dataclass(frozen=True, slots=True)
class _AssignmentNode:
    """One immutable assignment link used to recover the selected levels."""

    parent: _AssignmentNode | None
    level: str


@dataclass(frozen=True, slots=True)
class _DPState:
    """Best objective and lexicographic rank for one exact cost."""

    objective: int
    rank_code: int
    node: _AssignmentNode | None


def _as_decimal(value: Any, *, field: str) -> Decimal:
    """Convert a real numeric value to a finite Decimal without coercing text."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, Decimal)):
        raise TypeError(f"{field} must be a finite numeric value")
    try:
        converted = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be a finite numeric value") from exc
    if not converted.is_finite():
        raise ValueError(f"{field} must be a finite numeric value")
    return converted


def _decimal_ratio(value: Decimal, *, field: str) -> tuple[int, int]:
    """Return the exact integer numerator and denominator of a Decimal."""
    sign, digits, exponent = value.as_tuple()
    if all(digit == 0 for digit in digits):
        return 0, 1
    if (
        len(digits) > _MAX_DECIMAL_PRECISION
        or abs(exponent) > _MAX_DECIMAL_PRECISION
    ):
        raise ValueError(
            f"{field} exceeds supported decimal precision "
            f"({_MAX_DECIMAL_PRECISION} digits/exponent)"
        )
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    if sign:
        coefficient = -coefficient
    if exponent >= 0:
        return coefficient * (10**exponent), 1
    return coefficient, 10 ** (-exponent)


def _finite_result_float(value: Any, *, field: str) -> float:
    """Convert a reporting value to float or raise a field-specific error."""
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be representable as a finite float") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be representable as a finite float")
    if converted == 0.0 and value != 0:
        raise ValueError(f"{field} must be representable as a finite float")
    return converted


def _validate_fraction_bits(value: Fraction, *, field: str) -> None:
    """Reject exact arithmetic values that exceed the configured bit bound."""
    if (
        value.numerator.bit_length() > MAX_INTEGER_BITS
        or value.denominator.bit_length() > MAX_INTEGER_BITS
    ):
        raise ValueError(f"{field} exceeds integer bit-length limit")


def _reachable_state_upper_bound(
    *, block_count: int, costs: Mapping[str, int], budget_units: int
) -> int:
    """Bound reachable exact costs using the gcd of eligible cost deltas."""
    base_cost = min(costs.values())
    floor_units = block_count * base_cost
    deltas = tuple(cost - base_cost for cost in costs.values())
    positive_deltas = tuple(delta for delta in deltas if delta > 0)
    if not positive_deltas or block_count == 0:
        return 1
    step = reduce(math.gcd, positive_deltas)
    reachable_span = min(
        budget_units - floor_units,
        block_count * max(positive_deltas),
    )
    return reachable_span // step + 1


class CostAwareAllocator:
    """Allocate levels optimally under a discretized, exactly scaled budget."""

    allocator_type = "cost_aware_dp"

    def __init__(self, cost_model: CostModel, policy: AllocationPolicy) -> None:
        """Validate and snapshot a research allocation policy."""
        if not isinstance(cost_model, CostModel):
            raise TypeError("cost_model must be a CostModel")
        if not isinstance(policy, AllocationPolicy):
            raise TypeError("policy must be an AllocationPolicy")

        raw_levels = policy.allowed_levels
        if isinstance(raw_levels, (str, bytes)) or not isinstance(raw_levels, Sequence):
            raise TypeError("allowed_levels must be a nonempty sequence of level names")
        levels = tuple(raw_levels)
        if not levels:
            raise ValueError("allowed_levels must not be empty")
        if any(not isinstance(level, str) or level not in LEVEL_RANK for level in levels):
            raise ValueError("allowed_levels contains an unsupported level")
        if len(set(levels)) != len(levels):
            raise ValueError("allowed_levels must not contain duplicates")

        utilities = policy.protection_utility
        if not isinstance(utilities, Mapping):
            raise TypeError("protection_utility must be a mapping")
        if set(utilities) != set(levels):
            raise ValueError("protection_utility keys must exactly match allowed_levels")
        utility_fractions: dict[str, Fraction] = {}
        for level in levels:
            utility = _as_decimal(
                utilities[level], field=f"protection_utility[{level}]"
            )
            if utility < 0:
                raise ValueError("protection_utility values must be nonnegative")
            numerator, denominator = _decimal_ratio(
                utility, field=f"protection_utility[{level}]"
            )
            _finite_result_float(
                utility, field=f"protection_utility[{level}]"
            )
            utility_fractions[level] = Fraction(numerator, denominator)

        scale = policy.budget_scale
        if isinstance(scale, (bool, np.bool_)) or not isinstance(scale, Integral):
            raise TypeError("budget_scale must be a positive integer")
        scale_value = int(scale)
        if scale_value <= 0:
            raise ValueError("budget_scale must be a positive integer")
        if scale_value >= 10**MAX_BUDGET_SCALE_DIGITS:
            raise ValueError(
                "budget_scale exceeds configured digit limit "
                f"({MAX_BUDGET_SCALE_DIGITS})"
            )

        minimum = policy.minimum_level
        if not isinstance(minimum, str) or minimum not in LEVEL_RANK:
            raise ValueError("minimum_level is unsupported")
        eligible = tuple(
            level for level in levels if LEVEL_RANK[level] >= LEVEL_RANK[minimum]
        )
        if not eligible:
            raise ValueError(
                f"minimum_level={minimum} leaves no eligible allowed_levels"
            )

        self.cost_model = cost_model
        self.policy = AllocationPolicy(
            allowed_levels=levels,
            protection_utility=MappingProxyType(
                {level: utilities[level] for level in levels}
            ),
            budget_scale=scale_value,
            minimum_level=minimum,
        )
        self._levels = levels
        self._eligible_levels = eligible
        self._utility_fractions = utility_fractions
        self._budget_scale = scale_value
        self._minimum_level = minimum

    def _scaled_cost(self, level: str, budget_type: str) -> int:
        """Return a level cost in exact integer budget units."""
        raw_cost = self.cost_model.get_cost(level, budget_type)
        cost = _as_decimal(raw_cost, field=f"cost for level {level}")
        if cost < 0:
            raise ValueError(f"cost for level {level} must be nonnegative")
        numerator, denominator = _decimal_ratio(
            cost, field=f"cost for level {level}"
        )
        _finite_result_float(cost, field=f"cost for level {level}")
        units, remainder = divmod(numerator * self._budget_scale, denominator)
        if remainder:
            raise ValueError(
                f"cost for level {level} ({raw_cost}) is not integral at "
                f"budget_scale={self._budget_scale}"
            )
        return units

    @staticmethod
    def _scores(scores: Any) -> np.ndarray:
        """Return a private flat float64 copy after strict score validation."""
        try:
            raw = np.asarray(scores)
        except (TypeError, ValueError) as exc:
            raise TypeError("scores must contain finite numeric values") from exc
        if raw.size > MAX_BLOCKS:
            raise ValueError(
                f"scores block count exceeds configured limit ({MAX_BLOCKS})"
            )
        if raw.dtype.kind not in {"i", "u", "f"}:
            raise TypeError("scores must contain finite numeric non-boolean values")
        values = np.array(raw, dtype=np.float64, copy=True).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("scores must contain only finite values")
        return values

    def allocate(
        self,
        scores: np.ndarray,
        budget: float | Decimal,
        budget_type: str = "time",
    ) -> dict[str, Any]:
        """Maximize utility; scores use ``Decimal(str(float(score)))`` exactly."""
        values = self._scores(scores)
        budget_decimal = _as_decimal(budget, field="budget")
        if budget_decimal < 0:
            raise ValueError("budget must be nonnegative")
        budget_numerator, budget_denominator = _decimal_ratio(
            budget_decimal, field="budget"
        )
        reported_budget = _finite_result_float(budget_decimal, field="budget")

        costs = {
            level: self._scaled_cost(level, budget_type)
            for level in self._eligible_levels
        }
        budget_units = (
            budget_numerator * self._budget_scale // budget_denominator
        )
        if budget_units.bit_length() > MAX_INTEGER_BITS:
            raise ValueError("budget_units exceeds integer bit-length limit")
        if budget_units > MAX_BUDGET_UNITS:
            raise ValueError(
                f"budget_units exceeds configured limit ({MAX_BUDGET_UNITS})"
            )
        minimum_cost_units = min(costs.values())
        floor_units = len(values) * minimum_cost_units
        if floor_units > budget_units:
            raise ValueError(
                f"minimum_level={self._minimum_level} costs {floor_units} units "
                f"but budget is {budget_units} units"
            )

        reachable_states = _reachable_state_upper_bound(
            block_count=len(values), costs=costs, budget_units=budget_units
        )
        if reachable_states > MAX_STATES:
            raise ValueError(
                "reachable DP state upper bound exceeds configured DP state "
                f"limit ({reachable_states} > {MAX_STATES})"
            )
        transition_estimate = (
            len(values) * len(self._eligible_levels) * reachable_states
        )
        if transition_estimate > MAX_TRANSITIONS:
            raise ValueError(
                "DP transition estimate exceeds configured limit "
                f"({transition_estimate} > {MAX_TRANSITIONS})"
            )

        exact_contributions: list[tuple[Fraction, ...]] = []
        common_denominator = 1
        for score in values:
            score_decimal = Decimal(str(float(score)))
            score_numerator, score_denominator = _decimal_ratio(
                score_decimal, field="scores"
            )
            score_fraction = Fraction(score_numerator, score_denominator)
            block_contributions = tuple(
                score_fraction * self._utility_fractions[level]
                for level in self._eligible_levels
            )
            for contribution in block_contributions:
                _validate_fraction_bits(
                    contribution, field="allocation arithmetic"
                )
                common_denominator = math.lcm(
                    common_denominator, contribution.denominator
                )
                if common_denominator.bit_length() > MAX_INTEGER_BITS:
                    raise ValueError(
                        "allocation common denominator exceeds integer "
                        "bit-length limit"
                    )
            exact_contributions.append(block_contributions)

        integer_contributions = [
            tuple(
                contribution.numerator
                * (common_denominator // contribution.denominator)
                for contribution in block
            )
            for block in exact_contributions
        ]
        for block in integer_contributions:
            if any(value.bit_length() > MAX_INTEGER_BITS for value in block):
                raise ValueError(
                    "allocation contribution exceeds integer bit-length limit"
                )

        states: dict[int, _DPState] = {
            0: _DPState(0, 0, None)
        }
        transitions = 0
        for index, contributions in enumerate(integer_contributions):
            remaining_floor_units = (
                len(values) - index - 1
            ) * minimum_cost_units
            next_states: dict[int, _DPState] = {}
            for used_units, state in states.items():
                for level_index, level in enumerate(self._eligible_levels):
                    transitions += 1
                    if transitions > MAX_TRANSITIONS:
                        raise ValueError(
                            "DP transition count exceeds configured limit "
                            f"({MAX_TRANSITIONS})"
                        )
                    new_cost = used_units + costs[level]
                    if new_cost + remaining_floor_units > budget_units:
                        continue
                    candidate_objective = (
                        state.objective + contributions[level_index]
                    )
                    if candidate_objective.bit_length() > MAX_INTEGER_BITS:
                        raise ValueError(
                            "allocation objective exceeds integer bit-length limit"
                        )
                    incumbent = next_states.get(new_cost)
                    if (
                        incumbent is not None
                        and candidate_objective < incumbent.objective
                    ):
                        continue
                    candidate_rank = (
                        state.rank_code << _RANK_BITS_PER_BLOCK
                    ) | LEVEL_RANK[level]
                    if candidate_rank.bit_length() > MAX_INTEGER_BITS:
                        raise ValueError(
                            "allocation rank code exceeds integer bit-length limit"
                        )
                    if incumbent is None or (
                        candidate_objective,
                        candidate_rank,
                    ) > (incumbent.objective, incumbent.rank_code):
                        next_states[new_cost] = _DPState(
                            candidate_objective,
                            candidate_rank,
                            _AssignmentNode(state.node, level),
                        )
            if not next_states:
                raise ValueError(
                    "no feasible dynamic-programming states remain after "
                    f"input block {index}"
                )

            # A lower-cost prefix with an equal or better objective dominates
            # every continuation of a higher-cost prefix. Equal-cost states
            # were already reduced by objective and input-order rank above.
            pruned_states: dict[int, _DPState] = {}
            best_lower_cost_objective: int | None = None
            for exact_cost in sorted(next_states):
                state = next_states[exact_cost]
                if (
                    best_lower_cost_objective is not None
                    and state.objective <= best_lower_cost_objective
                ):
                    continue
                pruned_states[exact_cost] = state
                best_lower_cost_objective = state.objective
            if len(pruned_states) > MAX_STATES:
                raise ValueError(
                    f"DP state count exceeds configured limit ({MAX_STATES})"
                )
            states = pruned_states

        if not states:
            raise ValueError("no feasible dynamic-programming allocation state")
        used_units, selected = max(
            states.items(),
            key=lambda item: (
                item[1].objective,
                -item[0],
                item[1].rank_code,
            ),
        )
        assigned: list[str] = []
        node = selected.node
        while node is not None:
            assigned.append(node.level)
            node = node.parent
        assigned.reverse()
        if len(assigned) != len(values):
            raise RuntimeError("allocation back-reference length mismatch")
        used = _finite_result_float(
            Fraction(used_units, self._budget_scale), field="budget_used"
        )
        reported_objective = _finite_result_float(
            Fraction(selected.objective, common_denominator),
            field="allocation_objective",
        )
        result = _build_result(
            assigned, used, reported_budget, self.allocator_type
        )
        result["stats"].update(
            {
                "allocation_objective": reported_objective,
                "optimality_method": "exact_multiple_choice_knapsack_dp",
                "budget_scale": self._budget_scale,
                "tie_break_rule": _TIE_BREAK_RULE,
            }
        )
        return result
