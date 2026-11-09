"""Additive allocator registry preserving the frozen V1 implementation."""

from __future__ import annotations

from typing import Any, Mapping

from src.budget_allocator.allocator import LegacyHardAllocator, ThresholdAllocator
from src.budget_allocator.cost_model import CostModel
from src.budget_allocator.research_allocator import AllocationPolicy, CostAwareAllocator


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    """Validate a nested configuration node with a field-specific error."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _required(config: Mapping[str, Any], key: str, *, field: str) -> Any:
    """Read a required configuration key without leaking a bare KeyError."""
    if key not in config:
        raise ValueError(f"{field}.{key} is required")
    return config[key]


def build_allocator(
    name: str,
    cost_model: CostModel,
    config: Mapping[str, Any],
) -> object:
    """Construct a named allocator without changing the legacy default alias."""
    if name == "legacy_hard":
        return LegacyHardAllocator(cost_model)
    if name == "threshold":
        return ThresholdAllocator(cost_model)
    if name != "cost_aware_dp":
        raise ValueError(f"unsupported allocator: {name}")

    root = _mapping(config, field="config")
    research = _mapping(root.get("research", {}), field="research")
    if "allocator" not in research:
        raise ValueError("research.allocator is required")
    allocator = _mapping(research["allocator"], field="research.allocator")

    policy = AllocationPolicy(
        allowed_levels=_required(
            allocator, "allowed_levels", field="research.allocator"
        ),
        protection_utility=_required(
            allocator, "protection_utility", field="research.allocator"
        ),
        budget_scale=_required(
            allocator, "budget_scale", field="research.allocator"
        ),
        minimum_level=allocator.get("minimum_level", "none"),
    )
    return CostAwareAllocator(cost_model, policy)

