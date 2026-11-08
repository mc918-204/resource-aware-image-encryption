"""Additive construction registry for baseline and research scorers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.policy_model.leakage_aware_scorer import LeakageAwareScorer
from src.policy_model.rule_based import RuleBasedScorer


def _required_mapping(
    parent: Mapping[str, Any], key: str, path: str
) -> Mapping[str, Any]:
    """Return a required nested mapping with a path-specific error."""
    if key not in parent:
        raise ValueError(f"{path} mapping is required")
    value = parent[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def build_scorer(mode: str, config: Mapping[str, Any]) -> object:
    """Construct an exact scorer mode without changing legacy aliases."""
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")

    if mode == "rule_based":
        policy = _required_mapping(config, "policy", "config.policy")
        return RuleBasedScorer(dict(policy))
    if mode == "leakage_aware":
        research = _required_mapping(config, "research", "config.research")
        scorer_config = _required_mapping(
            research, "scorer", "config.research.scorer"
        )
        return LeakageAwareScorer(scorer_config)
    raise ValueError(f"unsupported scorer mode: {mode}")
