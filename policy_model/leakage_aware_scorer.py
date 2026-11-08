"""Leakage-aware block risk signal contracts and deterministic fusion."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


SIGNAL_NAMES = (
    "rule",
    "semantic",
    "object",
    "boundary",
    "context",
    "uncertainty",
)


@dataclass(frozen=True)
class RiskSignals:
    """Raw block-level inputs used by the leakage-aware scorer."""

    rule: np.ndarray
    semantic: np.ndarray
    object: np.ndarray
    boundary: np.ndarray
    context: np.ndarray
    uncertainty: np.ndarray


@dataclass(frozen=True)
class LeakageAwareScoreResult:
    """Fused scores and per-component diagnostics for one scoring call."""

    scores: np.ndarray
    raw_components: dict[str, np.ndarray]
    normalized_components: dict[str, np.ndarray]
    normalized_weights: dict[str, float]
    uncertainty_direction: str


def build_risk_signals(
    *,
    rule_scores: np.ndarray,
    semantic_scores: np.ndarray,
    object_scores: np.ndarray,
    boundary_scores: np.ndarray,
    context_scores: np.ndarray,
    uncertainty_scores: np.ndarray,
) -> RiskSignals:
    """Convert six equally sized score arrays into owned risk-signal arrays."""
    inputs = {
        "rule": rule_scores,
        "semantic": semantic_scores,
        "object": object_scores,
        "boundary": boundary_scores,
        "context": context_scores,
        "uncertainty": uncertainty_scores,
    }
    converted: dict[str, np.ndarray] = {}
    for name, values in inputs.items():
        try:
            converted[name] = np.asarray(
                values, dtype=np.float64
            ).reshape(-1).copy()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} risk signal must be numeric") from exc

    lengths = {name: len(values) for name, values in converted.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"risk signal length mismatch: {lengths}")
    return RiskSignals(**converted)


class LeakageAwareScorer:
    """Fuse six independently normalized risk signals with fixed weights."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Validate and normalize the leakage-aware scorer configuration."""
        try:
            weights = dict(config.get("weights", {}))
        except (TypeError, ValueError) as exc:
            raise ValueError("weights must be a mapping") from exc
        if set(weights) != set(SIGNAL_NAMES):
            raise ValueError(f"weights must contain exactly {SIGNAL_NAMES}")

        numeric: dict[str, float] = {}
        for name in SIGNAL_NAMES:
            value = weights[name]
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("weights must be numeric, not boolean")
            try:
                numeric[name] = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("weights must be numeric") from exc

        if any(
            not np.isfinite(value) or value < 0.0
            for value in numeric.values()
        ):
            raise ValueError("weights must be finite and nonnegative")
        total = sum(numeric.values())
        if not np.isfinite(total):
            raise ValueError("weight sum must be finite")
        if total <= 0.0:
            raise ValueError("weight sum must be positive")

        direction = config.get("uncertainty_direction", "protect_more")
        if not isinstance(direction, str) or direction not in {
            "protect_more",
            "discount",
        }:
            raise ValueError(
                "uncertainty_direction must be protect_more or discount"
            )

        self._normalized_weights: Mapping[str, float] = MappingProxyType(
            {name: value / total for name, value in numeric.items()}
        )
        self.uncertainty_direction = direction

    @property
    def normalized_weights(self) -> Mapping[str, float]:
        """Expose the validated normalized weights as a read-only mapping."""
        return self._normalized_weights

    @staticmethod
    def _normalize(values: np.ndarray, name: str) -> np.ndarray:
        """Return a constant-safe min-max normalization of one component."""
        if values.size == 0:
            raise ValueError(f"{name} risk signal cannot be empty")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} risk signal must be finite")
        lower = float(values.min())
        upper = float(values.max())
        if upper == lower:
            return np.zeros_like(values, dtype=np.float64)
        scale = max(abs(lower), abs(upper))
        scaled_values = values / scale
        scaled_lower = lower / scale
        scaled_upper = upper / scale
        normalized = (
            (scaled_values - scaled_lower) / (scaled_upper - scaled_lower)
        )
        if not np.all(np.isfinite(normalized)):
            raise ValueError(f"{name} normalized risk signal must be finite")
        return np.clip(normalized, 0.0, 1.0)

    def fuse(self, signals: RiskSignals) -> LeakageAwareScoreResult:
        """Normalize and fuse one equally sized set of block risk signals."""
        raw: dict[str, np.ndarray] = {}
        for name in SIGNAL_NAMES:
            try:
                raw[name] = np.asarray(
                    getattr(signals, name), dtype=np.float64
                ).reshape(-1).copy()
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} risk signal must be numeric") from exc

        expected = len(raw["rule"])
        for name, values in raw.items():
            if len(values) != expected:
                raise ValueError(
                    f"{name} signal length {len(values)} != rule length {expected}"
                )
        if expected == 0:
            raise ValueError("risk signals cannot be empty")

        normalized = {
            name: self._normalize(values, name) for name, values in raw.items()
        }
        risk = np.zeros(expected, dtype=np.float64)
        for name in SIGNAL_NAMES:
            if name != "uncertainty":
                risk += self._normalized_weights[name] * normalized[name]
        uncertainty = (
            self._normalized_weights["uncertainty"]
            * normalized["uncertainty"]
        )
        if self.uncertainty_direction == "protect_more":
            scores = risk + uncertainty
        else:
            scores = risk - uncertainty
        if not np.all(np.isfinite(scores)):
            raise ValueError("fused scores must be finite")

        return LeakageAwareScoreResult(
            scores=np.clip(scores, 0.0, 1.0).astype(np.float32),
            raw_components={name: values.copy() for name, values in raw.items()},
            normalized_components={
                name: values.copy() for name, values in normalized.items()
            },
            normalized_weights=dict(self._normalized_weights),
            uncertainty_direction=self.uncertainty_direction,
        )
