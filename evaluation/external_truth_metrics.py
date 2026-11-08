"""Protection metrics evaluated against external, method-independent labels.

Why this module exists (plan item F4/C4)
----------------------------------------
The historical selective metrics in :mod:`src.evaluation.selective_metrics`
define "sensitive" as *the top quantile of the scoring method's own risk
scores*. Every method is therefore graded against a target it defines
itself, which makes cross-method comparison circular -- the project's own
``paper_numbers.json`` records ``cross_method_security_comparable: false``.

The metrics here take the sensitive mask as an **input**, sourced from
Open Images V7 official bounding boxes (``Human face``, ``Vehicle
registration plate``) already built and integrity-audited in
``results/labels/openimages_privacy_external_labels_v2.json``. Because
every method is scored against the same fixed mask, differences are
attributable to the methods.

Metric vocabulary
-----------------
``recall@strong``
    Share of externally sensitive blocks that received a level in
    ``strong_levels`` (default ``high``/``medium``, i.e. authenticated
    encryption). This is the primary dependent variable.
``exposed_raw_ratio``
    Share of externally sensitive blocks left at ``none`` -- plaintext.
    This is the operational definition of *missed protection*.
``nonsensitive_encrypted_ratio``
    Share of non-sensitive blocks that consumed strong protection, i.e.
    wasted budget (false positives).
``pixel_recall@strong``
    Area-weighted counterpart of ``recall@strong``, computed over the
    union of annotated regions so overlapping boxes are not double
    counted.
``oracle_recall@strong`` / ``oracle_gap@strong``
    A label-aware upper bound under the same strong-block count, and the
    gap to it. The oracle is **not a deployable method**; it bounds what
    any scorer could achieve at that coverage.

Scope limitation to state in the paper
--------------------------------------
The labels cover two privacy categories only. They are a *frozen
external reference*, not a universal definition of image privacy.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "DEFAULT_STRONG_LEVELS",
    "EXTERNAL_TRUTH_METRIC_KEYS",
    "compute_external_truth_metrics",
    "empty_external_truth_metrics",
    "oracle_strong_allocation",
]


DEFAULT_STRONG_LEVELS: tuple[str, ...] = ("high", "medium")
"""Levels that constitute authenticated encryption.

``low`` is excluded deliberately: it applies keystream diffusion without
AEAD and, under the V1 profile, with a publicly derivable key (finding
F1). Counting it as protection would overstate coverage.
"""

EXTERNAL_TRUTH_METRIC_KEYS: tuple[str, ...] = (
    "ext_sensitive_blocks",
    "ext_nonsensitive_blocks",
    "ext_sensitive_block_recall_strong",
    "ext_sensitive_block_recall_any",
    "ext_sensitive_exposed_raw_ratio",
    "ext_nonsensitive_encrypted_ratio",
    "ext_sensitive_pixel_recall_strong",
    "ext_oracle_recall_strong",
    "ext_oracle_gap_strong",
)


class ExternalTruthMetricError(ValueError):
    """Raised when label and allocation shapes are inconsistent."""


def _as_bool_mask(sensitive_mask: Sequence[bool] | np.ndarray) -> np.ndarray:
    """Coerce a label sequence into a 1-D boolean array."""
    mask = np.asarray(sensitive_mask, dtype=bool).reshape(-1)
    return mask


def _validate(levels: Sequence[str], mask: np.ndarray) -> None:
    """Reject length mismatches instead of silently truncating."""
    if len(levels) != mask.size:
        raise ExternalTruthMetricError(
            "protection_levels/sensitive_mask length mismatch: "
            f"levels={len(levels)}, mask={mask.size}"
        )


def oracle_strong_allocation(
    sensitive_mask: Sequence[bool] | np.ndarray,
    strong_block_count: int,
    strong_level: str = "high",
) -> list[str]:
    """Build a label-aware allocation that spends ``strong_block_count`` best.

    The oracle assigns strong protection to sensitive blocks first, then to
    arbitrary remaining blocks if budget is left over. It is a *bound*, not a
    method: it reads the ground-truth labels a deployable scorer cannot see.

    Args:
        sensitive_mask: External per-block sensitivity labels.
        strong_block_count: How many blocks may receive strong protection,
            normally the count the evaluated method actually used.
        strong_level: Level name to assign, default ``high``.

    Returns:
        A level list with ``strong_level`` on the chosen blocks, ``none``
        elsewhere.
    """
    mask = _as_bool_mask(sensitive_mask)
    budget = max(0, min(int(strong_block_count), mask.size))
    order = np.argsort(~mask, kind="stable")  # sensitive blocks first
    levels = ["none"] * mask.size
    for index in order[:budget]:
        levels[int(index)] = strong_level
    return levels


def compute_external_truth_metrics(
    *,
    protection_levels: Sequence[str],
    sensitive_mask: Sequence[bool] | np.ndarray,
    strong_levels: Iterable[str] = DEFAULT_STRONG_LEVELS,
    sensitive_pixel_counts: Sequence[float] | np.ndarray | None = None,
    label_source: str = "",
    label_digest: str = "",
) -> Dict[str, Any]:
    """Score one allocation against an external sensitive mask.

    Args:
        protection_levels: Per-block assigned level, in original block order.
        sensitive_mask: Per-block external sensitivity label, same order.
        strong_levels: Levels counted as authenticated protection.
        sensitive_pixel_counts: Optional per-block count of annotated sensitive
            pixels, used for the area-weighted recall. When ``None`` the pixel
            metric is reported as ``None`` rather than approximated.
        label_source: Provenance string recorded on the row.
        label_digest: Label artifact digest recorded on the row.

    Returns:
        A dict with every key in :data:`EXTERNAL_TRUTH_METRIC_KEYS` plus
        ``external_label_source`` and ``external_label_digest``. Ratios are
        ``None`` when their denominator is zero, never silently ``0.0``.
    """
    mask = _as_bool_mask(sensitive_mask)
    levels = list(protection_levels)
    _validate(levels, mask)

    strong = set(strong_levels)
    level_array = np.asarray(levels, dtype=object)
    strong_mask = np.array([level in strong for level in level_array], dtype=bool)
    any_protection_mask = np.array([level != "none" for level in level_array], dtype=bool)
    raw_mask = ~any_protection_mask

    sensitive_total = int(np.sum(mask))
    nonsensitive_total = int(mask.size - sensitive_total)

    def _ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator > 0 else None

    recall_strong = _ratio(int(np.sum(strong_mask & mask)), sensitive_total)
    recall_any = _ratio(int(np.sum(any_protection_mask & mask)), sensitive_total)
    exposed_raw = _ratio(int(np.sum(raw_mask & mask)), sensitive_total)
    wasted = _ratio(int(np.sum(strong_mask & ~mask)), nonsensitive_total)

    pixel_recall: float | None = None
    if sensitive_pixel_counts is not None:
        pixels = np.asarray(sensitive_pixel_counts, dtype=np.float64).reshape(-1)
        if pixels.size != mask.size:
            raise ExternalTruthMetricError(
                "sensitive_pixel_counts length mismatch: "
                f"pixels={pixels.size}, mask={mask.size}"
            )
        total_pixels = float(np.sum(pixels))
        if total_pixels > 0:
            pixel_recall = float(np.sum(pixels[strong_mask]) / total_pixels)

    # Oracle bound at the same strong-block count the method actually spent.
    strong_used = int(np.sum(strong_mask))
    oracle_levels = oracle_strong_allocation(mask, strong_used)
    oracle_strong_mask = np.array(
        [level != "none" for level in oracle_levels], dtype=bool
    )
    oracle_recall = _ratio(int(np.sum(oracle_strong_mask & mask)), sensitive_total)
    oracle_gap = (
        None
        if (oracle_recall is None or recall_strong is None)
        else float(oracle_recall - recall_strong)
    )

    return {
        "ext_sensitive_blocks": sensitive_total,
        "ext_nonsensitive_blocks": nonsensitive_total,
        "ext_sensitive_block_recall_strong": recall_strong,
        "ext_sensitive_block_recall_any": recall_any,
        "ext_sensitive_exposed_raw_ratio": exposed_raw,
        "ext_nonsensitive_encrypted_ratio": wasted,
        "ext_sensitive_pixel_recall_strong": pixel_recall,
        "ext_oracle_recall_strong": oracle_recall,
        "ext_oracle_gap_strong": oracle_gap,
        "external_label_source": label_source,
        "external_label_digest": label_digest,
    }


def empty_external_truth_metrics(reason: str = "no_external_labels") -> Dict[str, Any]:
    """Return an all-``None`` payload for images without external labels.

    Rows for corpora that carry no privacy annotations (DIV2K, Kodak, ...) must
    show ``None`` here, not zeros, so that a missing label can never be read as
    "zero sensitive content".
    """
    payload: Dict[str, Any] = {key: None for key in EXTERNAL_TRUTH_METRIC_KEYS}
    payload["external_label_source"] = reason
    payload["external_label_digest"] = ""
    return payload


def load_label_lookup(
    artifact: Any,
) -> Mapping[str, Sequence[bool]]:
    """Expose an artifact's per-image block labels as a plain mapping.

    Accepts an :class:`~src.evaluation.external_sensitive_labels.ExternalLabelArtifact`
    and returns its ``labels`` mapping (``image_id -> per-block bool``), so
    callers need not reach into artifact internals.
    """
    labels = getattr(artifact, "labels", None)
    if labels is None:
        raise ExternalTruthMetricError("artifact does not expose `labels`")
    return labels
