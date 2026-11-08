"""Unified semantic backend type definitions.

All runner, audit, export, and summary code must use these constants
and helper functions instead of ad-hoc string comparisons.
"""

from __future__ import annotations

from typing import Final

# Canonical real backend identifiers
REAL_BACKENDS: Final[set[str]] = {
    "real",
    "dinov2",
    "dinov2_huggingface",
}

# Canonical fallback backend identifiers
FALLBACK_BACKENDS: Final[set[str]] = {
    "fallback",
    "fallback_visual_embedding",
    "mock",
    "synthetic",
}

# Normalization map: alias -> canonical name
_BACKEND_NORMALIZATION: Final[dict[str, str]] = {
    # Real backends
    "real": "dinov2_huggingface",
    "dinov2": "dinov2_huggingface",
    "dinov2_huggingface": "dinov2_huggingface",
    "huggingface_dinov2": "dinov2_huggingface",
    # Fallback backends
    "fallback": "fallback_visual_embedding",
    "fallback_visual_embedding": "fallback_visual_embedding",
    "mock": "mock",
    "synthetic": "synthetic",
    "not_applicable": "not_applicable",
    "none": "not_applicable",
}


def is_real_semantic_backend(name: str | None) -> bool:
    """Return True if the given backend name is a real semantic backend."""
    if name is None:
        return False
    normalized = _normalize_raw(name)
    return normalized in REAL_BACKENDS


def is_fallback_semantic_backend(name: str | None) -> bool:
    """Return True if the given backend name is a fallback backend."""
    if name is None:
        return False
    normalized = _normalize_raw(name)
    return normalized in FALLBACK_BACKENDS


def normalize_semantic_backend(name: str | None) -> str:
    """Normalize a backend name to its canonical form.
    
    Returns the canonical name if known, or the original name (lowercased)
    if it doesn't match any known pattern.
    """
    if name is None:
        return "unknown"
    normalized = _normalize_raw(name)
    return _BACKEND_NORMALIZATION.get(normalized, normalized.lower())


def _normalize_raw(name: str) -> str:
    """Lowercase and strip whitespace."""
    return name.strip().lower()
