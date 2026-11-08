"""Deterministic conversion from object detections to block risk signals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Sequence

import numpy as np


_COMPONENT_NAMES = ("object", "boundary", "context", "uncertainty")
MAX_IMAGE_DIMENSION = 100_000
MAX_BLOCK_SIZE = 4_096
MAX_MARGIN_PX = 100_000
MAX_BLOCK_COUNT = 4_000_000


def _finite_number(value: object, name: str) -> float:
    """Return one finite real number while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: object, name: str) -> int:
    """Validate a strictly positive built-in integer geometry value."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    """Validate a non-negative built-in integer margin."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class Detection:
    """One validated detector box in source-image pixel coordinates.

    Boxes must have positive source area. A valid box that becomes empty only
    after clipping to the image is ignored by :func:`detections_to_signals`.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str
    confidence: float

    def __post_init__(self) -> None:
        """Normalize numeric values and reject ambiguous detections."""
        coordinates = {
            name: _finite_number(getattr(self, name), name)
            for name in ("x_min", "y_min", "x_max", "y_max")
        }
        if coordinates["x_max"] <= coordinates["x_min"] or coordinates[
            "y_max"
        ] <= coordinates["y_min"]:
            raise ValueError("detection box must have positive area")
        if type(self.label) is not str or not self.label.strip():
            raise ValueError("detection label must be a non-blank string")
        confidence = _finite_number(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("detection confidence must be in [0, 1]")
        for name, value in coordinates.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "confidence", confidence)


def _validated_component(value: np.ndarray, name: str) -> np.ndarray:
    """Return a detached, read-only float32 signal component."""
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"spatial signal {name} cannot use object dtype")
    if array.ndim != 1:
        raise ValueError(f"spatial signal {name} must be one-dimensional")
    try:
        numeric = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"spatial signal {name} must be numeric") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"spatial signal {name} must be finite")
    if np.any(numeric < 0.0) or np.any(numeric > 1.0):
        raise ValueError(f"spatial signal {name} must be in [0, 1]")
    # A bytes-backed array cannot have write access re-enabled by callers,
    # unlike an owning ndarray whose WRITEABLE flag can be toggled back on.
    detached = np.frombuffer(
        np.ascontiguousarray(numeric, dtype=np.float32).tobytes(),
        dtype=np.float32,
    )
    return detached


@dataclass(frozen=True)
class SpatialSignals:
    """Four aligned, immutable row-major block signals in ``[0, 1]``."""

    object: np.ndarray
    boundary: np.ndarray
    context: np.ndarray
    uncertainty: np.ndarray

    def __post_init__(self) -> None:
        """Validate, detach, and freeze every public component array."""
        converted = {
            name: _validated_component(getattr(self, name), name)
            for name in _COMPONENT_NAMES
        }
        lengths = {name: len(value) for name, value in converted.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"spatial signal length mismatch: {lengths}")
        for name, value in converted.items():
            object.__setattr__(self, name, value)


def _intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Return the exact intersection area of two axis-aligned rectangles."""
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _clip_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Clip a rectangle to the closed source-image extent."""
    return (
        min(float(width), max(0.0, box[0])),
        min(float(height), max(0.0, box[1])),
        min(float(width), max(0.0, box[2])),
        min(float(height), max(0.0, box[3])),
    )


def _contract_box(
    box: tuple[float, float, float, float],
    amount: int,
) -> tuple[float, float, float, float]:
    """Contract each side, collapsing narrow dimensions at their midpoint."""
    x_mid = (box[0] + box[2]) / 2.0
    y_mid = (box[1] + box[3]) / 2.0
    return (
        min(box[0] + amount, x_mid),
        min(box[1] + amount, y_mid),
        max(box[2] - amount, x_mid),
        max(box[3] - amount, y_mid),
    )


def _expand_box(
    box: tuple[float, float, float, float],
    amount: int,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Expand a rectangle on each side and clip it to the image."""
    return _clip_box(
        (
            box[0] - amount,
            box[1] - amount,
            box[2] + amount,
            box[3] + amount,
        ),
        width,
        height,
    )


def _validated_priors(label_priors: Mapping[str, float]) -> dict[str, float]:
    """Copy a finite label-prior mapping with values in ``[0, 1]``."""
    if not isinstance(label_priors, Mapping):
        raise ValueError("label_priors must be a mapping")
    result: dict[str, float] = {}
    for label, prior in label_priors.items():
        if type(label) is not str or not label.strip():
            raise ValueError("label prior keys must be non-blank strings")
        number = _finite_number(prior, f"label prior {label!r}")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"label prior {label!r} must be in [0, 1]")
        normalized = label.strip()
        if normalized in result:
            raise ValueError(f"duplicate normalized label prior {normalized!r}")
        result[normalized] = number
    return result


def detections_to_signals(
    *,
    detections: Sequence[Detection],
    width: int,
    height: int,
    block_size: int,
    label_priors: Mapping[str, float],
    context_margin_px: int,
    boundary_width_px: int,
) -> SpatialSignals:
    """Convert frozen detections into row-major block signals in ``[0, 1]``.

    All overlap fractions retain ``block_size ** 2`` as the denominator,
    including blocks padded beyond the source image. A valid source box that
    clips to zero image area is ignored. Candidates are clipped and combined
    by component-wise maximum across detections.
    """
    checked_width = _positive_integer(width, "width")
    checked_height = _positive_integer(height, "height")
    checked_block_size = _positive_integer(block_size, "block_size")
    checked_context_margin = _nonnegative_integer(
        context_margin_px,
        "context_margin_px",
    )
    checked_boundary_width = _nonnegative_integer(
        boundary_width_px,
        "boundary_width_px",
    )
    if checked_width > MAX_IMAGE_DIMENSION:
        raise ValueError(f"width must not exceed {MAX_IMAGE_DIMENSION}")
    if checked_height > MAX_IMAGE_DIMENSION:
        raise ValueError(f"height must not exceed {MAX_IMAGE_DIMENSION}")
    if checked_block_size > MAX_BLOCK_SIZE:
        raise ValueError(f"block_size must not exceed {MAX_BLOCK_SIZE}")
    if checked_context_margin > MAX_MARGIN_PX:
        raise ValueError(f"context_margin_px must not exceed {MAX_MARGIN_PX}")
    if checked_boundary_width > MAX_MARGIN_PX:
        raise ValueError(f"boundary_width_px must not exceed {MAX_MARGIN_PX}")
    priors = _validated_priors(label_priors)
    if isinstance(detections, (str, bytes)) or not isinstance(detections, Sequence):
        raise ValueError("detections must be a sequence of Detection values")
    copied_detections = tuple(detections)
    if any(not isinstance(item, Detection) for item in copied_detections):
        raise ValueError("detections must contain only Detection values")

    columns = (checked_width + checked_block_size - 1) // checked_block_size
    rows = (checked_height + checked_block_size - 1) // checked_block_size
    if rows * columns > MAX_BLOCK_COUNT:
        raise ValueError(f"block_count must not exceed {MAX_BLOCK_COUNT}")
    components = {
        name: np.zeros(rows * columns, dtype=np.float32)
        for name in _COMPONENT_NAMES
    }
    denominator = float(checked_block_size * checked_block_size)

    candidates: list[
        tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float, float],
            float,
            float,
        ]
    ] = []
    for detection in copied_detections:
        raw_box = (
            detection.x_min,
            detection.y_min,
            detection.x_max,
            detection.y_max,
        )
        original = _clip_box(
            raw_box,
            checked_width,
            checked_height,
        )
        if original[2] <= original[0] or original[3] <= original[1]:
            continue
        prior = priors.get(detection.label, 0.0)
        if prior == 0.0:
            continue
        inner = _clip_box(
            _contract_box(raw_box, checked_boundary_width),
            checked_width,
            checked_height,
        )
        outer = _expand_box(
            raw_box,
            checked_boundary_width,
            checked_width,
            checked_height,
        )
        expanded = _expand_box(
            raw_box,
            checked_context_margin,
            checked_width,
            checked_height,
        )
        candidates.append(
            (original, inner, outer, expanded, detection.confidence, prior)
        )

    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            block = (
                float(column * checked_block_size),
                float(row * checked_block_size),
                float((column + 1) * checked_block_size),
                float((row + 1) * checked_block_size),
            )
            for original, inner, outer, expanded, confidence, prior in candidates:
                original_area = _intersection_area(block, original)
                inner_area = _intersection_area(block, inner)
                outer_area = _intersection_area(block, outer)
                expanded_area = _intersection_area(block, expanded)
                values = {
                    "object": original_area / denominator * confidence * prior,
                    "uncertainty": (
                        original_area / denominator * (1.0 - confidence) * prior
                    ),
                    "boundary": (
                        max(0.0, outer_area - inner_area)
                        / denominator
                        * confidence
                        * prior
                    ),
                    "context": (
                        max(0.0, expanded_area - original_area)
                        / denominator
                        * confidence
                        * prior
                    ),
                }
                for name, value in values.items():
                    clipped = float(np.clip(value, 0.0, 1.0))
                    if clipped > components[name][index]:
                        components[name][index] = np.float32(clipped)

    return SpatialSignals(**components)
