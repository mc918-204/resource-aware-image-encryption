"""Pure domain contracts for the Open Images privacy benchmark subset."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any


TARGET_CLASS_MAP: Mapping[str, str] = MappingProxyType(
    {
        "Human face": "face",
        "Vehicle registration plate": "license_plate",
    }
)
OFFICIAL_SPLITS = ("train", "validation", "test")

_SOURCE_ROW_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_json_bytes(value: object) -> bytes:
    """Encode a value using deterministic compact JSON."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def source_row_digest(row: Mapping[str, str]) -> str:
    """Return a canonical SHA-256 digest for one official source row."""
    if not isinstance(row, Mapping):
        raise TypeError("row must be a mapping")
    validated: dict[str, str] = {}
    for index, (key, value) in enumerate(row.items()):
        if type(key) is not str:
            raise TypeError(f"row key at index {index} must have exact type str")
        if type(value) is not str:
            raise TypeError(f"row[{key!r}] must have exact type str")
        validated[key] = value
    return hashlib.sha256(_canonical_json_bytes(validated)).hexdigest()


def _normalized_coordinate(value: Any, name: str) -> float:
    """Validate one normalized coordinate without accepting booleans."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Open Images coordinate {name} must be finite in [0, 1]")
    coordinate = float(value)
    if not 0.0 <= coordinate <= 1.0:
        raise ValueError(f"Open Images coordinate {name} must be finite in [0, 1]")
    return coordinate


def _positive_geometry(value: Any, name: str) -> int:
    """Require exact positive integer image geometry."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class OpenImagesDetection:
    """One normalized Open Images detection with source-row provenance."""

    label: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    is_occluded: bool
    is_truncated: bool
    is_group_of: bool
    is_depiction: bool
    is_inside: bool
    source_row_digest: str

    def __post_init__(self) -> None:
        """Validate normalized geometry, flags, and source identity."""
        if type(self.label) is not str or not self.label.strip():
            raise ValueError("detection label must be non-blank")

        x_min = _normalized_coordinate(self.x_min, "x_min")
        x_max = _normalized_coordinate(self.x_max, "x_max")
        y_min = _normalized_coordinate(self.y_min, "y_min")
        y_max = _normalized_coordinate(self.y_max, "y_max")
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Open Images detection must have positive area")

        flags = (
            self.is_occluded,
            self.is_truncated,
            self.is_group_of,
            self.is_depiction,
            self.is_inside,
        )
        if any(type(flag) is not bool for flag in flags):
            raise TypeError("Open Images flags must be booleans")
        if (
            type(self.source_row_digest) is not str
            or _SOURCE_ROW_DIGEST_PATTERN.fullmatch(self.source_row_digest) is None
        ):
            raise ValueError("source_row_digest must be a lowercase SHA-256 digest")
        object.__setattr__(self, "x_min", x_min)
        object.__setattr__(self, "x_max", x_max)
        object.__setattr__(self, "y_min", y_min)
        object.__setattr__(self, "y_max", y_max)

    @property
    def normalized_area(self) -> Fraction:
        """Return context-independent exact area at source decimal precision."""
        x_min = Fraction(str(self.x_min))
        x_max = Fraction(str(self.x_max))
        y_min = Fraction(str(self.y_min))
        y_max = Fraction(str(self.y_max))
        return (x_max - x_min) * (y_max - y_min)

    @property
    def is_qualifying(self) -> bool:
        """Return whether the detection qualifies for privacy selection."""
        return (
            self.label in TARGET_CLASS_MAP
            and not self.is_group_of
            and not self.is_depiction
        )


@dataclass(frozen=True)
class OpenImagesCandidate:
    """One validated image and its Open Images detections."""

    openimages_id: str
    split: str
    image_path: str
    width: int
    height: int
    detections: tuple[OpenImagesDetection, ...]

    def __post_init__(self) -> None:
        """Validate identity, official split, geometry, and detections."""
        if type(self.openimages_id) is not str or not self.openimages_id.strip():
            raise ValueError("openimages_id must be non-blank")
        if self.openimages_id != self.openimages_id.strip():
            raise ValueError(
                "openimages_id must not contain leading or trailing whitespace"
            )
        if self.split not in OFFICIAL_SPLITS:
            raise ValueError(f"unsupported Open Images split: {self.split}")
        if type(self.image_path) is not str or not self.image_path.strip():
            raise ValueError("image_path must be non-blank")
        if self.image_path != self.image_path.strip():
            raise ValueError(
                "image_path must not contain leading or trailing whitespace"
            )
        _positive_geometry(self.width, "width")
        _positive_geometry(self.height, "height")
        if type(self.detections) is not tuple:
            raise TypeError("detections must be a tuple of OpenImagesDetection values")
        if any(
            not isinstance(detection, OpenImagesDetection)
            for detection in self.detections
        ):
            raise TypeError("detections must contain only OpenImagesDetection values")


def normalized_box_to_pixels(
    detection: OpenImagesDetection,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Convert a normalized detection box to floating-point pixel edges."""
    if not isinstance(detection, OpenImagesDetection):
        raise TypeError("detection must be an OpenImagesDetection")
    try:
        image_width = _positive_geometry(width, "width")
        image_height = _positive_geometry(height, "height")
    except ValueError as exc:
        raise ValueError("image geometry must be positive integers") from exc
    return (
        detection.x_min * image_width,
        detection.y_min * image_height,
        detection.x_max * image_width,
        detection.y_max * image_height,
    )


def _selection_key(image_id: str, split: str, label: str, seed: int) -> str:
    """Return the frozen Open Images V7 seeded selection key."""
    payload = f"Open Images V7|{split}|{label}|{image_id}|{seed}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assigned_label(candidate: OpenImagesCandidate, seed: int) -> str | None:
    """Assign a candidate to its largest qualifying target-class area."""
    areas = {
        label: sum(
            (
                detection.normalized_area
                for detection in candidate.detections
                if detection.is_qualifying and detection.label == label
            ),
            start=Fraction(0),
        )
        for label in TARGET_CLASS_MAP
    }
    qualifying = {label: area for label, area in areas.items() if area > Fraction(0)}
    if not qualifying:
        return None
    largest_area = max(qualifying.values())
    tied_labels = (
        label for label, area in qualifying.items() if area == largest_area
    )
    return min(
        tied_labels,
        key=lambda label: (
            _selection_key(candidate.openimages_id, candidate.split, label, seed),
            label,
        ),
    )


def _validated_quotas(
    quotas: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    """Materialize and validate requested split/class quota cells."""
    if not isinstance(quotas, Mapping):
        raise TypeError("quotas must be a mapping")
    validated: dict[str, dict[str, int]] = {}
    for split, class_quotas in quotas.items():
        if split not in OFFICIAL_SPLITS:
            raise ValueError(f"unsupported quota split: {split}")
        if not isinstance(class_quotas, Mapping):
            raise TypeError(f"quotas for {split} must be a mapping")
        validated[split] = {}
        for label, count in class_quotas.items():
            if label not in TARGET_CLASS_MAP or type(count) is not int or count <= 0:
                raise ValueError(f"invalid quota for {split}/{label}")
            validated[split][label] = count
    return validated


def select_balanced_candidates(
    candidates: Iterable[OpenImagesCandidate],
    quotas: Mapping[str, Mapping[str, int]],
    *,
    seed: int,
) -> tuple[OpenImagesCandidate, ...]:
    """Select exact deterministic split/class quotas without borrowing."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    requested = _validated_quotas(quotas)
    try:
        iterator = iter(candidates)
    except TypeError as exc:
        raise TypeError("candidates must be an iterable") from exc
    materialized = list(iterator)
    if any(
        not isinstance(candidate, OpenImagesCandidate)
        for candidate in materialized
    ):
        raise TypeError("candidates must contain only OpenImagesCandidate values")

    seen_ids: set[str] = set()
    for candidate in materialized:
        if candidate.openimages_id in seen_ids:
            raise ValueError(
                "candidate inventory contains duplicate Open Images ID "
                f"{candidate.openimages_id!r}"
            )
        seen_ids.add(candidate.openimages_id)

    buckets: dict[tuple[str, str], list[OpenImagesCandidate]] = {}
    for candidate in materialized:
        label = _assigned_label(candidate, seed)
        if label is not None:
            buckets.setdefault((candidate.split, label), []).append(candidate)

    selected: list[OpenImagesCandidate] = []
    for split, class_quotas in requested.items():
        for label, count in class_quotas.items():
            ordered = sorted(
                buckets.get((split, label), ()),
                key=lambda item: (
                    _selection_key(item.openimages_id, split, label, seed),
                    item.openimages_id,
                ),
            )
            if len(ordered) < count:
                raise ValueError(
                    f"quota unavailable for {split}/{label}: "
                    f"available {len(ordered)}, requested {count}"
                )
            selected.extend(ordered[:count])

    return tuple(
        sorted(selected, key=lambda item: (item.split, item.openimages_id))
    )
