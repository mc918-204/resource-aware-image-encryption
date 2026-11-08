"""Versioned external sensitive-region labels for leakage evaluation.

The v2 artifact binds independently sourced block labels to every field of a
dataset manifest. Legacy v1 loading is intentionally explicit and is retained
only so historical learned-attack fixtures remain readable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np


EXTERNAL_LABEL_ARTIFACT_VERSION = "external_sensitive_labels_v2"
LEGACY_LABEL_ARTIFACT_VERSION = "external_sensitive_labels_v1"

_SUMMARY_FIELDS = {
    "image_count",
    "block_count",
    "sensitive_block_count",
    "nonsensitive_block_count",
}
_PUBLICATION_FORBIDDEN_LABEL_TOKENS = (
    "scorer",
    "prediction",
    "method_defined",
    "self_label",
)


@dataclass(frozen=True)
class RegionAnnotation:
    """One externally annotated rectangular privacy region."""

    image_id: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    privacy_attribute: str
    annotation_source: str


def _validate_json_value(value: Any, path: str = "value") -> None:
    """Reject Python values that do not have one unambiguous JSON encoding."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} contains a non-string JSON object key.")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains a non-JSON value of type {type(value).__name__}.")


def _canonical_json_bytes(value: Any) -> bytes:
    """Encode a value using the project's deterministic compact JSON form."""
    _validate_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _materialize_manifest_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    try:
        materialized = list(rows)
    except TypeError as exc:
        raise TypeError("Manifest rows must be an iterable of mappings.") from exc
    if not materialized:
        raise ValueError("Manifest must contain at least one row.")

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise TypeError(f"Manifest row {index} must be a mapping.")
        normalized_row = dict(row)
        _validate_json_value(normalized_row, f"manifest[{index}]")
        image_id = normalized_row.get("image_id")
        if type(image_id) is not str or not image_id.strip():
            raise ValueError(f"Manifest row {index} requires a non-blank image_id.")
        if image_id in seen_ids:
            raise ValueError(f"Manifest contains duplicate image_id={image_id!r}.")
        seen_ids.add(image_id)
        normalized.append(normalized_row)
    return normalized


def compute_manifest_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    """Return a full canonical SHA-256 binding every manifest field.

    Object keys and rows are sorted for deterministic identity. Values outside
    strict JSON, non-finite numbers, and ambiguous image identities fail closed.
    """
    normalized = _materialize_manifest_rows(rows)
    canonical_rows = sorted(normalized, key=lambda row: row["image_id"])
    return hashlib.sha256(_canonical_json_bytes(canonical_rows)).hexdigest()


def compute_legacy_manifest_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    """Return the historical ordered image-id/path-only 12-character digest."""
    normalized = _materialize_manifest_rows(rows)
    payload = [
        {
            "image_id": row.get("image_id", ""),
            "image_path": row.get("image_path", ""),
        }
        for row in normalized
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _ceil_dividend(value: int, divisor: int) -> int:
    """Return an exact integer ceiling for positive integer dimensions."""
    return (value + divisor - 1) // divisor


def _manifest_dimension(value: Any, name: str) -> int:
    """Accept positive integer geometry, including values read from CSV."""
    if type(value) is int:
        result = value
    elif type(value) is str and value.strip():
        try:
            result = int(value)
        except ValueError as exc:
            raise ValueError(f"Manifest {name} must be a positive integer.") from exc
        if str(result) != value.strip():
            raise ValueError(f"Manifest {name} must be a positive integer.")
    else:
        raise ValueError(f"Manifest {name} must be a positive integer.")
    if result <= 0:
        raise ValueError(f"Manifest {name} must be a positive integer.")
    return result


def _valid_threshold(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("overlap_threshold must be a finite number in (0, 1].")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("overlap_threshold must be a finite number in (0, 1].")
    return threshold


def _finite_coordinate(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"Region {name} must be a finite number.")
    coordinate = float(value)
    if not math.isfinite(coordinate):
        raise ValueError(f"Region {name} must be a finite number.")
    return coordinate


def build_block_labels(
    width: int,
    height: int,
    block_size: int,
    regions: Iterable[RegionAnnotation],
    overlap_threshold: float,
    *,
    target_image_id: str | None = None,
) -> np.ndarray:
    """Map rectangles to padded, row-major blocks using full-block overlap.

    Regions are clipped to the original image boundary. The denominator stays
    ``block_size ** 2`` for edge blocks because the protection pipeline pads
    those blocks to full size before splitting.
    """
    image_width = _positive_integer(width, "width")
    image_height = _positive_integer(height, "height")
    size = _positive_integer(block_size, "block_size")
    threshold = _valid_threshold(overlap_threshold)
    if target_image_id is not None and (
        type(target_image_id) is not str or not target_image_id.strip()
    ):
        raise ValueError("target_image_id must be a non-blank string when provided.")
    region_list = list(regions)

    validated_regions: list[tuple[float, float, float, float]] = []
    image_ids: set[str] = set()
    for index, region in enumerate(region_list):
        if not isinstance(region, RegionAnnotation):
            raise TypeError(f"regions[{index}] must be a RegionAnnotation.")
        if type(region.image_id) is not str or not region.image_id.strip():
            raise ValueError(f"regions[{index}] requires a non-blank image_id.")
        if target_image_id is not None and region.image_id != target_image_id:
            raise ValueError(
                f"Region image_id={region.image_id!r} does not match "
                f"target_image_id={target_image_id!r}."
            )
        image_ids.add(region.image_id)
        x_min = _finite_coordinate(region.x_min, "x_min")
        y_min = _finite_coordinate(region.y_min, "y_min")
        x_max = _finite_coordinate(region.x_max, "x_max")
        y_max = _finite_coordinate(region.y_max, "y_max")
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"regions[{index}] must have positive area.")
        validated_regions.append((x_min, y_min, x_max, y_max))
    if len(image_ids) > 1:
        raise ValueError("All regions in one block-label mapping must share one image_id.")

    row_count = _ceil_dividend(image_height, size)
    column_count = _ceil_dividend(image_width, size)
    labels = np.zeros(row_count * column_count, dtype=np.bool_)
    full_block_area = float(size * size)

    for x_min, y_min, x_max, y_max in validated_regions:
        clipped_x_min = max(0.0, min(float(image_width), x_min))
        clipped_y_min = max(0.0, min(float(image_height), y_min))
        clipped_x_max = max(0.0, min(float(image_width), x_max))
        clipped_y_max = max(0.0, min(float(image_height), y_max))
        if clipped_x_max <= clipped_x_min or clipped_y_max <= clipped_y_min:
            continue
        for row in range(row_count):
            block_y_min = float(row * size)
            block_y_max = block_y_min + size
            for column in range(column_count):
                index = row * column_count + column
                if labels[index]:
                    continue
                block_x_min = float(column * size)
                block_x_max = block_x_min + size
                intersection_width = max(
                    0.0,
                    min(clipped_x_max, block_x_max) - max(clipped_x_min, block_x_min),
                )
                intersection_height = max(
                    0.0,
                    min(clipped_y_max, block_y_max) - max(clipped_y_min, block_y_min),
                )
                overlap = intersection_width * intersection_height / full_block_area
                if overlap >= threshold:
                    labels[index] = True
    return labels


class _FrozenBoolSequence(tuple[bool, ...]):
    """Immutable boolean sequence that compares naturally with lists."""

    def __new__(cls, values: Iterable[bool]) -> "_FrozenBoolSequence":
        return tuple.__new__(cls, values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes, bytearray)):
            return tuple(self) == tuple(other)
        return tuple.__eq__(self, other)  # type: ignore[return-value]

    __hash__ = tuple.__hash__


def _canonicalize_labels(
    labels: Any,
    *,
    require_lists: bool = False,
) -> dict[str, list[bool]]:
    if not isinstance(labels, Mapping):
        raise TypeError("labels must be a mapping from image_id to boolean lists.")
    canonical: dict[str, list[bool]] = {}
    for image_id, values in labels.items():
        if type(image_id) is not str or not image_id.strip():
            raise ValueError("Every labels key must be a non-blank image_id string.")
        if require_lists and type(values) is not list:
            raise TypeError(f"Labels for image_id={image_id} must be a list.")
        if not require_lists and (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes, bytearray))
        ):
            raise TypeError(f"Labels for image_id={image_id} must be a sequence.")
        if any(type(value) is not bool for value in values):
            raise TypeError(f"Labels for image_id={image_id} must contain real booleans.")
        canonical[image_id] = list(values)
    return {image_id: canonical[image_id] for image_id in sorted(canonical)}


def _freeze_labels(labels: Any) -> Mapping[str, Sequence[bool]]:
    canonical = _canonicalize_labels(labels)
    return MappingProxyType(
        {
            image_id: _FrozenBoolSequence(values)
            for image_id, values in canonical.items()
        }
    )


def _label_digest(labels: Mapping[str, Sequence[bool]]) -> str:
    plain_labels = {
        image_id: list(values) for image_id, values in labels.items()
    }
    return hashlib.sha256(_canonical_json_bytes(plain_labels)).hexdigest()


def _label_summary(labels: Mapping[str, Sequence[bool]]) -> dict[str, int]:
    block_count = sum(len(values) for values in labels.values())
    sensitive_count = sum(sum(values) for values in labels.values())
    return {
        "image_count": len(labels),
        "block_count": block_count,
        "sensitive_block_count": sensitive_count,
        "nonsensitive_block_count": block_count - sensitive_count,
    }


def _nonempty_text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


@dataclass(frozen=True)
class ExternalLabelArtifact:
    """A manifest-bound external block-label artifact."""

    artifact_version: str
    manifest_digest: str
    block_size: int
    overlap_threshold: float
    label_source: str
    source_dataset: str
    source_version: str
    license_note: str
    mapping_version: str
    labels: Mapping[str, Sequence[bool]]
    label_digest: str
    summary: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze nested labels and summary state at construction time."""
        object.__setattr__(self, "labels", _freeze_labels(self.labels))
        if not isinstance(self.summary, Mapping):
            raise TypeError("summary must be a mapping.")
        summary = dict(self.summary)
        _validate_json_value(summary, "summary")
        object.__setattr__(self, "summary", MappingProxyType(summary))

    @classmethod
    def from_labels(
        cls,
        *,
        manifest_rows: Iterable[Mapping[str, Any]],
        block_size: int,
        overlap_threshold: float,
        label_source: str,
        source_dataset: str,
        source_version: str,
        license_note: str,
        labels: Mapping[str, list[bool]],
        mapping_version: str = "region_overlap_v1",
    ) -> "ExternalLabelArtifact":
        """Build and validate a canonical v2 artifact from explicit labels."""
        rows = _materialize_manifest_rows(manifest_rows)
        size = _positive_integer(block_size, "block_size")
        threshold = _valid_threshold(overlap_threshold)
        canonical_labels = _canonicalize_labels(labels, require_lists=True)
        artifact = cls(
            artifact_version=EXTERNAL_LABEL_ARTIFACT_VERSION,
            manifest_digest=compute_manifest_digest(rows),
            block_size=size,
            overlap_threshold=threshold,
            label_source=label_source,
            source_dataset=source_dataset,
            source_version=source_version,
            license_note=license_note,
            mapping_version=mapping_version,
            labels=canonical_labels,
            label_digest=_label_digest(canonical_labels),
            summary=_label_summary(canonical_labels),
        )
        artifact.validate(rows)
        return artifact

    def validate(
        self,
        manifest_rows: Iterable[Mapping[str, Any]],
        require_publication_ready: bool = False,
    ) -> None:
        """Validate identity, provenance, geometry, labels, and integrity."""
        rows = _materialize_manifest_rows(manifest_rows)
        if self.artifact_version == LEGACY_LABEL_ARTIFACT_VERSION:
            if require_publication_ready:
                raise ValueError("Publication-ready external labels require the v2 schema.")
            self._validate_legacy(rows)
            return
        if self.artifact_version != EXTERNAL_LABEL_ARTIFACT_VERSION:
            raise ValueError(f"Unsupported external label artifact_version={self.artifact_version!r}.")

        expected_manifest_digest = compute_manifest_digest(rows)
        if self.manifest_digest != expected_manifest_digest:
            raise ValueError(
                "External label manifest_digest mismatch: "
                f"expected {expected_manifest_digest}, got {self.manifest_digest}."
            )
        size = _positive_integer(self.block_size, "block_size")
        _valid_threshold(self.overlap_threshold)
        _nonempty_text(self.label_source, "label_source")
        _nonempty_text(self.source_dataset, "source_dataset")
        _nonempty_text(self.source_version, "source_version")
        _nonempty_text(self.license_note, "license_note")
        _nonempty_text(self.mapping_version, "mapping_version")
        if require_publication_ready:
            normalized_source = re.sub(
                r"[^a-z0-9]+", "_", self.label_source.strip().lower()
            ).strip("_")
            if any(token in normalized_source for token in _PUBLICATION_FORBIDDEN_LABEL_TOKENS):
                raise ValueError(
                    "Publication-ready label_source must be independent of scorers, "
                    "predictions, and evaluated methods."
                )

        canonical_labels = _canonicalize_labels(self.labels)
        manifest_ids = {row["image_id"] for row in rows}
        label_ids = set(canonical_labels)
        if label_ids != manifest_ids:
            missing = sorted(manifest_ids - label_ids)
            unknown = sorted(label_ids - manifest_ids)
            raise ValueError(
                f"External label image IDs must exactly match the manifest; "
                f"missing={missing}, unknown={unknown}."
            )
        for row in rows:
            image_id = row["image_id"]
            if "width" not in row or "height" not in row:
                raise ValueError(f"Manifest image_id={image_id} requires width and height.")
            width = _manifest_dimension(row["width"], "width")
            height = _manifest_dimension(row["height"], "height")
            expected_count = _ceil_dividend(width, size) * _ceil_dividend(height, size)
            actual_count = len(canonical_labels[image_id])
            if actual_count != expected_count:
                raise ValueError(
                    f"External label block count mismatch for image_id={image_id}: "
                    f"expected {expected_count}, got {actual_count}."
                )

        expected_label_digest = _label_digest(canonical_labels)
        if self.label_digest != expected_label_digest:
            raise ValueError(
                "External label label_digest mismatch: "
                f"expected {expected_label_digest}, got {self.label_digest}."
            )
        expected_summary = _label_summary(canonical_labels)
        _validate_summary(self.summary, expected_summary)

    def _validate_legacy(self, rows: list[dict[str, Any]]) -> None:
        expected_digest = compute_legacy_manifest_digest(rows)
        if self.manifest_digest != expected_digest:
            raise ValueError("Legacy external label manifest_digest mismatch.")
        canonical_labels = _canonicalize_labels(self.labels)
        if set(canonical_labels) != {row["image_id"] for row in rows}:
            raise ValueError("Legacy external label image IDs must exactly match the manifest.")
        if self.label_digest != _label_digest(canonical_labels):
            raise ValueError("Legacy external label label_digest mismatch.")
        if self.summary.get("evidence_status") != "PARTIAL":
            raise ValueError("Legacy label evidence_status cannot exceed PARTIAL.")
        if self.summary.get("reason") != "legacy_label_schema":
            raise ValueError("Legacy label summary requires reason=legacy_label_schema.")

    def save(self, path: str | Path) -> Path:
        """Write canonical JSON bytes followed by one newline."""
        self._validate_internal_integrity()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_canonical_json_bytes(_artifact_payload(self)) + b"\n")
        return target

    def _validate_internal_integrity(self) -> None:
        """Verify fields that can be authenticated without the source manifest."""
        canonical_labels = _canonicalize_labels(self.labels)
        expected_digest = _label_digest(canonical_labels)
        if self.label_digest != expected_digest:
            raise ValueError(
                "External label label_digest mismatch: "
                f"expected {expected_digest}, got {self.label_digest}."
            )
        if self.artifact_version == EXTERNAL_LABEL_ARTIFACT_VERSION:
            _validate_summary(self.summary, _label_summary(canonical_labels))
        elif self.artifact_version == LEGACY_LABEL_ARTIFACT_VERSION:
            if self.summary.get("evidence_status") != "PARTIAL":
                raise ValueError("Legacy label evidence_status cannot exceed PARTIAL.")
            if self.summary.get("reason") != "legacy_label_schema":
                raise ValueError("Legacy label summary requires reason=legacy_label_schema.")
        else:
            raise ValueError(
                f"Unsupported external label artifact_version={self.artifact_version!r}."
            )


def _validate_summary(summary: Any, expected: Mapping[str, int]) -> None:
    if not isinstance(summary, Mapping) or set(summary) != _SUMMARY_FIELDS:
        raise ValueError(
            "External label summary must contain exactly image_count, block_count, "
            "sensitive_block_count, and nonsensitive_block_count."
        )
    if any(type(summary[field]) is not int or summary[field] < 0 for field in _SUMMARY_FIELDS):
        raise ValueError("External label summary values must be non-negative integers.")
    if summary != dict(expected):
        raise ValueError(f"External label summary mismatch: expected {dict(expected)}.")


def _artifact_payload(artifact: ExternalLabelArtifact) -> dict[str, Any]:
    payload = {field.name: getattr(artifact, field.name) for field in fields(artifact)}
    payload["labels"] = {
        image_id: list(values) for image_id, values in artifact.labels.items()
    }
    payload["summary"] = dict(artifact.summary)
    if artifact.artifact_version == LEGACY_LABEL_ARTIFACT_VERSION:
        return {
            "artifact_version": artifact.artifact_version,
            "manifest_digest": artifact.manifest_digest,
            "label_source": artifact.label_source,
            "labels": payload["labels"],
        }
    return payload


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value} is not permitted.")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}.")
        result[key] = value
    return result


def _parse_json_object(raw_bytes: bytes) -> dict[str, Any]:
    try:
        text = raw_bytes.decode("utf-8")
        payload = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("External label artifact is not valid UTF-8 JSON.") from exc
    if type(payload) is not dict:
        raise ValueError("External label artifact root must be a JSON object.")
    _validate_json_value(payload)
    return payload


def _strict_v2_artifact(payload: dict[str, Any]) -> ExternalLabelArtifact:
    expected_fields = {field.name for field in fields(ExternalLabelArtifact)}
    actual_fields = set(payload)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise ValueError(f"Invalid v2 external label schema; missing={missing}, extra={extra}.")
    if payload["artifact_version"] != EXTERNAL_LABEL_ARTIFACT_VERSION:
        raise ValueError("Invalid v2 external label artifact_version.")
    if type(payload["manifest_digest"]) is not str:
        raise TypeError("manifest_digest must be a string.")
    _positive_integer(payload["block_size"], "block_size")
    threshold = _valid_threshold(payload["overlap_threshold"])
    for name in (
        "label_source",
        "source_dataset",
        "source_version",
        "license_note",
        "mapping_version",
        "label_digest",
    ):
        if type(payload[name]) is not str:
            raise TypeError(f"{name} must be a string.")
    canonical_labels = _canonicalize_labels(payload["labels"], require_lists=True)
    if type(payload["summary"]) is not dict:
        raise TypeError("summary must be a JSON object.")
    return ExternalLabelArtifact(
        artifact_version=payload["artifact_version"],
        manifest_digest=payload["manifest_digest"],
        block_size=payload["block_size"],
        overlap_threshold=threshold,
        label_source=payload["label_source"],
        source_dataset=payload["source_dataset"],
        source_version=payload["source_version"],
        license_note=payload["license_note"],
        mapping_version=payload["mapping_version"],
        labels=canonical_labels,
        label_digest=payload["label_digest"],
        summary=dict(payload["summary"]),
    )


def _load_legacy_artifact(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
) -> ExternalLabelArtifact:
    expected_fields = {"artifact_version", "manifest_digest", "label_source", "labels"}
    if set(payload) != expected_fields:
        raise ValueError("Invalid legacy external label schema.")
    if payload["manifest_digest"] != compute_legacy_manifest_digest(rows):
        raise ValueError("Legacy external label manifest_digest mismatch.")
    label_source = _nonempty_text(payload["label_source"], "label_source")
    if label_source == "unavailable":
        raise ValueError("Legacy external label artifact requires a concrete label_source.")
    raw_labels = payload["labels"]
    if type(raw_labels) is not dict:
        raise TypeError("Legacy labels must be a JSON object.")
    manifest_ids = {row["image_id"] for row in rows}
    if set(raw_labels) != manifest_ids:
        raise ValueError("Legacy external label image IDs must exactly match the manifest.")

    labels: dict[str, list[bool]] = {}
    for image_id in sorted(raw_labels):
        values = raw_labels[image_id]
        if type(values) is not list:
            raise TypeError(f"Legacy labels for image_id={image_id} must be a list.")
        if any(type(value) not in (bool, int) or value not in (True, False, 0, 1) for value in values):
            raise TypeError(f"Legacy labels for image_id={image_id} must be boolean or 0/1.")
        labels[image_id] = [bool(value) for value in values]

    summary: dict[str, object] = {
        **_label_summary(labels),
        "evidence_status": "PARTIAL",
        "reason": "legacy_label_schema",
    }
    artifact = ExternalLabelArtifact(
        artifact_version=LEGACY_LABEL_ARTIFACT_VERSION,
        manifest_digest=payload["manifest_digest"],
        block_size=0,
        overlap_threshold=0.0,
        label_source=label_source,
        source_dataset="legacy_schema_unspecified",
        source_version=LEGACY_LABEL_ARTIFACT_VERSION,
        license_note="legacy_schema_unspecified",
        mapping_version="legacy_schema_unspecified",
        labels=labels,
        label_digest=_label_digest(labels),
        summary=summary,
    )
    artifact.validate(rows)
    return artifact


def load_external_label_artifact(
    path: str | Path,
    manifest_rows: Iterable[Mapping[str, Any]],
    allow_legacy_v1: bool = False,
) -> ExternalLabelArtifact:
    """Load a strict v2 artifact or an explicitly permitted historical v1 artifact."""
    if type(allow_legacy_v1) is not bool:
        raise TypeError("allow_legacy_v1 must be a boolean.")
    rows = _materialize_manifest_rows(manifest_rows)
    payload = _parse_json_object(Path(path).read_bytes())
    version = payload.get("artifact_version")
    if version == LEGACY_LABEL_ARTIFACT_VERSION:
        if allow_legacy_v1 is not True:
            raise ValueError("Legacy v1 labels require allow_legacy_v1=True.")
        return _load_legacy_artifact(payload, rows)
    if version != EXTERNAL_LABEL_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported external label artifact_version={version!r}.")
    artifact = _strict_v2_artifact(payload)
    artifact.validate(rows, require_publication_ready=True)
    return artifact


__all__ = [
    "EXTERNAL_LABEL_ARTIFACT_VERSION",
    "LEGACY_LABEL_ARTIFACT_VERSION",
    "ExternalLabelArtifact",
    "RegionAnnotation",
    "build_block_labels",
    "compute_legacy_manifest_digest",
    "compute_manifest_digest",
    "load_external_label_artifact",
]
