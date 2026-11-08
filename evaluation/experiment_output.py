"""Strict schema-aware utilities for experiment output."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.evaluation.attack_schema import ATTACK_SCHEMA
from src.evaluation.publication_schema import (
    PUBLICATION_RAW_SCHEMA,
    PUBLICATION_STATUS_VALUES,
    PublicationSchemaError,
    normalize_publication_record,
    normalize_publication_status,
)


LEGACY_STATUS_MAPPING: Dict[str, str] = {
    "failed": "DEGRADED",
    "feasible": "FULL",
    "exceeded": "DEGRADED",
    "full": "FULL",
    "partial": "PARTIAL",
    "degraded": "DEGRADED",
    "unknown": "DEGRADED",
    "success": "FULL",
    "skipped": "PARTIAL",
    "budget_exceeded": "DEGRADED",
    "runtime_error": "DEGRADED",
    "invalid_config": "DEGRADED",
    "missing_dependency": "DEGRADED",
    "invalid_baseline": "DEGRADED",
}


@dataclass(frozen=True)
class OutputSchema:
    """Schema definition for experiment records."""

    name: str
    fieldnames: Tuple[str, ...]
    required_fields: Tuple[str, ...]
    status_field: str = "status"
    allowed_statuses: Tuple[str, ...] = PUBLICATION_STATUS_VALUES


@dataclass(frozen=True)
class SchemaValidationResult:
    """Normalized schema validation result."""

    record: Dict[str, Any]
    missing_fields: Tuple[str, ...]
    unknown_fields: Tuple[str, ...]


class SchemaValidationError(ValueError):
    """Raised when an output record does not match the required schema."""


PAPER_RESULT_REQUIRED_FIELDS: Tuple[str, ...] = (
    "run_id",
    "experiment_name",
    "method_name",
    "method_family",
    "dataset",
    "image_id",
    "budget_type",
    "budget_ratio",
    "seed",
    "status",
    "error_message",
)

PAPER_RESULT_OPTIONAL_FIELDS: Tuple[str, ...] = (
    "image_path",
    "split",
    "category",
    "budget_limit",
    "budget_used",
    "budget_utilization",
    "runtime_ms",
    "encrypted_ratio",
    "protected_sensitive_ratio",
    "unprotected_sensitive_ratio",
    "high_count",
    "medium_count",
    "low_count",
    "none_count",
    "high_ratio",
    "medium_ratio",
    "low_ratio",
    "none_ratio",
    "semantic_backend",
    "model_path_resolved",
    "device_resolved",
    "fallback_reason",
    "skip_reason",
    "notes",
    "timestamp",
    "execution_status",
)

PAPER_RESULT_SCHEMA = OutputSchema(
    name="paper_result",
    fieldnames=PAPER_RESULT_REQUIRED_FIELDS + PAPER_RESULT_OPTIONAL_FIELDS,
    required_fields=PAPER_RESULT_REQUIRED_FIELDS,
)

SOTA_RESULT_SCHEMA = OutputSchema(
    name="sota_result",
    fieldnames=(
        "run_id",
        "method_name",
        "method_family",
        "dataset",
        "image_id",
        "budget_type",
        "budget_ratio",
        "runtime_ms",
        "encrypted_ratio",
        "protected_sensitive_ratio",
        "unprotected_sensitive_ratio",
        "entropy",
        "npcr",
        "uaci",
        "ssim",
        "psnr",
        "semantic_exposure_rate",
        "topk_sensitive_exposure",
        "budget_normalized_gain",
        "status",
        "error_message",
        "timestamp",
    ),
    required_fields=(
        "run_id",
        "method_name",
        "method_family",
        "dataset",
        "image_id",
        "budget_type",
        "budget_ratio",
        "runtime_ms",
        "encrypted_ratio",
        "protected_sensitive_ratio",
        "unprotected_sensitive_ratio",
        "entropy",
        "npcr",
        "uaci",
        "ssim",
        "psnr",
        "semantic_exposure_rate",
        "topk_sensitive_exposure",
        "budget_normalized_gain",
        "status",
        "error_message",
    ),
)

LEGACY_RESULT_SCHEMA = OutputSchema(
    name="legacy_result",
    fieldnames=(
        "method",
        "mode",
        "budget_limit",
        "budget_used",
        "budget_utilization",
        "status",
        "time_ms",
        "high_count",
        "medium_count",
        "low_count",
        "none_count",
        "high_ratio",
        "medium_ratio",
        "low_ratio",
        "none_ratio",
        "timestamp",
        "alpha",
        "beta",
        "category",
        "image",
        "n_blocks",
        "entropy",
        "npcr",
        "uaci",
        "correlation_h",
        "correlation_v",
        "correlation_d",
        "plain_cipher_corr",
        "pearson_rule_final",
        "pearson_sem_final",
        "topk_overlap_rule",
        "topk_overlap_sem",
        "calibrated",
        "allocator_type",
        "seed",
        "error_message",
    ),
    required_fields=(
        "method",
        "mode",
        "budget_limit",
        "budget_used",
        "budget_utilization",
        "status",
        "time_ms",
        "high_count",
        "medium_count",
        "low_count",
        "none_count",
        "high_ratio",
        "medium_ratio",
        "low_ratio",
        "none_ratio",
        "timestamp",
    ),
)


# Additive V3 publication schema. Historical schemas above intentionally retain
# their original field names and widths for backward-compatible V1/V2 evidence.
V3_MANDATORY_FIELDS: Tuple[str, ...] = (
    "method",
    "mode",
    "budget_limit",
    "budget_used",
    "budget_utilization",
    "status",
    "time_ms",
    "high_count",
    "medium_count",
    "low_count",
    "none_count",
    "high_ratio",
    "medium_ratio",
    "low_ratio",
    "none_ratio",
    "timestamp",
)

V3_RESEARCH_FIELDS: Tuple[str, ...] = (
    "image_id",
    "seed",
    "condition_key",
    "manifest_digest",
    "external_label_source",
    "external_label_digest",
    "external_sensitive_block_count",
    "external_sensitive_protected_count",
    "external_sensitive_protection_ratio",
    "external_sensitive_exposure_ratio",
    "method_defined_risk_protection_ratio",
    "directly_accessible_block_ratio",
    "score_digest",
    "allocation_digest",
    "scorer_evidence_status",
    "allocation_objective",
    "scoring_time_ms",
    "allocation_time_ms",
    "encryption_time_ms",
    "execution_status",
    "error_message",
    "skip_reason",
)

V3_RESULT_SCHEMA = OutputSchema(
    name="v3_result",
    fieldnames=V3_MANDATORY_FIELDS + V3_RESEARCH_FIELDS,
    required_fields=V3_MANDATORY_FIELDS + V3_RESEARCH_FIELDS,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_V3_COUNT_FIELDS = ("high_count", "medium_count", "low_count", "none_count")
_V3_RATIO_FIELDS = ("high_ratio", "medium_ratio", "low_ratio", "none_ratio")
_V3_FLOAT_FIELDS = (
    "budget_limit",
    "budget_used",
    "budget_utilization",
    "time_ms",
    "scoring_time_ms",
    "allocation_time_ms",
    "encryption_time_ms",
)


def compute_v3_condition_key(
    *,
    manifest_digest: str,
    image_id: str,
    method: str,
    budget_limit: float,
    seed: int,
) -> str:
    """Return a full digest identifying one V3 image/method/budget/seed row."""
    payload = {
        "budget_limit_hex": float(budget_limit).hex(),
        "image_id": str(image_id),
        "manifest_digest": str(manifest_digest),
        "method": str(method),
        "seed": int(seed),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _v3_is_blank(value: Any) -> bool:
    return value is None or value == ""


def _v3_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise SchemaValidationError(f"V3 field {field} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"V3 field {field} must be a finite number.") from exc
    if not math.isfinite(number):
        raise SchemaValidationError(f"V3 field {field} must be a finite number.")
    return number


def _v3_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise SchemaValidationError(f"V3 field {field} must be a non-negative integer.")
    if isinstance(value, int):
        result = value
    else:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(
                f"V3 field {field} must be a non-negative integer."
            ) from exc
        if not math.isfinite(number) or not number.is_integer():
            raise SchemaValidationError(f"V3 field {field} must be a non-negative integer.")
        result = int(number)
    if result < 0:
        raise SchemaValidationError(f"V3 field {field} must be a non-negative integer.")
    return result


def _validate_v3_record(record: Mapping[str, Any]) -> None:
    """Apply status-aware semantic validation to one normalized V3 row."""
    for field in (
        "method",
        "mode",
        "image_id",
        "condition_key",
        "manifest_digest",
        "timestamp",
        "execution_status",
    ):
        if not isinstance(record.get(field), str) or not str(record[field]).strip():
            raise SchemaValidationError(f"V3 field {field} must be a non-empty string.")
    for field in ("condition_key", "manifest_digest"):
        if _SHA256_RE.fullmatch(str(record[field])) is None:
            raise SchemaValidationError(f"V3 field {field} must be a lowercase SHA-256.")
    for field in ("score_digest", "allocation_digest"):
        if not _v3_is_blank(record.get(field)) and _SHA256_RE.fullmatch(
            str(record[field])
        ) is None:
            raise SchemaValidationError(f"V3 field {field} must be blank or a SHA-256.")
    scorer_status = record.get("scorer_evidence_status")
    if scorer_status not in {"FULL", "PARTIAL", "DEGRADED"}:
        raise SchemaValidationError("V3 scorer_evidence_status is invalid.")
    allocation_objective = record.get("allocation_objective")
    if not _v3_is_blank(allocation_objective):
        _v3_finite_number(allocation_objective, "allocation_objective")
    seed = _v3_nonnegative_int(record.get("seed"), "seed")
    numeric = {field: _v3_finite_number(record.get(field), field) for field in _V3_FLOAT_FIELDS}
    if any(numeric[field] < 0.0 for field in _V3_FLOAT_FIELDS):
        raise SchemaValidationError("V3 budget and timing fields must be non-negative.")
    expected_utilization = (
        numeric["budget_used"] / numeric["budget_limit"] * 100.0
        if numeric["budget_limit"] > 0.0
        else 0.0
    )
    if not math.isclose(
        numeric["budget_utilization"],
        expected_utilization,
        abs_tol=1e-9,
    ):
        raise SchemaValidationError("V3 budget utilization is inconsistent.")
    expected_condition_key = compute_v3_condition_key(
        manifest_digest=str(record["manifest_digest"]),
        image_id=str(record["image_id"]),
        method=str(record["method"]),
        budget_limit=numeric["budget_limit"],
        seed=seed,
    )
    if record["condition_key"] != expected_condition_key:
        raise SchemaValidationError("V3 condition_key does not match row identity.")

    status = str(record["status"])
    execution_status = str(record["execution_status"])
    if execution_status not in {"success", "skipped", "runtime_error"}:
        raise SchemaValidationError("V3 execution_status is invalid.")
    if status == "FULL" and execution_status != "success":
        raise SchemaValidationError("V3 FULL rows require execution_status=success.")
    if execution_status == "success" and _v3_is_blank(record.get("allocation_digest")):
        raise SchemaValidationError("Successful V3 rows require allocation_digest.")
    if execution_status == "success" and record.get("mode") != "none" and _v3_is_blank(
        record.get("score_digest")
    ):
        raise SchemaValidationError("Scored V3 rows require score_digest.")

    counts = {field: _v3_nonnegative_int(record.get(field), field) for field in _V3_COUNT_FIELDS}
    total_blocks = sum(counts.values())
    if execution_status == "success" and total_blocks <= 0:
        raise SchemaValidationError("Successful V3 rows require a non-empty block count.")
    if total_blocks > 0:
        ratios = {field: _v3_finite_number(record.get(field), field) for field in _V3_RATIO_FIELDS}
        if any(not 0.0 <= value <= 1.0 for value in ratios.values()):
            raise SchemaValidationError("V3 protection ratios must be in [0, 1].")
        if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
            raise SchemaValidationError("V3 protection ratios must sum to 1.")
        for count_field, ratio_field in zip(_V3_COUNT_FIELDS, _V3_RATIO_FIELDS):
            if not math.isclose(
                ratios[ratio_field],
                counts[count_field] / total_blocks,
                abs_tol=1e-9,
            ):
                raise SchemaValidationError("V3 count and ratio fields are inconsistent.")
        direct_ratio = _v3_finite_number(
            record.get("directly_accessible_block_ratio"),
            "directly_accessible_block_ratio",
        )
        if not math.isclose(direct_ratio, counts["none_count"] / total_blocks, abs_tol=1e-9):
            raise SchemaValidationError(
                "V3 directly_accessible_block_ratio must equal none_count / block_count."
            )
    elif not _v3_is_blank(record.get("directly_accessible_block_ratio")):
        raise SchemaValidationError("Empty-block V3 rows require a blank access ratio.")

    method_risk = record.get("method_defined_risk_protection_ratio")
    if not _v3_is_blank(method_risk):
        method_risk_value = _v3_finite_number(
            method_risk,
            "method_defined_risk_protection_ratio",
        )
        if not 0.0 <= method_risk_value <= 1.0:
            raise SchemaValidationError("V3 method-defined risk ratio must be in [0, 1].")

    external_fields = (
        "external_sensitive_block_count",
        "external_sensitive_protected_count",
        "external_sensitive_protection_ratio",
        "external_sensitive_exposure_ratio",
    )
    source = record.get("external_label_source")
    digest = record.get("external_label_digest")
    if _v3_is_blank(source) and _v3_is_blank(digest):
        if any(not _v3_is_blank(record.get(field)) for field in external_fields):
            raise SchemaValidationError(
                "V3 external metrics cannot exist without external label identity."
            )
        if status == "FULL":
            raise SchemaValidationError("V3 FULL rows require complete external evidence.")
    else:
        if not isinstance(source, str) or not source.strip():
            raise SchemaValidationError("V3 external_label_source must be non-empty.")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise SchemaValidationError("V3 external_label_digest must be a SHA-256.")
        if execution_status == "runtime_error" and all(
            _v3_is_blank(record.get(field)) for field in external_fields
        ):
            if status != "DEGRADED":
                raise SchemaValidationError(
                    "V3 runtime-error rows with blank external metrics must be DEGRADED."
                )
            sensitive_count = None
            protected_count = None
        else:
            sensitive_count = _v3_nonnegative_int(
                record.get("external_sensitive_block_count"),
                "external_sensitive_block_count",
            )
            protected_count = _v3_nonnegative_int(
                record.get("external_sensitive_protected_count"),
                "external_sensitive_protected_count",
            )
        if sensitive_count is None or protected_count is None:
            pass
        elif protected_count > sensitive_count:
            raise SchemaValidationError("Protected sensitive count exceeds sensitive count.")
        elif sensitive_count == 0:
            if any(
                not _v3_is_blank(record.get(field))
                for field in (
                    "external_sensitive_protection_ratio",
                    "external_sensitive_exposure_ratio",
                )
            ):
                raise SchemaValidationError("Zero-sensitive V3 rows require blank external ratios.")
            if status == "FULL":
                raise SchemaValidationError("V3 FULL rows require defined external ratios.")
        else:
            protection_ratio = _v3_finite_number(
                record.get("external_sensitive_protection_ratio"),
                "external_sensitive_protection_ratio",
            )
            exposure_ratio = _v3_finite_number(
                record.get("external_sensitive_exposure_ratio"),
                "external_sensitive_exposure_ratio",
            )
            if not math.isclose(protection_ratio, protected_count / sensitive_count, abs_tol=1e-9):
                raise SchemaValidationError("V3 external sensitive counts and ratio disagree.")
            if not math.isclose(protection_ratio + exposure_ratio, 1.0, abs_tol=1e-9):
                raise SchemaValidationError("V3 external protection/exposure ratios must sum to 1.")

    error_message = record.get("error_message")
    skip_reason = record.get("skip_reason")
    if not isinstance(error_message, str) or not isinstance(skip_reason, str):
        raise SchemaValidationError("V3 error_message and skip_reason must be strings.")
    if status == "FULL" and (error_message or skip_reason):
        raise SchemaValidationError("V3 FULL rows cannot carry errors or skip reasons.")
    if status == "PARTIAL" and not skip_reason:
        raise SchemaValidationError("V3 PARTIAL rows require a documented skip_reason.")
    if status == "DEGRADED" and not (error_message or skip_reason):
        raise SchemaValidationError("V3 DEGRADED rows require an error or blocking reason.")


def coerce_v3_csv_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert one DictReader V3 row into typed values for strict validation."""
    typed = dict(record)
    for field in (*_V3_COUNT_FIELDS, "seed", "external_sensitive_block_count", "external_sensitive_protected_count"):
        if not _v3_is_blank(typed.get(field)):
            typed[field] = int(typed[field])
    for field in (
        *_V3_FLOAT_FIELDS,
        *_V3_RATIO_FIELDS,
        "external_sensitive_protection_ratio",
        "external_sensitive_exposure_ratio",
        "method_defined_risk_protection_ratio",
        "directly_accessible_block_ratio",
        "allocation_objective",
    ):
        if not _v3_is_blank(typed.get(field)):
            typed[field] = float(typed[field])
    return typed


def _coerce_schema(schema: OutputSchema | Sequence[str]) -> OutputSchema:
    """Accept either an OutputSchema or a plain fieldname sequence."""
    if isinstance(schema, OutputSchema):
        return schema
    fieldnames = tuple(schema)
    return OutputSchema(
        name="ad_hoc",
        fieldnames=fieldnames,
        required_fields=fieldnames,
    )


def normalize_status(status: Any) -> str:
    """Normalize legacy status values into the publication-safe status set."""
    if status is None:
        raise ValueError("Status cannot be None.")

    normalized = str(status).strip()
    if not normalized:
        raise ValueError("Status cannot be empty.")

    lowered = normalized.lower()
    mapped = LEGACY_STATUS_MAPPING.get(lowered, lowered)
    try:
        return normalize_publication_status(mapped, context={"context": "record"})
    except PublicationSchemaError as exc:
        raise ValueError(str(exc)) from exc


def validate_record_schema(
    record: Mapping[str, Any],
    schema: OutputSchema | Sequence[str],
    strict: bool = True,
) -> SchemaValidationResult:
    """Validate one record against the provided schema."""
    output_schema = _coerce_schema(schema)
    fieldnames = set(output_schema.fieldnames)
    unknown_fields = tuple(sorted(key for key in record.keys() if key not in fieldnames))

    if unknown_fields and strict:
        raise SchemaValidationError(
            f"Record contains unknown fields for schema `{output_schema.name}`: {list(unknown_fields)}"
        )

    raw_status = record.get(output_schema.status_field)
    normalized_status = None
    if raw_status is not None:
        mapped_status = LEGACY_STATUS_MAPPING.get(str(raw_status).strip().lower(), raw_status)
        try:
            normalized_status = normalize_publication_status(
                mapped_status,
                context={
                    "method_name": record.get("method_name", record.get("method")),
                    "image_id": record.get("image_id", record.get("image")),
                    "budget_ratio": record.get("budget_ratio", record.get("budget_limit")),
                    "seed": record.get("seed"),
                    "context": "record",
                },
            )
        except PublicationSchemaError as exc:
            raise SchemaValidationError(str(exc)) from exc

    normalized: Dict[str, Any] = {}
    for field in output_schema.fieldnames:
        value = record.get(field)
        if field == output_schema.status_field and normalized_status is not None:
            value = normalized_status
        elif (
            field == "skip_reason"
            and not value
            and str(raw_status or "").strip().lower() == "skipped_not_applicable"
        ):
            value = "not_applicable"
        elif field == "timestamp" and value in (None, ""):
            value = datetime.now().isoformat()
        normalized[field] = value

    missing_fields = tuple(
        field
        for field in output_schema.required_fields
        if field not in record or record.get(field) is None
    )
    if missing_fields:
        raise SchemaValidationError(
            f"Record is missing required fields for schema `{output_schema.name}`: {list(missing_fields)}"
        )
    if output_schema.name == V3_RESULT_SCHEMA.name:
        _validate_v3_record(normalized)

    return SchemaValidationResult(
        record=normalized,
        missing_fields=missing_fields,
        unknown_fields=unknown_fields,
    )


def write_csv_with_schema(
    csv_path: str | Path,
    records: Iterable[Mapping[str, Any]],
    schema: OutputSchema | Sequence[str],
    mode: str = "append",
    strict: bool = True,
    unknown_fields_log_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Write records to CSV using a strict schema.

    `mode="overwrite"` always rewrites the header.
    Unknown fields are never silently discarded.
    """
    output_schema = _coerce_schema(schema)
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    open_mode = "a" if mode == "append" else "w"
    write_header = mode == "overwrite" or not path.exists() or path.stat().st_size == 0

    normalized_records: List[Dict[str, Any]] = []
    unknown_log_records: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        result = validate_record_schema(record, output_schema, strict=strict)
        normalized_records.append(result.record)
        if result.unknown_fields:
            unknown_log_records.append(
                {
                    "row_index": index,
                    "unknown_fields": list(result.unknown_fields),
                }
            )

    if not strict and unknown_log_records and unknown_fields_log_path is not None:
        unknown_log_file = Path(unknown_fields_log_path)
        unknown_log_file.parent.mkdir(parents=True, exist_ok=True)
        with unknown_log_file.open("w", encoding="utf-8") as handle:
            json.dump({"unknown_fields": unknown_log_records}, handle, indent=2)

    with path.open(open_mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_schema.fieldnames))
        if write_header:
            writer.writeheader()
        for record in normalized_records:
            writer.writerow(record)

    return normalized_records


class IncrementalCheckpointWriter:
    """Append-only partial raw/attack CSV writer plus progress summary."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        enabled: bool = True,
        every_records: int = 100,
        expected_rows: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.enabled = bool(enabled)
        self.every_records = max(1, int(every_records or 100))
        self.expected_rows = expected_rows
        self.metadata = dict(metadata or {})
        self.raw_partial_path = self.run_dir / "raw_results_partial.csv"
        self.attack_partial_path = self.run_dir / "attack_results_partial.csv"
        self.progress_path = self.run_dir / "progress_summary.json"
        self._buffer: List[Dict[str, Any]] = []
        self.records_written = 0
        self.status_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_record: Dict[str, Any] = {}

    def append(self, record: Mapping[str, Any]) -> None:
        """Buffer a record and flush when the configured threshold is reached."""
        if not self.enabled:
            return
        self._buffer.append(dict(record))
        if len(self._buffer) >= self.every_records:
            self.flush()

    def seed_records(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Write already-completed records into a fresh resumed run checkpoint."""
        if not self.enabled:
            return
        if self.records_written or self._buffer:
            raise RuntimeError("Cannot seed checkpoint writer after new records were written.")
        normalized = [normalize_publication_record(record) for record in records]
        if normalized:
            self._append_csv(self.raw_partial_path, PUBLICATION_RAW_SCHEMA, normalized)
            self._append_csv(
                self.attack_partial_path,
                tuple(ATTACK_SCHEMA),
                [self._attack_row(record) for record in normalized],
            )
            for record in normalized:
                status = str(record.get("status", ""))
                self.status_counts[status] = self.status_counts.get(status, 0) + 1
                if record.get("error_message"):
                    self.error_counts["error_message"] = self.error_counts.get("error_message", 0) + 1
                if record.get("metric_error"):
                    self.error_counts["metric_error"] = self.error_counts.get("metric_error", 0) + 1
                self.last_record = dict(record)
            self.records_written = len(normalized)
        self._write_progress()

    def flush(self, *, force: bool = False) -> None:
        """Write buffered records to partial CSVs and refresh progress JSON."""
        if not self.enabled:
            return
        if not self._buffer and not force:
            return

        normalized = [normalize_publication_record(record) for record in self._buffer]
        if normalized:
            self._append_csv(self.raw_partial_path, PUBLICATION_RAW_SCHEMA, normalized)
            self._append_csv(
                self.attack_partial_path,
                tuple(ATTACK_SCHEMA),
                [self._attack_row(record) for record in normalized],
            )
            for record in normalized:
                status = str(record.get("status", ""))
                self.status_counts[status] = self.status_counts.get(status, 0) + 1
                if record.get("error_message"):
                    self.error_counts["error_message"] = self.error_counts.get("error_message", 0) + 1
                if record.get("metric_error"):
                    self.error_counts["metric_error"] = self.error_counts.get("metric_error", 0) + 1
                self.last_record = record
            self.records_written += len(normalized)
            self._buffer.clear()

        self._write_progress()

    def _append_csv(
        self,
        path: Path,
        fieldnames: Sequence[str],
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            if write_header:
                writer.writeheader()
            for record in records:
                writer.writerow({field: record.get(field) for field in fieldnames})

    def _attack_row(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "source_run_id": record.get("run_id"),
            "method_name": record.get("method_name"),
            "image_id": record.get("image_id"),
            "budget_ratio": record.get("budget_ratio"),
            "seed": record.get("seed"),
            "attack_name": record.get("attack_name"),
            "leakage_score": record.get("reconstruction_leakage_score"),
            "attack_success_rate": record.get("attack_success_rate"),
            "status": normalize_publication_status(
                record.get("status"),
                context={
                    "method_name": record.get("method_name"),
                    "image_id": record.get("image_id"),
                    "budget_ratio": record.get("budget_ratio"),
                    "seed": record.get("seed"),
                    "context": "record",
                },
            ),
            "error_message": record.get("error_message") or record.get("metric_error") or "",
        }

    def _write_progress(self) -> None:
        payload = {
            "records_written": self.records_written,
            "expected_rows": self.expected_rows,
            "last_method": self.last_record.get("method_name", ""),
            "last_image_id": self.last_record.get("image_id", ""),
            "last_budget_ratio": self.last_record.get("budget_ratio", ""),
            "last_seed": self.last_record.get("seed", ""),
            "last_update_time": datetime.now().isoformat(),
            "status_counts_so_far": dict(sorted(self.status_counts.items())),
            "error_counts_so_far": dict(sorted(self.error_counts.items())),
        }
        payload.update(self.metadata)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.progress_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)


class ExperimentOutput:
    """
    Backward-compatible CSV writer backed by the strict schema utilities.
    """

    MANDATORY_COLUMNS = list(LEGACY_RESULT_SCHEMA.required_fields)
    OPTIONAL_COLUMNS = [
        column
        for column in LEGACY_RESULT_SCHEMA.fieldnames
        if column not in LEGACY_RESULT_SCHEMA.required_fields
    ]
    ALL_COLUMNS = list(LEGACY_RESULT_SCHEMA.fieldnames)
    VALID_STATUS_VALUES = list(PUBLICATION_STATUS_VALUES)

    def __init__(
        self,
        csv_path: str,
        mode: str = "append",
        schema: OutputSchema = LEGACY_RESULT_SCHEMA,
        strict: bool = True,
        unknown_fields_log_path: Optional[str] = None,
    ):
        self.csv_path = csv_path
        self.mode = mode
        self.schema = schema
        self.strict = strict
        self.unknown_fields_log_path = unknown_fields_log_path
        self._buffer: List[Dict[str, Any]] = []
        self._row_count = 0

    def open(self) -> "ExperimentOutput":
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        return self

    def __enter__(self) -> "ExperimentOutput":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if self._buffer:
            write_csv_with_schema(
                self.csv_path,
                self._buffer,
                self.schema,
                mode=self.mode if self._row_count == 0 else "append",
                strict=self.strict,
                unknown_fields_log_path=self.unknown_fields_log_path,
            )
            self._buffer.clear()

    def _validate_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        result = validate_record_schema(row, self.schema, strict=self.strict)
        return result.record

    def write_row(self, row: Dict[str, Any]) -> None:
        validated = self._validate_row(row)
        write_csv_with_schema(
            self.csv_path,
            [validated],
            self.schema,
            mode=self.mode if self._row_count == 0 else "append",
            strict=self.strict,
            unknown_fields_log_path=self.unknown_fields_log_path,
        )
        self._row_count += 1

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        validated_rows = [self._validate_row(row) for row in rows]
        write_csv_with_schema(
            self.csv_path,
            validated_rows,
            self.schema,
            mode=self.mode if self._row_count == 0 else "append",
            strict=self.strict,
            unknown_fields_log_path=self.unknown_fields_log_path,
        )
        self._row_count += len(validated_rows)

    def save_config_snapshot(self, config: Dict[str, Any], experiment_name: str | None = None) -> str:
        snapshot_dir = Path(self.csv_path).parent / "config_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = experiment_name or Path(self.csv_path).stem
        snapshot_path = snapshot_dir / f"{exp_name}_{timestamp}.json"
        with snapshot_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        return str(snapshot_path)

    def save_json_summary(self, summary: Dict[str, Any], name: str | None = None) -> str:
        summary_dir = Path(self.csv_path).parent / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = name or Path(self.csv_path).stem
        summary_path = summary_dir / f"{base_name}_{timestamp}.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return str(summary_path)

    @property
    def row_count(self) -> int:
        return self._row_count


class StatusValidator:
    """Compatibility shim for legacy status validation call sites."""

    VALID = list(PUBLICATION_STATUS_VALUES)

    @classmethod
    def normalize(cls, status: Any, budget_utilization: float | None = None) -> str:
        del budget_utilization
        return normalize_status(status)

    @classmethod
    def validate_csv(cls, csv_path: str) -> Dict[str, Any]:
        issues: List[str] = []
        total_rows = 0
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                total_rows += 1
                status = row.get("status")
                try:
                    normalize_status(status)
                except ValueError as exc:
                    issues.append(f"Row {index}: {exc}")

        return {
            "total_rows": total_rows,
            "invalid_status_count": len(issues),
            "issues": issues[:10],
        }
