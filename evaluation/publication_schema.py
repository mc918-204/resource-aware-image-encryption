"""Publication-level raw result schema for M2-A and M2-B."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PUBLICATION_SCHEMA_VERSION = "m2a_v3"
PUBLICATION_STATUS_VALUES = (
    "DEGRADED",
    "PARTIAL",
    "FULL",
)
EXECUTION_STATUS_VALUES = (
    "success",
    "budget_exceeded",
    "runtime_error",
    "invalid_config",
    "missing_dependency",
    "invalid_baseline",
    "skipped",
)
LEGACY_EXECUTION_TO_PUBLICATION_STATUS = {
    "success": "FULL",
    "feasible": "FULL",
    "full": "FULL",
    "partial": "PARTIAL",
    "skipped": "PARTIAL",
    "skipped_not_applicable": "PARTIAL",
    "degraded": "DEGRADED",
    "failed": "DEGRADED",
    "exceeded": "DEGRADED",
    "budget_exceeded": "DEGRADED",
    "runtime_error": "DEGRADED",
    "invalid_config": "DEGRADED",
    "missing_dependency": "DEGRADED",
    "invalid_baseline": "DEGRADED",
    "unknown": "DEGRADED",
}
MANDATORY_EXPERIMENT_COLUMNS = (
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
RECORD_NOT_APPLICABLE_STATUS_VALUES = {
    "skipped_not_applicable",
}
METRIC_NOT_APPLICABLE_STATUS_VALUES = {
    "not_applicable",
    "metric_not_applicable",
    "unsupported_metric",
    "n/a",
    "na",
}
PUBLICATION_RAW_SCHEMA = (
    "publication_schema_version",
    "run_id",
    "experiment_name",
    "experiment_stage",
    "mode",
    "method_name",
    "method",
    "method_family",
    "dataset_name",
    "split",
    "category",
    "image_id",
    "image_path",
    "seed",
    "status",
    "execution_status",
    "error_message",
    "timestamp",
    "budget_type",
    "budget_ratio",
    "budget_value",
    "budget_limit",
    "budget_unit",
    "budget_used",
    "budget_utilization",
    "budget_satisfied",
    "scorer_type",
    "allocator_type",
    "protection_levels",
    "semantic_backend",
    "model_path_resolved",
    "device_resolved",
    "fallback_reason",
    "runtime_ms",
    "time_ms",
    "encryption_time_ms",
    "scoring_time_ms",
    "allocation_time_ms",
    "reconstruction_time_ms",
    "memory_peak_mb",
    "total_blocks",
    "protected_blocks",
    "high_protected_blocks",
    "high_count",
    "medium_protected_blocks",
    "medium_count",
    "low_protected_blocks",
    "low_count",
    "unprotected_blocks",
    "none_count",
    "encrypted_ratio",
    "protected_sensitive_ratio",
    "unprotected_sensitive_ratio",
    "protection_density",
    "entropy",
    "npcr",
    "uaci",
    "psnr",
    "ssim",
    "adjacent_correlation_horizontal",
    "adjacent_correlation_vertical",
    "adjacent_correlation_diagonal",
    "semantic_exposure_rate",
    "topk_sensitive_exposure",
    "budget_normalized_protection_gain",
    "oracle_gap",
    "utility",
    "utility_per_cost",
    "attack_name",
    "attacker_knowledge",
    "detector_name",
    "detector_confidence_before",
    "detector_confidence_after",
    "attack_success_rate",
    "reconstruction_leakage_score",
    "comparison_role",
    "protection_profile",
    "metric_error",
    "skip_reason",
    "metric_status",
    "unsupported_metrics",
    "metric_notes",
    "allowed_table",
    "is_encryption",
    "notes",
    "high_ratio",
    "medium_ratio",
    "low_ratio",
    "none_ratio",
    "strong_protected_ratio",
    "weighted_protection_ratio",
    # ---- v4 research columns (improvement plan C2-C6) --------------------
    # C2: which protection implementation produced this row. `real_chain`
    # means src.pipeline actually encrypted; `visual_proxy` means the
    # historical SHA-256 byte-expansion stand-in, which carries no
    # encryption cost and must never be read as a cipher measurement.
    "protection_backend",
    "kdf_profile",
    "roundtrip_sad",
    "roundtrip_verified",
    # C3: staged timing. `scoring_time_source` distinguishes a measured
    # semantic cost from one replicated across (seed, budget) repeats, so a
    # cached 0.0 can never masquerade as a real measurement again.
    "scoring_time_source",
    "decode_ms",
    "feature_ms",
    "semantic_ms",
    "permute_ms",
    "diffuse_ms",
    "aead_ms",
    "serialize_ms",
    "total_ms",
    "is_warmup",
    "model_load_ms",
    "ciphertext_bytes_total",
    "metadata_bytes",
    "expansion_ratio",
    # C5: the two independent fairness axes, and honest budget accounting.
    "fairness_axis",
    "k_selected",
    "budget_mode",
    "budget_utilization_true",
    "budget_feasible",
    "budget_affordable_high_blocks",
    # C6: metric definition provenance.
    "topk_mode",
    "topk_resolved",
    "protection_weight_low",
    "semantic_exposure_rate_selfdefined",
    "topk_sensitive_exposure_selfdefined",
    # C4: external ground-truth metrics. These are the intended dependent
    # variables for the paper's main tables; the `*_selfdefined` columns
    # above are diagnostics only.
    "external_label_source",
    "external_label_digest",
    "ext_sensitive_blocks",
    "ext_nonsensitive_blocks",
    "ext_sensitive_block_recall_strong",
    "ext_sensitive_block_recall_any",
    "ext_sensitive_exposed_raw_ratio",
    "ext_nonsensitive_encrypted_ratio",
    "ext_sensitive_pixel_recall_strong",
    "ext_oracle_recall_strong",
    "ext_oracle_gap_strong",
    # C8: exact-optimum allocator comparison. `dp_fallback_reason` is non-empty
    # only when the exact allocator refused an instance for exceeding its
    # documented guard rails, which must be recorded rather than hidden.
    "allocation_objective",
    "optimality_method",
    "dp_fallback_reason",
)

_CORE_REQUIRED_PUBLICATION_COLUMNS = (
    "publication_schema_version",
    "run_id",
    "experiment_name",
    "experiment_stage",
    "method_name",
    "method_family",
    "dataset_name",
    "split",
    "category",
    "image_id",
    "image_path",
    "seed",
    "status",
    "error_message",
    "budget_type",
    "budget_ratio",
    "budget_value",
    "budget_unit",
    "budget_used",
    "budget_satisfied",
    "scorer_type",
    "allocator_type",
    "protection_levels",
    "semantic_backend",
    "model_path_resolved",
    "device_resolved",
    "fallback_reason",
    "runtime_ms",
    "encryption_time_ms",
    "scoring_time_ms",
    "allocation_time_ms",
    "reconstruction_time_ms",
    "memory_peak_mb",
    "total_blocks",
    "protected_blocks",
    "high_protected_blocks",
    "medium_protected_blocks",
    "low_protected_blocks",
    "unprotected_blocks",
    "encrypted_ratio",
    "protected_sensitive_ratio",
    "unprotected_sensitive_ratio",
    "protection_density",
)
REQUIRED_PUBLICATION_COLUMNS = tuple(
    dict.fromkeys(_CORE_REQUIRED_PUBLICATION_COLUMNS + MANDATORY_EXPERIMENT_COLUMNS)
)
DERIVED_PUBLICATION_COLUMNS = {
    "method",
    "mode",
    "budget_limit",
    "budget_utilization",
    "time_ms",
    "high_count",
    "medium_count",
    "low_count",
    "none_count",
    "timestamp",
    "execution_status",
    "high_ratio",
    "medium_ratio",
    "low_ratio",
    "none_ratio",
}
OPTIONAL_PUBLICATION_COLUMNS = tuple(
    column for column in PUBLICATION_RAW_SCHEMA if column not in REQUIRED_PUBLICATION_COLUMNS
)


class PublicationSchemaError(ValueError):
    """Raised when a publication record or table violates the frozen schema."""


def get_publication_schema_version() -> str:
    """Return the frozen publication schema version."""
    return PUBLICATION_SCHEMA_VERSION


def _status_context_value(context: Any, key: str) -> Any:
    if isinstance(context, Mapping):
        return context.get(key, "")
    return ""


def _status_error_message(value: Any, context: Any, suggested_fix: str) -> str:
    return (
        f"Unsupported publication status `{value}`. "
        f"Allowed values: {list(PUBLICATION_STATUS_VALUES)}. "
        f"bad status={value}; "
        f"method_name={_status_context_value(context, 'method_name')}; "
        f"image_id={_status_context_value(context, 'image_id')}; "
        f"budget_ratio={_status_context_value(context, 'budget_ratio')}; "
        f"seed={_status_context_value(context, 'seed')}; "
        f"suggested fix={suggested_fix}"
    )


def normalize_publication_status(raw_status: Any, context: Any = None) -> str:
    """
    Normalize record-level publication status without relaxing the schema.

    Historical execution outcomes are accepted as inputs, but the publication
    record stores only evidence-completeness states. The original outcome is
    retained separately in ``execution_status``.
    """
    if raw_status is None:
        raise PublicationSchemaError(
            _status_error_message(
                raw_status,
                context,
                "Set a concrete publication status before writing the record.",
            )
        )
    normalized = str(raw_status).strip().lower()
    if not normalized:
        raise PublicationSchemaError(
            _status_error_message(
                raw_status,
                context,
                "Set status to DEGRADED, PARTIAL, or FULL.",
            )
        )

    context_kind = ""
    if isinstance(context, str):
        context_kind = context
    elif isinstance(context, Mapping):
        context_kind = str(context.get("context", context.get("status_context", ""))).strip().lower()

    if normalized in METRIC_NOT_APPLICABLE_STATUS_VALUES:
        if context_kind == "metric":
            return "not_applicable"
        raise PublicationSchemaError(
            _status_error_message(
                raw_status,
                context,
                "Use a three-state publication status and record not_applicable in "
                "skip_reason, metric_status, unsupported_metrics, or notes.",
            )
        )
    canonical = {value.lower(): value for value in PUBLICATION_STATUS_VALUES}
    if normalized in canonical:
        return canonical[normalized]
    if normalized in LEGACY_EXECUTION_TO_PUBLICATION_STATUS:
        return LEGACY_EXECUTION_TO_PUBLICATION_STATUS[normalized]
    if normalized not in canonical:
        raise PublicationSchemaError(
            _status_error_message(
                raw_status,
                context,
                "Map this value before writing CSV; do not extend the publication status schema.",
            )
        )
    return canonical[normalized]


def _normalize_status(value: Any, context: Any = None) -> str:
    """Backward-compatible wrapper for strict publication status normalization."""
    return normalize_publication_status(value, context=context)


def _append_note(existing: Any, note: str) -> str:
    current = str(existing or "").strip()
    if not current:
        return note
    if note in current:
        return current
    return f"{current}; {note}"


def _normalize_status_metadata(record: Mapping[str, Any], normalized: Dict[str, Any]) -> None:
    raw_status = str(record.get("status", "")).strip().lower()
    execution_status = str(record.get("execution_status") or "").strip().lower()
    if not execution_status:
        if raw_status in EXECUTION_STATUS_VALUES:
            execution_status = raw_status
        elif raw_status == "feasible":
            execution_status = "success"
        elif raw_status in {"failed", "unknown"}:
            execution_status = "runtime_error"
        elif raw_status == "exceeded":
            execution_status = "budget_exceeded"
        elif raw_status in {"full", "partial", "degraded"}:
            execution_status = raw_status
        elif raw_status in RECORD_NOT_APPLICABLE_STATUS_VALUES:
            execution_status = "skipped"
    normalized["execution_status"] = execution_status
    if raw_status in RECORD_NOT_APPLICABLE_STATUS_VALUES:
        normalized["skip_reason"] = record.get("skip_reason") or "not_applicable"
        normalized["notes"] = _append_note(
            record.get("notes"),
            "record_status_normalized_from_skipped_not_applicable",
        )


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _first_value(record: Mapping[str, Any], *keys: str) -> Any:
    """Return the first present, non-empty value among equivalent fields."""
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _budget_utilization(record: Mapping[str, Any]) -> Any:
    explicit = record.get("budget_utilization")
    if explicit not in (None, ""):
        return explicit
    used = _first_value(record, "budget_used")
    limit = _first_value(record, "budget_limit", "budget_value")
    try:
        limit_value = float(limit)
        if limit_value == 0.0:
            return None
        return 100.0 * float(used) / limit_value
    except (TypeError, ValueError):
        return None


def _protection_ratio(record: Mapping[str, Any], ratio_key: str, *count_keys: str) -> Any:
    """Return an explicit protection ratio or derive it from block counts."""
    explicit = record.get(ratio_key)
    if explicit not in (None, ""):
        return explicit
    count = _first_value(record, *count_keys)
    total = _first_value(record, "total_blocks")
    if total in (None, ""):
        counts = [
            _first_value(record, "high_count", "high_protected_blocks"),
            _first_value(record, "medium_count", "medium_protected_blocks"),
            _first_value(record, "low_count", "low_protected_blocks"),
            _first_value(record, "none_count", "unprotected_blocks"),
        ]
        try:
            total = sum(float(value or 0.0) for value in counts)
        except (TypeError, ValueError):
            total = None
    try:
        total_value = float(total)
        if total_value <= 0.0:
            return 0.0
        return float(count or 0.0) / total_value
    except (TypeError, ValueError):
        return None


def _coerce_records(
    table: str | Path | Iterable[Mapping[str, Any]] | Any,
) -> List[Dict[str, Any]]:
    if isinstance(table, (str, Path)):
        table_path = Path(table)
        if not table_path.exists():
            raise FileNotFoundError(f"Publication table not found: {table_path}")
        with table_path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    if hasattr(table, "to_dict"):
        return [dict(row) for row in table.to_dict(orient="records")]

    return [dict(row) for row in table]


def normalize_publication_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize one publication record into the canonical column set."""
    normalized = {column: record.get(column) for column in PUBLICATION_RAW_SCHEMA}
    normalized["publication_schema_version"] = (
        record.get("publication_schema_version") or PUBLICATION_SCHEMA_VERSION
    )
    normalized["method"] = _first_value(record, "method", "method_name")
    normalized["mode"] = _first_value(record, "mode", "scorer_type", "experiment_stage")
    normalized["budget_limit"] = _first_value(record, "budget_limit", "budget_value")
    normalized["budget_utilization"] = _budget_utilization(record)
    normalized["time_ms"] = _first_value(record, "time_ms", "runtime_ms")
    normalized["high_count"] = _first_value(record, "high_count", "high_protected_blocks")
    normalized["medium_count"] = _first_value(
        record, "medium_count", "medium_protected_blocks"
    )
    normalized["low_count"] = _first_value(record, "low_count", "low_protected_blocks")
    normalized["none_count"] = _first_value(record, "none_count", "unprotected_blocks")
    normalized["high_ratio"] = _protection_ratio(
        record, "high_ratio", "high_count", "high_protected_blocks"
    )
    normalized["medium_ratio"] = _protection_ratio(
        record, "medium_ratio", "medium_count", "medium_protected_blocks"
    )
    normalized["low_ratio"] = _protection_ratio(
        record, "low_ratio", "low_count", "low_protected_blocks"
    )
    normalized["none_ratio"] = _protection_ratio(
        record, "none_ratio", "none_count", "unprotected_blocks"
    )
    normalized["timestamp"] = record.get("timestamp") or datetime.now().isoformat()
    status_context = {
        "method_name": record.get("method_name"),
        "image_id": record.get("image_id"),
        "budget_ratio": record.get("budget_ratio"),
        "seed": record.get("seed"),
        "context": "record",
    }
    normalized["status"] = _normalize_status(record.get("status"), context=status_context)
    normalized["budget_satisfied"] = _coerce_bool(record.get("budget_satisfied"))
    normalized["protection_levels"] = record.get("protection_levels") or ""
    normalized["comparison_role"] = record.get("comparison_role") or ""
    normalized["protection_profile"] = record.get("protection_profile") or ""
    normalized["metric_error"] = record.get("metric_error") or ""
    normalized["skip_reason"] = record.get("skip_reason") or ""
    normalized["metric_status"] = record.get("metric_status") or ""
    normalized["unsupported_metrics"] = record.get("unsupported_metrics") or ""
    normalized["metric_notes"] = record.get("metric_notes") or ""
    normalized["allowed_table"] = record.get("allowed_table") or ""
    normalized["is_encryption"] = record.get("is_encryption")
    normalized["notes"] = record.get("notes") or ""
    _normalize_status_metadata(record, normalized)
    return normalized


def validate_publication_record(
    record: Mapping[str, Any],
    strict: bool = True,
    expected_schema_version: str | None = PUBLICATION_SCHEMA_VERSION,
) -> Dict[str, Any]:
    """Validate one publication record and return the normalized form."""
    missing_columns = [
        column
        for column in REQUIRED_PUBLICATION_COLUMNS
        if column not in record and column not in DERIVED_PUBLICATION_COLUMNS
    ]
    if missing_columns:
        raise PublicationSchemaError(
            f"Publication record is missing required columns: {missing_columns}"
        )

    unknown_columns = sorted(column for column in record.keys() if column not in PUBLICATION_RAW_SCHEMA)
    if strict and unknown_columns:
        raise PublicationSchemaError(
            f"Publication record contains unknown columns: {unknown_columns}"
        )

    normalized = normalize_publication_record(record)
    if expected_schema_version is not None and normalized["publication_schema_version"] != expected_schema_version:
        raise PublicationSchemaError(
            "Publication schema version mismatch: "
            f"expected `{expected_schema_version}`, got `{normalized['publication_schema_version']}`"
        )
    return normalized


def validate_publication_table(
    table: str | Path | Iterable[Mapping[str, Any]] | Any,
    strict: bool = True,
    expected_schema_version: str | None = PUBLICATION_SCHEMA_VERSION,
) -> List[Dict[str, Any]]:
    """Validate a full publication raw-result table."""
    records = _coerce_records(table)
    if not records:
        return []
    return [
        validate_publication_record(
            record,
            strict=strict,
            expected_schema_version=expected_schema_version,
        )
        for record in records
    ]


def write_publication_table(output_path: str | Path, records: Sequence[Mapping[str, Any]]) -> Path:
    """Write a canonical publication raw-result table."""
    validated = validate_publication_table(records, strict=False)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PUBLICATION_RAW_SCHEMA))
        writer.writeheader()
        for record in validated:
            writer.writerow({column: record.get(column) for column in PUBLICATION_RAW_SCHEMA})
    return target
