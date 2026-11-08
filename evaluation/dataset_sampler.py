"""Deterministic dataset sampling utilities for publication experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.evaluation.dataset_manifest import validate_manifest_schema, write_manifest_csv


VALID_BALANCE_KEYS = {"dataset_name", "category"}
VALID_SAMPLING_MODES = {"full_manifest", "balanced_subset", "subset_100", "legacy_category_cap"}


class DatasetSamplingError(ValueError):
    """Raised when sampling config or coverage is invalid."""


def _reject_config_keys(config: Mapping[str, Any], keys: Sequence[str], *, sampling_mode: str) -> None:
    """Reject config keys that would silently undermine one sampling mode."""
    present = [key for key in keys if config.get(key) is not None]
    if present:
        raise DatasetSamplingError(
            f"{sampling_mode} does not allow the following config keys: {sorted(present)}."
        )


def _normalize_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def _coerce_ready(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _row_is_publication_ready(row: Mapping[str, Any]) -> bool:
    """Accept canonical and legacy readiness keys when filtering manifest rows."""
    if "is_publication_ready" in row:
        return _coerce_ready(row.get("is_publication_ready"))
    return _coerce_ready(row.get("publication_ready"))


def _stable_row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset_name", "")),
        str(row.get("category", "")),
        str(row.get("image_id", "")),
        str(row.get("image_path", "")),
    )


def _seeded_sort_key(row: Mapping[str, Any], seed: int) -> str:
    payload = "|".join([str(seed), *[str(part) for part in _stable_row_key(row)]])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _group_counts(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        group = str(row.get(key, "unknown"))
        counts[group] = counts.get(group, 0) + 1
    return dict(sorted(counts.items()))


def _filter_datasets(
    rows: Sequence[Mapping[str, Any]],
    include_datasets: Sequence[str] | None = None,
    exclude_datasets: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    include = {item for item in (include_datasets or [])}
    exclude = {item for item in (exclude_datasets or [])}
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        dataset_name = str(row.get("dataset_name", ""))
        if include and dataset_name not in include:
            continue
        if dataset_name in exclude:
            continue
        filtered.append(dict(row))
    return filtered


def load_manifest(manifest: str | Path | Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Load and normalize a dataset manifest."""
    return validate_manifest_schema(manifest, strict=False, check_image_paths=True)


def filter_publication_ready(manifest_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only manifest rows marked publication-ready."""
    return [dict(row) for row in manifest_rows if _row_is_publication_ready(row)]


def sample_full_manifest(publication_ready_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return all publication-ready rows in deterministic order."""
    return [dict(row) for row in sorted(publication_ready_rows, key=_stable_row_key)]


def sample_balanced_subset(
    publication_ready_rows: Sequence[Mapping[str, Any]],
    *,
    balance_by: str,
    max_images_per_group: int,
    min_images_per_group: int | None = None,
    include_datasets: Sequence[str] | None = None,
    exclude_datasets: Sequence[str] | None = None,
    sampling_seed: int = 0,
) -> List[Dict[str, Any]]:
    """Deterministically sample up to N rows per dataset/category group."""
    if balance_by not in VALID_BALANCE_KEYS:
        raise DatasetSamplingError(f"Unsupported balance_by `{balance_by}`. Expected one of {sorted(VALID_BALANCE_KEYS)}.")
    if max_images_per_group <= 0:
        raise DatasetSamplingError("max_images_per_group must be > 0.")

    filtered = _filter_datasets(publication_ready_rows, include_datasets, exclude_datasets)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in filtered:
        group = str(row.get(balance_by, "unknown"))
        groups.setdefault(group, []).append(dict(row))

    sampled: List[Dict[str, Any]] = []
    for group_name in sorted(groups):
        candidates = sorted(groups[group_name], key=lambda row: (_seeded_sort_key(row, sampling_seed), _stable_row_key(row)))
        if min_images_per_group is not None and len(candidates) < min_images_per_group:
            raise DatasetSamplingError(
                f"Group `{group_name}` has only {len(candidates)} images, below min_images_per_group={min_images_per_group}."
            )
        sampled.extend(candidates[:max_images_per_group])

    return sorted(sampled, key=_stable_row_key)


def sample_fixed_total_subset(
    publication_ready_rows: Sequence[Mapping[str, Any]],
    *,
    sample_total: int,
    sampling_seed: int = 0,
    include_datasets: Sequence[str] | None = None,
    exclude_datasets: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Deterministically sample a fixed total while covering datasets as broadly as possible."""
    if sample_total <= 0:
        raise DatasetSamplingError("sample_total must be > 0.")

    filtered = _filter_datasets(publication_ready_rows, include_datasets, exclude_datasets)
    if sample_total >= len(filtered):
        return sample_full_manifest(filtered)

    per_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for row in filtered:
        dataset_name = str(row.get("dataset_name", "unknown"))
        per_dataset.setdefault(dataset_name, []).append(dict(row))

    for dataset_name, rows in per_dataset.items():
        per_dataset[dataset_name] = sorted(rows, key=lambda row: (_seeded_sort_key(row, sampling_seed), _stable_row_key(row)))

    sampled: List[Dict[str, Any]] = []
    dataset_names = sorted(per_dataset)
    pointers = {dataset_name: 0 for dataset_name in dataset_names}

    while len(sampled) < sample_total:
        progressed = False
        for dataset_name in dataset_names:
            index = pointers[dataset_name]
            candidates = per_dataset[dataset_name]
            if index >= len(candidates):
                continue
            sampled.append(candidates[index])
            pointers[dataset_name] += 1
            progressed = True
            if len(sampled) >= sample_total:
                break
        if not progressed:
            break

    return sorted(sampled, key=_stable_row_key)


def summarize_sampling(
    manifest_rows: Sequence[Mapping[str, Any]],
    publication_ready_rows: Sequence[Mapping[str, Any]],
    sampled_rows: Sequence[Mapping[str, Any]],
    *,
    sampling_mode: str,
    sampling_seed: int = 0,
    balance_by: str | None = None,
    include_datasets: Sequence[str] | None = None,
    exclude_datasets: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Build a deterministic sampling summary for run artifacts."""
    available_datasets = sorted({str(row.get("dataset_name", "")) for row in publication_ready_rows})
    sampled_datasets = sorted({str(row.get("dataset_name", "")) for row in sampled_rows})
    return {
        "sampling_mode": sampling_mode,
        "sampling_seed": int(sampling_seed),
        "balance_by": balance_by or "N/A",
        "include_datasets": list(include_datasets or []),
        "exclude_datasets": list(exclude_datasets or []),
        "total_manifest_images": int(len(manifest_rows)),
        "publication_ready_images": int(len(publication_ready_rows)),
        "sampled_images": int(len(sampled_rows)),
        "dataset_sample_counts": _group_counts(sampled_rows, "dataset_name"),
        "category_sample_counts": _group_counts(sampled_rows, "category"),
        "missing_sampled_datasets": sorted(set(available_datasets) - set(sampled_datasets)),
    }


def validate_sampling_coverage(
    publication_ready_rows: Sequence[Mapping[str, Any]],
    sampled_rows: Sequence[Mapping[str, Any]],
    *,
    sampling_mode: str,
    include_datasets: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Validate coverage guarantees for one sampling mode."""
    if sampling_mode not in VALID_SAMPLING_MODES:
        raise DatasetSamplingError(f"Unsupported sampling_mode `{sampling_mode}`.")

    available_dataset_names = sorted({str(row.get("dataset_name", "")) for row in publication_ready_rows})
    sampled_dataset_names = sorted({str(row.get("dataset_name", "")) for row in sampled_rows})
    missing_sampled_datasets = sorted(set(available_dataset_names) - set(sampled_dataset_names))
    warnings: List[str] = []
    errors: List[str] = []

    if sampling_mode == "full_manifest" and len(sampled_rows) < len(publication_ready_rows):
        errors.append(
            f"full_manifest sampled_images={len(sampled_rows)} is less than publication_ready_images={len(publication_ready_rows)}."
        )
    if include_datasets:
        missing_requested = sorted(set(include_datasets) - set(sampled_dataset_names))
        if missing_requested:
            errors.append(f"Requested datasets missing from sample: {missing_requested}.")
    if sampling_mode == "subset_100" and missing_sampled_datasets:
        warnings.append(f"subset_100 did not cover all datasets: {missing_sampled_datasets}.")

    return {
        "sampling_mode": sampling_mode,
        "sampled_images": int(len(sampled_rows)),
        "publication_ready_images": int(len(publication_ready_rows)),
        "sampled_datasets": sampled_dataset_names,
        "missing_sampled_datasets": missing_sampled_datasets,
        "warnings": warnings,
        "errors": errors,
        "is_valid": not errors,
    }


def sample_from_config(
    manifest_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Resolve the final sampled row list and summary from experiment config."""
    all_rows = _normalize_rows(manifest_rows)
    publication_ready_rows = filter_publication_ready(all_rows)

    sampling_mode = str(config.get("sampling_mode") or "").strip()
    if not sampling_mode:
        if "max_images_per_category" in config:
            sampling_mode = "legacy_category_cap"
        else:
            sampling_mode = "full_manifest"

    if sampling_mode == "full_manifest":
        _reject_config_keys(
            config,
            (
                "max_images_per_category",
                "sample_total",
                "max_images_per_group",
                "min_images_per_group",
                "include_datasets",
                "exclude_datasets",
                "balance_by",
            ),
            sampling_mode="full_manifest",
        )
        sampled_rows = sample_full_manifest(publication_ready_rows)
        summary = summarize_sampling(
            all_rows,
            publication_ready_rows,
            sampled_rows,
            sampling_mode="full_manifest",
            sampling_seed=int(config.get("sampling_seed", 0)),
        )
    elif sampling_mode == "balanced_subset":
        sampled_rows = sample_balanced_subset(
            publication_ready_rows,
            balance_by=str(config.get("balance_by", "dataset_name")),
            max_images_per_group=int(config.get("max_images_per_group", 0)),
            min_images_per_group=int(config["min_images_per_group"]) if config.get("min_images_per_group") is not None else None,
            include_datasets=config.get("include_datasets"),
            exclude_datasets=config.get("exclude_datasets"),
            sampling_seed=int(config.get("sampling_seed", 0)),
        )
        summary = summarize_sampling(
            all_rows,
            publication_ready_rows,
            sampled_rows,
            sampling_mode="balanced_subset",
            sampling_seed=int(config.get("sampling_seed", 0)),
            balance_by=str(config.get("balance_by", "dataset_name")),
            include_datasets=config.get("include_datasets"),
            exclude_datasets=config.get("exclude_datasets"),
        )
    elif sampling_mode == "subset_100":
        sampled_rows = sample_fixed_total_subset(
            publication_ready_rows,
            sample_total=int(config.get("sample_total", 100)),
            sampling_seed=int(config.get("sampling_seed", 0)),
            include_datasets=config.get("include_datasets"),
            exclude_datasets=config.get("exclude_datasets"),
        )
        summary = summarize_sampling(
            all_rows,
            publication_ready_rows,
            sampled_rows,
            sampling_mode="subset_100",
            sampling_seed=int(config.get("sampling_seed", 0)),
            include_datasets=config.get("include_datasets"),
            exclude_datasets=config.get("exclude_datasets"),
        )
    elif sampling_mode == "legacy_category_cap":
        sampled_rows = sample_balanced_subset(
            publication_ready_rows,
            balance_by="category",
            max_images_per_group=int(config.get("max_images_per_category", 0)),
            sampling_seed=int(config.get("sampling_seed", 0)),
        )
        summary = summarize_sampling(
            all_rows,
            publication_ready_rows,
            sampled_rows,
            sampling_mode="legacy_category_cap",
            sampling_seed=int(config.get("sampling_seed", 0)),
            balance_by="category",
        )
    else:
        raise DatasetSamplingError(f"Unsupported sampling_mode `{sampling_mode}`.")

    coverage = validate_sampling_coverage(
        publication_ready_rows,
        sampled_rows,
        sampling_mode="subset_100" if sampling_mode == "legacy_category_cap" and int(config.get("sample_total", 0) or 0) == 100 else sampling_mode,
        include_datasets=config.get("include_datasets"),
    )
    if sampling_mode == "full_manifest" and not coverage["is_valid"]:
        raise DatasetSamplingError("; ".join(coverage["errors"]))
    return sampled_rows, summary, coverage


def write_sampling_artifacts(
    run_dir: str | Path,
    sampled_rows: Sequence[Mapping[str, Any]],
    sampling_summary: Mapping[str, Any],
) -> Dict[str, str]:
    """Persist the sampled manifest and sampling summary."""
    root = Path(run_dir)
    manifest_path = root / "sampled_dataset_manifest.csv"
    summary_path = root / "sampling_summary.json"
    write_manifest_csv(manifest_path, sampled_rows)
    summary_path.write_text(json.dumps(dict(sampling_summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "sampled_dataset_manifest": str(manifest_path),
        "sampling_summary": str(summary_path),
    }
