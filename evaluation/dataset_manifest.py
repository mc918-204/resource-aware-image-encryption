"""Dataset manifest utilities for debug, validation, and publication protocols."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
READY_COLUMN_ALIASES = ("is_publication_ready", "publication_ready")
DATASET_MANIFEST_COLUMNS = (
    "dataset_name",
    "split",
    "category",
    "image_id",
    "image_path",
    "width",
    "height",
    "channels",
    "file_size_bytes",
    "sha256",
    "source",
    "license_note",
    "is_publication_ready",
    "exclusion_reason",
)


class DatasetManifestError(ValueError):
    """Raised when a manifest record or manifest file is invalid."""


def compute_sha256(path: str | Path) -> str:
    """Compute the SHA-256 digest for one file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found for SHA-256 computation: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y"}


def _coerce_rows(manifest: str | Path | Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
    return [dict(row) for row in manifest]


def _normalize_ready_column(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize legacy readiness aliases onto the canonical manifest column."""
    normalized = dict(row)
    canonical_key = READY_COLUMN_ALIASES[0]
    legacy_key = READY_COLUMN_ALIASES[1]
    if canonical_key not in normalized and legacy_key in normalized:
        normalized[canonical_key] = normalized[legacy_key]
    normalized.pop(legacy_key, None)
    return normalized


def scan_image_dataset(
    dataset_root: str | Path,
    dataset_name: str,
    split: str,
    source: str = "local_debug_dataset",
    license_note: str = "local_debug_only",
    is_publication_ready: bool = False,
    exclusion_reason: str = "debug_only_small_scale_dataset",
    strict_images: bool = True,
) -> List[Dict[str, Any]]:
    """Scan a dataset directory into a manifest-ready row list."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    rows: List[Dict[str, Any]] = []
    for image_path in sorted(path for path in root.rglob("*") if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES):
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                channels = len(image.getbands())
        except (UnidentifiedImageError, OSError) as exc:
            if strict_images:
                raise DatasetManifestError(f"Failed to read image `{image_path}`: {exc}") from exc
            width = 0
            height = 0
            channels = 0
            exclusion_reason = f"invalid_image:{exc}"

        relative_path = image_path.relative_to(root)
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else "uncategorized"
        rows.append(
            {
                "dataset_name": dataset_name,
                "split": split,
                "category": category,
                "image_id": image_path.stem,
                "image_path": str(image_path.resolve()),
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "file_size_bytes": int(image_path.stat().st_size),
                "sha256": compute_sha256(image_path),
                "source": source,
                "license_note": license_note,
                "is_publication_ready": bool(is_publication_ready),
                "exclusion_reason": "" if is_publication_ready else exclusion_reason,
            }
        )
    return rows


def validate_manifest_schema(
    manifest: str | Path | Iterable[Mapping[str, Any]],
    strict: bool = True,
    check_image_paths: bool = True,
) -> List[Dict[str, Any]]:
    """Validate one manifest table and return normalized rows."""
    rows = _coerce_rows(manifest)
    normalized_rows: List[Dict[str, Any]] = []
    required_columns = set(DATASET_MANIFEST_COLUMNS)

    for index, row in enumerate(rows):
        row = _normalize_ready_column(row)
        missing_columns = [column for column in DATASET_MANIFEST_COLUMNS if column not in row]
        if missing_columns:
            raise DatasetManifestError(
                f"Manifest row {index} is missing required columns: {missing_columns}"
            )

        unknown_columns = sorted(column for column in row.keys() if column not in required_columns)
        if strict and unknown_columns:
            raise DatasetManifestError(
                f"Manifest row {index} contains unknown columns: {unknown_columns}"
            )

        normalized = {column: row.get(column) for column in DATASET_MANIFEST_COLUMNS}
        normalized["width"] = int(normalized["width"])
        normalized["height"] = int(normalized["height"])
        normalized["channels"] = int(normalized["channels"])
        normalized["file_size_bytes"] = int(normalized["file_size_bytes"])
        normalized["is_publication_ready"] = _coerce_bool(normalized["is_publication_ready"])
        normalized["sha256"] = str(normalized["sha256"]).strip().lower()
        if len(normalized["sha256"]) != 64:
            raise DatasetManifestError(
                f"Manifest row {index} has invalid sha256 value: {normalized['sha256']}"
            )

        image_path = Path(str(normalized["image_path"]))
        if check_image_paths and not image_path.exists():
            raise DatasetManifestError(
                f"Manifest row {index} points to a missing image path: {image_path}"
            )
        normalized_rows.append(normalized)

    return normalized_rows


def summarize_manifest(manifest: str | Path | Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize one manifest for reporting."""
    rows = validate_manifest_schema(manifest, strict=False, check_image_paths=False)
    category_counts: Dict[str, int] = {}
    publication_ready_count = 0
    excluded_count = 0

    for row in rows:
        category = str(row["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
        if row["is_publication_ready"]:
            publication_ready_count += 1
        else:
            excluded_count += 1

    dataset_names = sorted({str(row["dataset_name"]) for row in rows})
    splits = sorted({str(row["split"]) for row in rows})
    return {
        "num_images": len(rows),
        "dataset_names": dataset_names,
        "splits": splits,
        "categories": sorted(category_counts.keys()),
        "category_counts": category_counts,
        "publication_ready_count": publication_ready_count,
        "excluded_count": excluded_count,
    }


def filter_publication_ready(
    manifest: str | Path | Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return only publication-ready rows from a manifest."""
    rows = validate_manifest_schema(manifest, strict=False, check_image_paths=False)
    return [row for row in rows if row["is_publication_ready"]]


def write_manifest_csv(output_path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write a manifest with the canonical column order."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DATASET_MANIFEST_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in DATASET_MANIFEST_COLUMNS})
    return target
