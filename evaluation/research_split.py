"""Deterministic, image-disjoint research split artifacts bound to a manifest."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


ARTIFACT_VERSION = "v3_image_split_v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_LOWERCASE_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_FIELDS = {
    "artifact_version",
    "manifest_digest",
    "seed",
    "validation_fraction",
    "test_fraction",
    "stratify_field",
    "train_image_ids",
    "validation_image_ids",
    "test_image_ids",
    "per_stratum_counts",
    "split_digest",
}
_COUNT_FIELDS = {"train", "validation", "test", "total"}


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value using the canonical encoding used by all digests."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not JSON serializable: {exc}") from exc


def _sha256_payload(value: Any) -> str:
    """Return a full SHA-256 digest for a canonical JSON payload."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    """Return whether a value is a complete hexadecimal SHA-256 string."""
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _is_lowercase_sha256(value: Any) -> bool:
    """Return whether a value is a canonical lowercase SHA-256 string."""
    return (
        isinstance(value, str)
        and _LOWERCASE_SHA256_PATTERN.fullmatch(value) is not None
    )


def _validate_seed(seed: Any) -> int:
    """Validate and return a NumPy-compatible non-negative integer seed."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _validate_fractions(
    validation_fraction: Any,
    test_fraction: Any,
) -> tuple[float, float]:
    """Validate the two finite, positive split fractions."""
    values: list[float] = []
    for name, value in (
        ("validation_fraction", validation_fraction),
        ("test_fraction", test_fraction),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite numeric fraction")
        number = float(value)
        if not math.isfinite(number) or not 0.0 < number < 1.0:
            raise ValueError(f"{name} must be a finite fraction strictly between 0 and 1")
        values.append(number)
    if values[0] + values[1] >= 1.0:
        raise ValueError("validation_fraction and test_fraction must sum to less than 1")
    return values[0], values[1]


def _copy_and_validate_rows(
    rows: Sequence[Mapping[str, Any]],
    stratify_field: str,
) -> list[dict[str, Any]]:
    """Copy manifest rows and validate identity and stratification fields."""
    if not isinstance(stratify_field, str) or not stratify_field.strip():
        raise ValueError("stratify_field must be a non-empty string")
    if isinstance(rows, (str, bytes)):
        raise ValueError("manifest rows must be a non-empty sequence of mappings")

    copied: list[dict[str, Any]] = []
    try:
        iterator = enumerate(rows)
    except TypeError as exc:
        raise ValueError("manifest rows must be a non-empty sequence of mappings") from exc

    seen_ids: set[str] = set()
    image_id_by_sha256: dict[str, str] = {}
    for index, row in iterator:
        if not isinstance(row, Mapping):
            raise ValueError(f"manifest row {index} must be a mapping")
        plain_row = dict(row)
        if not all(isinstance(key, str) for key in plain_row):
            raise ValueError(f"manifest row {index} must use string field names")

        image_id = plain_row.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError(f"manifest row {index} has an empty or invalid image_id")
        if image_id in seen_ids:
            raise ValueError(f"manifest contains duplicate image_id {image_id!r}")
        seen_ids.add(image_id)

        stratum = plain_row.get(stratify_field)
        if not isinstance(stratum, str) or not stratum.strip():
            raise ValueError(
                f"manifest row {index} has an empty or invalid stratum in "
                f"{stratify_field!r}"
            )

        image_sha256 = plain_row.get("sha256")
        if not _is_sha256(image_sha256):
            raise ValueError(f"manifest row {index} has an invalid sha256")
        normalized_sha256 = str(image_sha256).lower()
        previous_image_id = image_id_by_sha256.get(normalized_sha256)
        if previous_image_id is not None:
            raise ValueError(
                "manifest contains duplicate content sha256 for distinct image IDs "
                f"{previous_image_id!r} and {image_id!r}"
            )
        image_id_by_sha256[normalized_sha256] = image_id

        try:
            _canonical_json_bytes(plain_row)
        except ValueError as exc:
            raise ValueError(f"manifest row {index} is not JSON serializable: {exc}") from exc
        copied.append(plain_row)

    if not copied:
        raise ValueError("manifest must contain at least one image row")
    return copied


def _manifest_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash complete manifest rows after ordering them by image identity."""
    return _sha256_payload(sorted(rows, key=lambda row: row["image_id"]))


def _split_payload(
    *,
    artifact_version: str,
    manifest_digest: str,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    stratify_field: str,
    train_image_ids: Sequence[str],
    validation_image_ids: Sequence[str],
    test_image_ids: Sequence[str],
    per_stratum_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Return the JSON-compatible artifact body protected by ``split_digest``."""
    return {
        "artifact_version": artifact_version,
        "manifest_digest": manifest_digest,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "stratify_field": stratify_field,
        "train_image_ids": list(train_image_ids),
        "validation_image_ids": list(validation_image_ids),
        "test_image_ids": list(test_image_ids),
        "per_stratum_counts": {
            stratum: dict(counts)
            for stratum, counts in per_stratum_counts.items()
        },
    }


def _validate_id_list(value: Any, field_name: str) -> tuple[str, ...]:
    """Validate one sorted, duplicate-free JSON image ID list."""
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON list")
    if any(not isinstance(image_id, str) or not image_id.strip() for image_id in value):
        raise ValueError(f"{field_name} must contain non-empty string image IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} contains duplicate image IDs")
    if value != sorted(value):
        raise ValueError(f"{field_name} must be sorted")
    return tuple(value)


def _normalize_id_partition(value: Any, field_name: str) -> tuple[str, ...]:
    """Validate a constructor partition and normalize JSON lists to tuples."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple of image IDs")
    image_ids = tuple(value)
    if any(
        not isinstance(image_id, str) or not image_id.strip()
        for image_id in image_ids
    ):
        raise ValueError(f"{field_name} must contain non-empty string image IDs")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{field_name} contains duplicate image IDs")
    if image_ids != tuple(sorted(image_ids)):
        raise ValueError(f"{field_name} must be sorted")
    return image_ids


def _validate_counts(value: Any) -> dict[str, dict[str, int]]:
    """Validate and copy the per-stratum partition counts."""
    if not isinstance(value, Mapping) or not value:
        raise ValueError("per_stratum_counts must be a non-empty mapping")
    result: dict[str, dict[str, int]] = {}
    for stratum, counts in value.items():
        if not isinstance(stratum, str) or not stratum.strip():
            raise ValueError("per_stratum_counts contains an invalid stratum")
        if not isinstance(counts, Mapping) or set(counts) != _COUNT_FIELDS:
            raise ValueError(
                f"per_stratum_counts[{stratum!r}] must contain exactly "
                f"{sorted(_COUNT_FIELDS)}"
            )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in counts.values()
        ):
            raise ValueError(
                f"per_stratum_counts[{stratum!r}] values must be positive integers"
            )
        if counts["train"] + counts["validation"] + counts["test"] != counts["total"]:
            raise ValueError(f"per_stratum_counts[{stratum!r}] does not sum to total")
        result[stratum] = dict(counts)
    return result


@dataclass(frozen=True, kw_only=True)
class ResearchSplitArtifact:
    """Frozen identity, membership, and provenance for a V3 image split."""

    artifact_version: str = ARTIFACT_VERSION
    manifest_digest: str
    seed: int
    validation_fraction: float
    test_fraction: float
    stratify_field: str
    train_image_ids: tuple[str, ...]
    validation_image_ids: tuple[str, ...]
    test_image_ids: tuple[str, ...]
    per_stratum_counts: Mapping[str, Mapping[str, int]]
    split_digest: str

    def __post_init__(self) -> None:
        """Validate all public-constructor invariants and deeply freeze counts."""
        if self.artifact_version != ARTIFACT_VERSION:
            raise ValueError(
                f"artifact_version must be {ARTIFACT_VERSION!r}, "
                f"got {self.artifact_version!r}"
            )
        if not _is_lowercase_sha256(self.manifest_digest):
            raise ValueError("manifest_digest must be a lowercase hexadecimal SHA-256")
        if not _is_lowercase_sha256(self.split_digest):
            raise ValueError("split_digest must be a lowercase hexadecimal SHA-256")

        checked_seed = _validate_seed(self.seed)
        checked_validation, checked_test = _validate_fractions(
            self.validation_fraction,
            self.test_fraction,
        )
        if not isinstance(self.stratify_field, str) or not self.stratify_field.strip():
            raise ValueError("stratify_field must be a non-empty string")

        train_ids = _normalize_id_partition(
            self.train_image_ids,
            "train_image_ids",
        )
        validation_ids = _normalize_id_partition(
            self.validation_image_ids,
            "validation_image_ids",
        )
        test_ids = _normalize_id_partition(
            self.test_image_ids,
            "test_image_ids",
        )
        train_set = set(train_ids)
        validation_set = set(validation_ids)
        test_set = set(test_ids)
        if not train_set and not validation_set and not test_set:
            raise ValueError("image ID partitions must contain at least one image ID")
        if (
            train_set & validation_set
            or train_set & test_set
            or validation_set & test_set
        ):
            raise ValueError("stored split membership has pairwise overlap")

        checked_counts = _validate_counts(self.per_stratum_counts)
        if sum(item["train"] for item in checked_counts.values()) != len(train_ids):
            raise ValueError("per_stratum_counts train total does not match membership")
        if sum(item["validation"] for item in checked_counts.values()) != len(
            validation_ids
        ):
            raise ValueError(
                "per_stratum_counts validation total does not match membership"
            )
        if sum(item["test"] for item in checked_counts.values()) != len(test_ids):
            raise ValueError("per_stratum_counts test total does not match membership")

        expected_split_digest = _sha256_payload(
            _split_payload(
                artifact_version=self.artifact_version,
                manifest_digest=self.manifest_digest,
                seed=checked_seed,
                validation_fraction=checked_validation,
                test_fraction=checked_test,
                stratify_field=self.stratify_field,
                train_image_ids=train_ids,
                validation_image_ids=validation_ids,
                test_image_ids=test_ids,
                per_stratum_counts=checked_counts,
            )
        )
        if self.split_digest != expected_split_digest:
            raise ValueError(
                "split_digest does not authenticate the canonical artifact payload"
            )

        frozen_counts = {
            stratum: MappingProxyType(dict(counts))
            for stratum, counts in checked_counts.items()
        }
        object.__setattr__(self, "seed", checked_seed)
        object.__setattr__(self, "validation_fraction", checked_validation)
        object.__setattr__(self, "test_fraction", checked_test)
        object.__setattr__(self, "train_image_ids", train_ids)
        object.__setattr__(self, "validation_image_ids", validation_ids)
        object.__setattr__(self, "test_image_ids", test_ids)
        object.__setattr__(
            self,
            "per_stratum_counts",
            MappingProxyType(frozen_counts),
        )

    @classmethod
    def from_manifest(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        validation_fraction: float,
        test_fraction: float,
        stratify_field: str,
    ) -> "ResearchSplitArtifact":
        """Build a deterministic image-disjoint split from complete manifest rows."""
        checked_seed = _validate_seed(seed)
        checked_validation, checked_test = _validate_fractions(
            validation_fraction,
            test_fraction,
        )
        copied_rows = _copy_and_validate_rows(rows, stratify_field)

        ids_by_stratum: dict[str, list[str]] = {}
        for row in copied_rows:
            ids_by_stratum.setdefault(row[stratify_field], []).append(row["image_id"])
        too_small = {
            stratum: len(image_ids)
            for stratum, image_ids in ids_by_stratum.items()
            if len(image_ids) < 3
        }
        if too_small:
            raise ValueError(
                "each stratum must contain at least 3 images; "
                f"invalid strata: {too_small}"
            )

        rng = np.random.default_rng(checked_seed)
        train_ids: list[str] = []
        validation_ids: list[str] = []
        test_ids: list[str] = []
        counts_by_stratum: dict[str, dict[str, int]] = {}
        for stratum in sorted(ids_by_stratum):
            sorted_ids = sorted(ids_by_stratum[stratum])
            permuted_ids = [str(value) for value in rng.permutation(sorted_ids)]
            image_count = len(permuted_ids)
            validation_count = max(
                1,
                math.floor(image_count * checked_validation),
            )
            test_count = max(1, math.floor(image_count * checked_test))
            train_count = image_count - validation_count - test_count
            if train_count < 1:
                raise ValueError(
                    f"split fractions leave no train image in stratum {stratum!r}"
                )

            validation_ids.extend(permuted_ids[:validation_count])
            test_ids.extend(
                permuted_ids[validation_count : validation_count + test_count]
            )
            train_ids.extend(permuted_ids[validation_count + test_count :])
            counts_by_stratum[stratum] = {
                "train": train_count,
                "validation": validation_count,
                "test": test_count,
                "total": image_count,
            }

        train_tuple = tuple(sorted(train_ids))
        validation_tuple = tuple(sorted(validation_ids))
        test_tuple = tuple(sorted(test_ids))
        source_digest = _manifest_digest(copied_rows)
        payload = _split_payload(
            artifact_version=ARTIFACT_VERSION,
            manifest_digest=source_digest,
            seed=checked_seed,
            validation_fraction=checked_validation,
            test_fraction=checked_test,
            stratify_field=stratify_field,
            train_image_ids=train_tuple,
            validation_image_ids=validation_tuple,
            test_image_ids=test_tuple,
            per_stratum_counts=counts_by_stratum,
        )
        return cls(
            manifest_digest=source_digest,
            seed=checked_seed,
            validation_fraction=checked_validation,
            test_fraction=checked_test,
            stratify_field=stratify_field,
            train_image_ids=train_tuple,
            validation_image_ids=validation_tuple,
            test_image_ids=test_tuple,
            per_stratum_counts=counts_by_stratum,
            split_digest=_sha256_payload(payload),
        )

    @classmethod
    def from_existing_partition(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        split_field: str,
        stratify_field: str,
    ) -> "ResearchSplitArtifact":
        """Build an artifact that preserves an existing official partition."""
        checked_seed = _validate_seed(seed)
        if not isinstance(split_field, str) or not split_field.strip():
            raise ValueError("split_field must be a non-empty string")
        copied_rows = _copy_and_validate_rows(rows, stratify_field)

        ids_by_split: dict[str, list[str]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        counts_by_stratum: dict[str, dict[str, int]] = {}
        for index, row in enumerate(copied_rows):
            split = row.get(split_field)
            if split not in ids_by_split:
                raise ValueError(
                    f"manifest row {index} has an unsupported split in "
                    f"{split_field!r}: {split!r}"
                )
            image_id = row["image_id"]
            stratum = row[stratify_field]
            ids_by_split[split].append(image_id)
            counts = counts_by_stratum.setdefault(
                stratum,
                {"train": 0, "validation": 0, "test": 0, "total": 0},
            )
            counts[split] += 1
            counts["total"] += 1

        row_count = len(copied_rows)
        validation_fraction = len(ids_by_split["validation"]) / row_count
        test_fraction = len(ids_by_split["test"]) / row_count
        checked_validation, checked_test = _validate_fractions(
            validation_fraction,
            test_fraction,
        )
        train_ids = tuple(sorted(ids_by_split["train"]))
        validation_ids = tuple(sorted(ids_by_split["validation"]))
        test_ids = tuple(sorted(ids_by_split["test"]))
        source_digest = _manifest_digest(copied_rows)
        payload = _split_payload(
            artifact_version=ARTIFACT_VERSION,
            manifest_digest=source_digest,
            seed=checked_seed,
            validation_fraction=checked_validation,
            test_fraction=checked_test,
            stratify_field=stratify_field,
            train_image_ids=train_ids,
            validation_image_ids=validation_ids,
            test_image_ids=test_ids,
            per_stratum_counts=counts_by_stratum,
        )
        return cls(
            manifest_digest=source_digest,
            seed=checked_seed,
            validation_fraction=checked_validation,
            test_fraction=checked_test,
            stratify_field=stratify_field,
            train_image_ids=train_ids,
            validation_image_ids=validation_ids,
            test_image_ids=test_ids,
            per_stratum_counts=counts_by_stratum,
            split_digest=_sha256_payload(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible representation of the artifact."""
        payload = _split_payload(
            artifact_version=self.artifact_version,
            manifest_digest=self.manifest_digest,
            seed=self.seed,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
            stratify_field=self.stratify_field,
            train_image_ids=self.train_image_ids,
            validation_image_ids=self.validation_image_ids,
            test_image_ids=self.test_image_ids,
            per_stratum_counts=self.per_stratum_counts,
        )
        payload["split_digest"] = self.split_digest
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchSplitArtifact":
        """Parse an artifact mapping with strict schema, type, and digest checks."""
        if not isinstance(value, dict):
            raise ValueError("split artifact root must be a JSON object")
        if set(value) != _ARTIFACT_FIELDS:
            missing = sorted(_ARTIFACT_FIELDS - set(value))
            extra = sorted(set(value) - _ARTIFACT_FIELDS)
            raise ValueError(
                f"split artifact fields are not exact; missing={missing}, extra={extra}"
            )
        if value["artifact_version"] != ARTIFACT_VERSION:
            raise ValueError(
                f"unsupported artifact_version {value['artifact_version']!r}; "
                f"expected {ARTIFACT_VERSION!r}"
            )
        if not _is_lowercase_sha256(value["manifest_digest"]):
            raise ValueError(
                "manifest_digest must be a full lowercase hexadecimal SHA-256"
            )
        if not _is_lowercase_sha256(value["split_digest"]):
            raise ValueError(
                "split_digest must be a full lowercase hexadecimal SHA-256"
            )
        checked_seed = _validate_seed(value["seed"])
        checked_validation, checked_test = _validate_fractions(
            value["validation_fraction"],
            value["test_fraction"],
        )
        stratify_field = value["stratify_field"]
        if not isinstance(stratify_field, str) or not stratify_field.strip():
            raise ValueError("stratify_field must be a non-empty string")

        train_ids = _validate_id_list(value["train_image_ids"], "train_image_ids")
        validation_ids = _validate_id_list(
            value["validation_image_ids"],
            "validation_image_ids",
        )
        test_ids = _validate_id_list(value["test_image_ids"], "test_image_ids")
        train_set = set(train_ids)
        validation_set = set(validation_ids)
        test_set = set(test_ids)
        if (
            train_set & validation_set
            or train_set & test_set
            or validation_set & test_set
        ):
            raise ValueError("stored split membership has pairwise overlap")

        counts = _validate_counts(value["per_stratum_counts"])
        if sum(item["train"] for item in counts.values()) != len(train_ids):
            raise ValueError("per_stratum_counts train total does not match membership")
        if sum(item["validation"] for item in counts.values()) != len(validation_ids):
            raise ValueError("per_stratum_counts validation total does not match membership")
        if sum(item["test"] for item in counts.values()) != len(test_ids):
            raise ValueError("per_stratum_counts test total does not match membership")

        payload = dict(value)
        stored_split_digest = payload.pop("split_digest")
        expected_split_digest = _sha256_payload(payload)
        if stored_split_digest != expected_split_digest:
            raise ValueError(
                "split_digest does not authenticate the canonical artifact payload"
            )

        return cls(
            manifest_digest=value["manifest_digest"],
            seed=checked_seed,
            validation_fraction=checked_validation,
            test_fraction=checked_test,
            stratify_field=stratify_field,
            train_image_ids=train_ids,
            validation_image_ids=validation_ids,
            test_image_ids=test_ids,
            per_stratum_counts=counts,
            split_digest=stored_split_digest,
        )

    def write_json(self, output: str | Path) -> Path:
        """Write canonical human-readable JSON with one trailing newline."""
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(
            (
                json.dumps(
                    self.to_dict(),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
        return output_path

    @classmethod
    def read_json(cls, path: str | Path) -> "ResearchSplitArtifact":
        """Read and strictly validate a split artifact JSON file."""
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return cls.from_dict(value)

    def validate(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        split_field: str | None = None,
    ) -> None:
        """Fail closed unless the artifact exactly rebuilds from ``rows``."""
        # Validate stored schema, disjointness, counts, and self-digest first.
        self.from_dict(self.to_dict())
        checked_rows = _copy_and_validate_rows(rows, self.stratify_field)

        if split_field is None:
            random_error: ValueError | None = None
            try:
                expected = self.from_manifest(
                    checked_rows,
                    seed=self.seed,
                    validation_fraction=self.validation_fraction,
                    test_fraction=self.test_fraction,
                    stratify_field=self.stratify_field,
                )
            except ValueError as exc:
                random_error = exc
                expected = None
            if expected != self:
                common_fields = set.intersection(
                    *(set(row) for row in checked_rows)
                )
                for candidate_field in sorted(common_fields):
                    raw_values = [row[candidate_field] for row in checked_rows]
                    if any(type(value) is not str for value in raw_values):
                        continue
                    values = set(raw_values)
                    if values != {"train", "validation", "test"}:
                        continue
                    candidate_expected = self.from_existing_partition(
                        checked_rows,
                        seed=self.seed,
                        split_field=candidate_field,
                        stratify_field=self.stratify_field,
                    )
                    if candidate_expected == self:
                        expected = candidate_expected
                        break
            if expected is None and random_error is not None:
                raise random_error
        else:
            expected = self.from_existing_partition(
                checked_rows,
                seed=self.seed,
                split_field=split_field,
                stratify_field=self.stratify_field,
            )
        manifest_ids = {row["image_id"] for row in checked_rows}
        train_ids = set(self.train_image_ids)
        validation_ids = set(self.validation_image_ids)
        test_ids = set(self.test_image_ids)
        if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
            raise ValueError("stored split membership has pairwise overlap")
        if train_ids | validation_ids | test_ids != manifest_ids:
            raise ValueError("stored split membership does not provide exact manifest coverage")
        if self != expected:
            mismatched = [
                field_name
                for field_name in _ARTIFACT_FIELDS
                if getattr(self, field_name) != getattr(expected, field_name)
            ]
            raise ValueError(
                "stored artifact does not match deterministic manifest rebuild; "
                f"mismatched fields: {sorted(mismatched)}"
            )
