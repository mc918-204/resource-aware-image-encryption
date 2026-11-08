"""Utilities for reproducible paper-grade experiment run artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml
from PIL import Image

from src.evaluation.experiment_output import (
    V3_RESULT_SCHEMA,
    SchemaValidationError,
    coerce_v3_csv_record,
    compute_v3_condition_key,
    validate_record_schema,
)


REQUIRED_ARTIFACT_FILES = (
    "config_snapshot.yaml",
    "command.txt",
    "random_seed.json",
    "git_commit.txt",
    "environment.txt",
    "dataset_manifest.csv",
    "artifact_bindings.json",
    "raw_results.csv",
    "summary.json",
)
REQUIRED_ARTIFACT_DIRS = ("logs", "figures")
REQUIRED_LOG_FILES = ("run.log",)
IDENTITY_BOUND_REQUIRED_FILES = (
    "config_snapshot.yaml",
    "dataset_manifest.csv",
    "random_seed.json",
)


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_import_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception:
        return "not_installed"
    return getattr(module, "__version__", "unknown")


def save_artifact_bindings(
    run_dir: str | Path,
    *,
    label_artifact_path: str | Path | None = None,
    label_audit_path: str | Path | None = None,
    split_artifact_path: str | Path | None = None,
    dataset_manifest_source_path: str | Path | None = None,
    paths_are_run_relative: bool = False,
) -> str:
    """Persist optional evidence bindings used by one paper run."""
    root = Path(run_dir)

    def normalize(value: str | Path | None) -> tuple[str, str]:
        if value is None:
            return "", ""
        path = Path(value)
        if paths_are_run_relative:
            if path.is_absolute() or len(path.parts) != 1 or path.name != str(value):
                raise ValueError("Run-relative artifact bindings must be local basenames.")
            resolved = root / path.name
            return path.name, _sha256_file(resolved) if resolved.is_file() else ""
        resolved = path.resolve()
        return str(resolved), _sha256_file(resolved) if resolved.is_file() else ""

    label_path, label_digest = normalize(label_artifact_path)
    audit_path, audit_digest = normalize(label_audit_path)
    split_path, split_digest = normalize(split_artifact_path)
    source_manifest_path, source_manifest_digest = normalize(
        dataset_manifest_source_path
    )

    payload = {
        "label_artifact_path": label_path,
        "label_artifact_sha256": label_digest,
        "label_audit_path": audit_path,
        "label_audit_sha256": audit_digest,
        "split_artifact_path": split_path,
        "split_artifact_sha256": split_digest,
        "required_artifact_sha256": {
            name: _sha256_file(root / name) if (root / name).is_file() else ""
            for name in IDENTITY_BOUND_REQUIRED_FILES
        },
    }
    if dataset_manifest_source_path is not None:
        payload["dataset_manifest_source_path"] = source_manifest_path
        payload["dataset_manifest_source_sha256"] = source_manifest_digest
    target = root / "artifact_bindings.json"
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return str(target)


def create_paper_run(
    experiment_name: str,
    base_dir: str | Path = "results/paper_runs",
    run_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    label_artifact_path: str | Path | None = None,
    label_audit_path: str | Path | None = None,
    split_artifact_path: str | Path | None = None,
) -> Dict[str, str]:
    """Create a reproducible paper run directory structure."""
    resolved_timestamp = timestamp or _now_timestamp()
    resolved_run_id = run_id or uuid.uuid4().hex[:8]
    run_dir = Path(base_dir) / f"{resolved_timestamp}_{resolved_run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    for directory_name in REQUIRED_ARTIFACT_DIRS:
        (run_dir / directory_name).mkdir(parents=True, exist_ok=True)

    artifact_bindings_path = save_artifact_bindings(
        run_dir,
        label_artifact_path=label_artifact_path,
        label_audit_path=label_audit_path,
        split_artifact_path=split_artifact_path,
    )

    return {
        "run_id": resolved_run_id,
        "timestamp": resolved_timestamp,
        "run_dir": str(run_dir),
        "logs_dir": str(run_dir / "logs"),
        "figures_dir": str(run_dir / "figures"),
        "config_snapshot_path": str(run_dir / "config_snapshot.yaml"),
        "command_path": str(run_dir / "command.txt"),
        "random_seed_path": str(run_dir / "random_seed.json"),
        "run_log_path": str(run_dir / "logs" / "run.log"),
        "git_commit_path": str(run_dir / "git_commit.txt"),
        "environment_path": str(run_dir / "environment.txt"),
        "dataset_manifest_path": str(run_dir / "dataset_manifest.csv"),
        "artifact_bindings_path": artifact_bindings_path,
        "raw_results_path": str(run_dir / "raw_results.csv"),
        "raw_results_partial_path": str(run_dir / "raw_results_partial.csv"),
        "attack_results_partial_path": str(run_dir / "attack_results_partial.csv"),
        "progress_summary_path": str(run_dir / "progress_summary.json"),
        "summary_path": str(run_dir / "summary.json"),
    }


def save_config_snapshot(run_dir: str | Path, config: Mapping[str, Any]) -> str:
    """Persist the exact experiment config."""
    target = Path(run_dir) / "config_snapshot.yaml"
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False, allow_unicode=True)
    return str(target)


def save_command(run_dir: str | Path, command: Sequence[str] | str | None = None) -> str:
    """Persist the invocation command line."""
    target = Path(run_dir) / "command.txt"
    if command is None:
        value = " ".join(sys.argv)
    elif isinstance(command, str):
        value = command
    else:
        value = " ".join(command)
    target.write_text(value.strip() + "\n", encoding="utf-8")
    return str(target)


def save_random_seed(run_dir: str | Path, seed: int | Sequence[int]) -> str:
    """Persist the exact scalar seed or ordered seed set used by a run."""
    target = Path(run_dir) / "random_seed.json"
    if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes, bytearray)):
        payload = {"random_seeds": [int(value) for value in seed]}
    else:
        payload = {"random_seed": int(seed)}
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(target)


def save_run_log(run_dir: str | Path, message: str, *, append: bool = True) -> str:
    """Write a non-empty textual run log entry."""
    entry = str(message).strip()
    if not entry:
        raise ValueError("Run log entries cannot be empty.")
    target = Path(run_dir) / "logs" / "run.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as handle:
        handle.write(entry + "\n")
    return str(target)


def save_git_commit(run_dir: str | Path, repo_root: str | Path | None = None) -> str:
    """Persist the current git commit, or `unknown` outside a repo."""
    repo = Path(repo_root or ".").resolve()
    target = Path(run_dir) / "git_commit.txt"

    commit = "unknown"
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        candidate = completed.stdout.strip()
        if completed.returncode == 0 and candidate:
            commit = candidate
    except Exception:
        commit = "unknown"

    target.write_text(commit + "\n", encoding="utf-8")
    return str(target)


def save_environment(run_dir: str | Path) -> str:
    """Persist the runtime environment summary."""
    target = Path(run_dir) / "environment.txt"
    lines = [
        f"python_version: {sys.version.replace(os.linesep, ' ')}",
        f"platform: {platform.platform()}",
        f"numpy_version: {_safe_import_version('numpy')}",
        f"pandas_version: {_safe_import_version('pandas')}",
        f"torch_version: {_safe_import_version('torch')}",
        f"sklearn_version: {_safe_import_version('sklearn')}",
        f"matplotlib_version: {_safe_import_version('matplotlib')}",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(target)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refresh_required_artifact_bindings(run_dir: str | Path) -> None:
    """Refresh identity digests after config, seed, and manifest are finalized."""
    root = Path(run_dir)
    target = root / "artifact_bindings.json"
    if not target.is_file():
        return
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact_bindings.json must contain an object")
    payload["required_artifact_sha256"] = {
        name: _sha256_file(root / name) if (root / name).is_file() else ""
        for name in IDENTITY_BOUND_REQUIRED_FILES
    }
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_dataset_manifest(
    run_dir: str | Path,
    dataset_entries: Iterable[Mapping[str, Any]],
) -> str:
    """
    Persist dataset metadata used by the run.

    Each entry should provide at least `image_path`, `dataset`, `split`, and
    `category`. Width, height, and sha256 are auto-derived when omitted.
    """
    target = Path(run_dir) / "dataset_manifest.csv"
    fieldnames = [
        "image_id",
        "image_path",
        "dataset",
        "split",
        "category",
        "width",
        "height",
        "sha256",
    ]

    normalized_rows: List[Dict[str, Any]] = []
    for entry in dataset_entries:
        image_path = Path(str(entry["image_path"]))
        with Image.open(image_path) as image:
            width, height = image.size
        normalized_rows.append(
            {
                "image_id": entry.get("image_id", image_path.stem),
                "image_path": str(image_path),
                "dataset": entry.get("dataset", "unknown"),
                "split": entry.get("split", "unknown"),
                "category": entry.get("category", "unknown"),
                "width": entry.get("width", width),
                "height": entry.get("height", height),
                "sha256": entry.get("sha256", _sha256_file(image_path)),
            }
        )

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)
    return str(target)


def save_summary(run_dir: str | Path, summary: Mapping[str, Any]) -> str:
    """Persist the run-level summary."""
    target = Path(run_dir) / "summary.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, indent=2)
    _refresh_required_artifact_bindings(run_dir)
    return str(target)


def get_raw_results_path(run_dir: str | Path) -> str:
    """Return the canonical raw results CSV path."""
    return str(Path(run_dir) / "raw_results.csv")


def get_raw_results_partial_path(run_dir: str | Path) -> str:
    """Return the incremental partial raw results CSV path."""
    return str(Path(run_dir) / "raw_results_partial.csv")


def get_progress_summary_path(run_dir: str | Path) -> str:
    """Return the incremental run progress JSON path."""
    return str(Path(run_dir) / "progress_summary.json")


def get_figures_dir(run_dir: str | Path) -> str:
    """Return the canonical figures directory."""
    figures_dir = Path(run_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return str(figures_dir)


def _validate_v3_result_artifacts(root: Path) -> List[str]:
    """Validate and reconcile V3 raw rows when the V3 header is present."""
    raw_path = root / "raw_results.csv"
    if not raw_path.is_file():
        return []
    errors: List[str] = []
    try:
        config_payload = yaml.safe_load(
            (root / "config_snapshot.yaml").read_text(encoding="utf-8")
        )
        runner_config = (
            config_payload.get("runner") if isinstance(config_payload, dict) else None
        )
        is_v3_run = (
            isinstance(runner_config, dict)
            and runner_config.get("entrypoint") == "experiments.run_v3_experiment"
        )
        with raw_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(V3_RESULT_SCHEMA.fieldnames):
                return ["v3_raw_results_schema_mismatch"] if is_v3_run else []
            rows = list(reader)
        if not rows:
            return ["v3_raw_results_empty"]
        typed_rows: List[Dict[str, Any]] = []
        for index, row in enumerate(rows):
            try:
                typed = coerce_v3_csv_record(row)
                validated = validate_record_schema(
                    typed,
                    V3_RESULT_SCHEMA,
                    strict=True,
                ).record
                typed_rows.append(validated)
            except (SchemaValidationError, TypeError, ValueError) as exc:
                errors.append(f"v3_row_{index}_invalid:{exc}")
        if errors:
            return errors
        condition_keys = [str(row["condition_key"]) for row in typed_rows]
        if len(condition_keys) != len(set(condition_keys)):
            errors.append("v3_condition_keys_not_unique")

        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            return errors + ["v3_summary_not_an_object"]
        if summary.get("result_rows") != len(typed_rows):
            errors.append("summary_result_rows_mismatch")
        statuses = [str(row["status"]) for row in typed_rows]
        status_counts = {
            status: statuses.count(status)
            for status in ("DEGRADED", "PARTIAL", "FULL")
        }
        if summary.get("status_counts") != status_counts:
            errors.append("summary_status_counts_mismatch")
        rank = {"FULL": 0, "PARTIAL": 1, "DEGRADED": 2}
        worst_status = max(statuses, key=lambda value: rank[value])
        if summary.get("status") != worst_status:
            errors.append("summary_status_mismatch")
        expected_execution_status = (
            "runtime_error"
            if any(row["execution_status"] == "runtime_error" for row in typed_rows)
            else "success"
        )
        if summary.get("execution_status") != expected_execution_status:
            errors.append("summary_execution_status_mismatch")
        if summary.get("methods") != list(dict.fromkeys(row["method"] for row in typed_rows)):
            errors.append("summary_methods_mismatch")
        if sorted(float(value) for value in summary.get("budgets", [])) != sorted(
            {float(row["budget_limit"]) for row in typed_rows}
        ):
            errors.append("summary_budgets_mismatch")
        if sorted(int(value) for value in summary.get("seeds", [])) != sorted(
            {int(row["seed"]) for row in typed_rows}
        ):
            errors.append("summary_seeds_mismatch")
        if summary.get("image_count") != len({row["image_id"] for row in typed_rows}):
            errors.append("summary_image_count_mismatch")
        manifest_digests = {str(row["manifest_digest"]) for row in typed_rows}
        if len(manifest_digests) != 1 or summary.get("manifest_digest") not in manifest_digests:
            errors.append("summary_manifest_digest_mismatch")
        expected_row_count = (
            int(summary.get("image_count", 0))
            * len(summary.get("methods", []))
            * len(summary.get("budgets", []))
            * len(summary.get("seeds", []))
        )
        if len(typed_rows) != expected_row_count:
            errors.append("v3_condition_matrix_incomplete")
        if len(manifest_digests) == 1:
            manifest_digest = next(iter(manifest_digests))
            expected_keys = {
                compute_v3_condition_key(
                    manifest_digest=manifest_digest,
                    image_id=image_id,
                    method=method,
                    budget_limit=float(budget),
                    seed=int(seed),
                )
                for image_id in {str(row["image_id"]) for row in typed_rows}
                for method in summary.get("methods", [])
                for budget in summary.get("budgets", [])
                for seed in summary.get("seeds", [])
            }
            if set(condition_keys) != expected_keys:
                errors.append("v3_condition_matrix_identity_mismatch")

        with (root / "dataset_manifest.csv").open(
            "r", newline="", encoding="utf-8"
        ) as handle:
            manifest_image_ids = {
                str(row.get("image_id", "")) for row in csv.DictReader(handle)
            }
        if manifest_image_ids != {str(row["image_id"]) for row in typed_rows}:
            errors.append("v3_dataset_manifest_image_ids_mismatch")
        seed_payload = json.loads((root / "random_seed.json").read_text(encoding="utf-8"))
        saved_seeds = seed_payload.get("random_seeds")
        if saved_seeds != summary.get("seeds"):
            errors.append("v3_random_seed_artifact_mismatch")
        if not isinstance(runner_config, dict):
            errors.append("v3_runner_config_snapshot_missing")
        else:
            for field in ("methods", "budgets", "seeds", "manifest_digest"):
                expected_value = summary.get(field)
                if field == "manifest_digest":
                    expected_value = summary.get("manifest_digest")
                if runner_config.get(field) != expected_value:
                    errors.append(f"v3_runner_config_{field}_mismatch")

        bindings = json.loads((root / "artifact_bindings.json").read_text(encoding="utf-8"))
        if "FULL" in statuses and (
            not bindings.get("label_artifact_path")
            or not bindings.get("label_audit_path")
        ):
            errors.append("v3_full_rows_require_bound_label_and_audit_artifacts")
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        errors.append(f"v3_artifact_parse_failure:{exc}")
    return errors


def validate_paper_run_artifacts(run_dir: str | Path) -> Dict[str, Any]:
    """Validate that a paper run directory contains all mandatory artifacts."""
    root = Path(run_dir)
    missing_files = [name for name in REQUIRED_ARTIFACT_FILES if not (root / name).exists()]
    missing_dirs = [name for name in REQUIRED_ARTIFACT_DIRS if not (root / name).is_dir()]
    missing_log_files = []
    for name in REQUIRED_LOG_FILES:
        path = root / "logs" / name
        if not path.is_file() or path.stat().st_size == 0:
            missing_log_files.append(f"logs/{name}")
    missing_bound_artifacts: List[str] = []
    digest_mismatches: List[str] = []
    bindings_path = root / "artifact_bindings.json"
    if bindings_path.is_file():
        try:
            bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
            if not isinstance(bindings, dict):
                raise ValueError("artifact bindings must be a JSON object")
            expected_fields = {
                "label_artifact_path",
                "label_artifact_sha256",
                "label_audit_path",
                "label_audit_sha256",
                "split_artifact_path",
                "split_artifact_sha256",
                "required_artifact_sha256",
            }
            source_manifest_fields = {
                "dataset_manifest_source_path",
                "dataset_manifest_source_sha256",
            }
            if set(bindings) not in (
                expected_fields,
                expected_fields | source_manifest_fields,
            ):
                raise ValueError("artifact bindings fields are not exact")
            bound_pairs = [
                ("label_artifact_path", "label_artifact_sha256"),
                ("label_audit_path", "label_audit_sha256"),
                ("split_artifact_path", "split_artifact_sha256"),
            ]
            if source_manifest_fields <= set(bindings):
                bound_pairs.append(
                    (
                        "dataset_manifest_source_path",
                        "dataset_manifest_source_sha256",
                    )
                )
            for path_field, digest_field in bound_pairs:
                value = bindings[path_field]
                stored_digest = bindings[digest_field]
                if not value:
                    if stored_digest:
                        digest_mismatches.append(f"{digest_field}:path_missing")
                    continue
                candidate = Path(str(value))
                resolved = candidate if candidate.is_absolute() else root / candidate
                if not resolved.is_file():
                    missing_bound_artifacts.append(str(resolved.resolve()))
                elif stored_digest != _sha256_file(resolved):
                    digest_mismatches.append(str(resolved.resolve()))
            required_digests = bindings["required_artifact_sha256"]
            if (
                not isinstance(required_digests, dict)
                or set(required_digests) != set(IDENTITY_BOUND_REQUIRED_FILES)
            ):
                raise ValueError("required artifact digest fields are not exact")
            for name in IDENTITY_BOUND_REQUIRED_FILES:
                path = root / name
                stored_digest = required_digests[name]
                if not path.is_file():
                    continue
                if stored_digest != _sha256_file(path):
                    digest_mismatches.append(name)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            missing_bound_artifacts.append(f"artifact_bindings_invalid:{exc}")
    v3_result_errors = _validate_v3_result_artifacts(root)
    is_valid = (
        not missing_files
        and not missing_dirs
        and not missing_log_files
        and not missing_bound_artifacts
        and not digest_mismatches
        and not v3_result_errors
    )
    return {
        "run_dir": str(root),
        "is_valid": is_valid,
        "missing_files": missing_files,
        "missing_dirs": missing_dirs,
        "missing_log_files": missing_log_files,
        "missing_bound_artifacts": missing_bound_artifacts,
        "digest_mismatches": digest_mismatches,
        "v3_result_errors": v3_result_errors,
    }
