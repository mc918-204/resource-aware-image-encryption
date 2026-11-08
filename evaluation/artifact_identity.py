"""Deterministic source-tree identity artifacts for research runs without Git."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


TRACKED_ROOTS = ("src", "experiments", "configs", "mobile-benchmark")
TRACKED_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".kt",
    ".kts",
    ".xml",
    ".properties",
    ".jar",
}
TRACKED_NAMES = {"gradlew", "gradlew.bat"}
EXCLUDED_PARTS = {"__pycache__", ".git", ".gradle", "build", "results"}


def sha256_file(path: str | Path) -> str:
    """Return the full SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_files(root: Path) -> Iterable[Path]:
    """Yield source-like files under the frozen set of tracked roots."""
    for root_name in TRACKED_ROOTS:
        source_root = root / root_name
        if not source_root.exists():
            continue
        for path in source_root.rglob("*"):
            relative = path.relative_to(root)
            if path.is_file() and not (set(relative.parts) & EXCLUDED_PARTS):
                if (
                    path.suffix.lower() in TRACKED_SUFFIXES
                    or path.name in TRACKED_NAMES
                ):
                    yield path


def build_source_tree_manifest(root: str | Path) -> dict[str, object]:
    """Build a canonical manifest and digest for tracked source files."""
    base = Path(root).resolve()
    if not base.exists():
        raise FileNotFoundError(f"source tree root does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"source tree root is not a directory: {base}")
    if not any((base / root_name).is_dir() for root_name in TRACKED_ROOTS):
        raise ValueError(f"source tree root has no tracked source directory: {base}")

    tracked_paths = sorted(
        _tracked_files(base),
        key=lambda candidate: candidate.relative_to(base).as_posix(),
    )
    if not tracked_paths:
        raise ValueError(f"source tree root has no tracked source files: {base}")
    files = [
        {
            "path": path.relative_to(base).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in tracked_paths
    ]
    payload = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "artifact_version": "source_tree_manifest_v1",
        "source_tree_digest": hashlib.sha256(payload).hexdigest(),
        "files": files,
    }


def write_source_tree_manifest(
    root: str | Path,
    output: str | Path,
) -> dict[str, object]:
    """Write a repeatable source-tree JSON artifact and return its manifest."""
    manifest = build_source_tree_manifest(root)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return manifest
