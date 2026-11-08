"""Reproducibility metadata shared by training, evaluation, and readiness."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping


def compute_file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_sha(repo_root: str | Path | None = None) -> str:
    """Return the current git commit, or ``unknown`` outside a repository."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root or Path.cwd()), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def dataset_provenance(
    dataset_path: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe a prepared dataset and the split it came from."""
    path = Path(dataset_path)
    manifest = manifest or {}
    manifest_path = path.parent / "manifest.json"
    return {
        "dataset_path": str(path),
        "dataset_sha256": compute_file_sha256(path) if path.is_file() else "unknown",
        "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
        "split_seed": manifest.get("seed"),
        "train_episode_ids": manifest.get("train_episode_ids"),
        "validation_episode_ids": manifest.get("validation_episode_ids"),
        "next_observations_source": manifest.get("next_observations_source"),
        "source_provenance": manifest.get("provenance"),
    }
