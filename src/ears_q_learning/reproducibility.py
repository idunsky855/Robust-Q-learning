"""Utilities that support deterministic and auditable runs."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set global RNG seeds for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_versions(packages: list[str]) -> dict[str, str]:
    """Capture installed dependency versions when available."""
    versions: dict[str, str] = {}
    for package in packages:
        try:
            module = importlib.import_module(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[package] = "not-installed"
    return versions


def to_serializable(value: Any) -> Any:
    """Convert common Python objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: to_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_run_metadata(config: Any, run_directory: Path) -> dict[str, Any]:
    """Create machine-readable run metadata."""
    return {
        "project_name": getattr(config.project, "name"),
        "run_directory": str(run_directory),
        "config_path": str(config.config_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependency_versions": dependency_versions(["numpy", "yaml", "pytest"]),
        "random_seed": getattr(config.project, "random_seed"),
        "config": to_serializable(config),
    }
