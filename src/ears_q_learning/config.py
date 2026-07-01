"""Configuration loading for the EARS-Net Q-learning project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Project-level settings."""

    name: str
    random_seed: int


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths used by the pipeline."""

    raw_snapshot: Path | None
    raw_snapshot_metadata: Path | None
    raw_snapshots: tuple[tuple[Path, Path], ...]
    processed_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class DataConfig:
    """Data and MDP settings."""

    organism: str
    training_year_start: int
    training_year_end: int
    evaluation_year_start: int
    evaluation_year_end: int
    minimum_training_transitions: int
    minimum_evaluation_transitions: int
    carbapenem_penalty: float
    smoothing_gamma: float
    weighting: str


@dataclass(frozen=True)
class LearningConfig:
    """Learning and tuning settings."""

    discount_grid: tuple[float, ...]
    exploration_grid: tuple[float, ...]
    updates: int
    q_norm: int
    epsilon_multipliers: tuple[float, ...]
    tuning_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]


@dataclass(frozen=True)
class Config:
    """Complete project configuration."""

    project: ProjectConfig
    paths: PathConfig
    data: DataConfig
    learning: LearningConfig
    config_path: Path


def _require(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise KeyError(f"Missing configuration key: {key}")
    return section[key]


def load_config(path: Path) -> Config:
    """Load a YAML configuration file into typed dataclasses."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")

    project_section = _require(raw, "project")
    path_section = _require(raw, "paths")
    data_section = _require(raw, "data")
    learning_section = _require(raw, "learning")

    resolved_path = path.resolve()
    if resolved_path.parent.name == "configs":
        base_dir = resolved_path.parent.parent
    else:
        base_dir = resolved_path.parent

    def resolve(value: str) -> Path:
        return (base_dir / value).resolve()

    return Config(
        project=ProjectConfig(
            name=_require(project_section, "name"),
            random_seed=int(_require(project_section, "random_seed")),
        ),
        paths=PathConfig(
            raw_snapshot=(
                resolve(path_section["raw_snapshot"])
                if "raw_snapshot" in path_section
                else None
            ),
            raw_snapshot_metadata=(
                resolve(path_section["raw_snapshot_metadata"])
                if "raw_snapshot_metadata" in path_section
                else None
            ),
            raw_snapshots=tuple(
                (
                    resolve(str(snapshot["path"])),
                    resolve(str(snapshot["metadata"])),
                )
                for snapshot in path_section.get("raw_snapshots", ())
            ),
            processed_dir=resolve(_require(path_section, "processed_dir")),
            results_dir=resolve(_require(path_section, "results_dir")),
        ),
        data=DataConfig(
            organism=_require(data_section, "organism"),
            training_year_start=int(_require(data_section, "training_year_start")),
            training_year_end=int(_require(data_section, "training_year_end")),
            evaluation_year_start=int(_require(data_section, "evaluation_year_start")),
            evaluation_year_end=int(_require(data_section, "evaluation_year_end")),
            minimum_training_transitions=int(
                _require(data_section, "minimum_training_transitions")
            ),
            minimum_evaluation_transitions=int(
                _require(data_section, "minimum_evaluation_transitions")
            ),
            carbapenem_penalty=float(_require(data_section, "carbapenem_penalty")),
            smoothing_gamma=float(_require(data_section, "smoothing_gamma")),
            weighting=str(_require(data_section, "weighting")),
        ),
        learning=LearningConfig(
            discount_grid=tuple(_require(learning_section, "discount_grid")),
            exploration_grid=tuple(_require(learning_section, "exploration_grid")),
            updates=int(_require(learning_section, "updates")),
            q_norm=int(_require(learning_section, "q_norm")),
            epsilon_multipliers=tuple(_require(learning_section, "epsilon_multipliers")),
            tuning_seeds=tuple(_require(learning_section, "tuning_seeds")),
            final_seeds=tuple(_require(learning_section, "final_seeds")),
        ),
        config_path=resolved_path,
    )
