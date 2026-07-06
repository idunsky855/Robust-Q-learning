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
    cost_input: Path | None
    site_dir: Path | None


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
    carbapenem_penalty_sensitivity: tuple[float, ...]
    stewardship_breadth_scores: dict[str, float]
    stewardship_beta_grid: tuple[float, ...]
    stewardship_delta_grid: tuple[float, ...]
    stewardship_training_scenarios: tuple[tuple[float, float], ...]
    economic_gamma_grid: tuple[float, ...]
    economic_training_scenario: tuple[float, float, float] | None
    smoothing_gamma: float
    weighting: str
    weighting_sensitivity: tuple[str, ...]


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
            cost_input=(
                resolve(path_section["cost_input"])
                if "cost_input" in path_section
                else None
            ),
            site_dir=(
                resolve(path_section["site_dir"])
                if "site_dir" in path_section
                else None
            ),
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
            carbapenem_penalty_sensitivity=tuple(
                float(value)
                for value in data_section.get(
                    "carbapenem_penalty_sensitivity",
                    (data_section["carbapenem_penalty"],),
                )
            ),
            stewardship_breadth_scores={
                str(key): float(value)
                for key, value in data_section.get(
                    "stewardship_breadth_scores",
                    {"3gc": 0.4, "fq": 0.6, "carb": 1.0},
                ).items()
            },
            stewardship_beta_grid=tuple(
                float(value)
                for value in data_section.get("stewardship_beta_grid", (0.0,))
            ),
            stewardship_delta_grid=tuple(
                float(value)
                for value in data_section.get("stewardship_delta_grid", (0.0,))
            ),
            stewardship_training_scenarios=tuple(
                (float(item["beta"]), float(item["delta"]))
                for item in data_section.get("stewardship_training_scenarios", ())
            ),
            economic_gamma_grid=tuple(
                float(value)
                for value in data_section.get("economic_gamma_grid", (0.0,))
            ),
            economic_training_scenario=(
                (
                    float(data_section["economic_training_scenario"]["beta"]),
                    float(data_section["economic_training_scenario"]["gamma"]),
                    float(data_section["economic_training_scenario"]["delta"]),
                )
                if "economic_training_scenario" in data_section
                else None
            ),
            smoothing_gamma=float(_require(data_section, "smoothing_gamma")),
            weighting=str(_require(data_section, "weighting")),
            weighting_sensitivity=tuple(
                str(value)
                for value in data_section.get("weighting_sensitivity", ())
            ),
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
