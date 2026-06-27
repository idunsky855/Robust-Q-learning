from __future__ import annotations

import json
from pathlib import Path

from ears_q_learning.config import load_config
from ears_q_learning.pipeline import run_pipeline


def test_pipeline_writes_blocked_status_without_raw_snapshot(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  name: Test",
                "  random_seed: 1",
                "paths:",
                "  raw_snapshot: data/raw/missing.csv",
                "  raw_snapshot_metadata: data/raw/missing.json",
                "  processed_dir: data/processed",
                "  results_dir: results",
                "data:",
                "  organism: Escherichia coli",
                "  training_year_start: 2015",
                "  training_year_end: 2019",
                "  evaluation_year_start: 2020",
                "  evaluation_year_end: 2024",
                "  minimum_training_transitions: 3",
                "  minimum_evaluation_transitions: 2",
                "  carbapenem_penalty: 0.1",
                "  smoothing_gamma: 0.1",
                "  weighting: equal",
                "learning:",
                "  discount_grid: [0.3]",
                "  exploration_grid: [0.1]",
                "  updates: 10",
                "  q_norm: 1",
                "  epsilon_multipliers: [1.0]",
                "  tuning_seeds: [1]",
                "  final_seeds: [2]",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    result = run_pipeline(config)
    assert result["status"] == "blocked_missing_raw_snapshot"


def test_pipeline_blocks_when_snapshot_metadata_is_missing(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "data" / "raw" / "present.csv"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        "\n".join(
            [
                "country,year,organism,antibiotic,resistance_percentage,tested_count",
                "Aland,2015,Escherichia coli,Cefotaxime,10,100",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  name: Test",
                "  random_seed: 1",
                "paths:",
                "  raw_snapshot: data/raw/present.csv",
                "  raw_snapshot_metadata: data/raw/present.metadata.json",
                "  processed_dir: data/processed",
                "  results_dir: results",
                "data:",
                "  organism: Escherichia coli",
                "  training_year_start: 2015",
                "  training_year_end: 2019",
                "  evaluation_year_start: 2020",
                "  evaluation_year_end: 2024",
                "  minimum_training_transitions: 3",
                "  minimum_evaluation_transitions: 2",
                "  carbapenem_penalty: 0.1",
                "  smoothing_gamma: 0.1",
                "  weighting: equal",
                "learning:",
                "  discount_grid: [0.3]",
                "  exploration_grid: [0.1]",
                "  updates: 10",
                "  q_norm: 1",
                "  epsilon_multipliers: [1.0]",
                "  tuning_seeds: [1]",
                "  final_seeds: [2]",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    result = run_pipeline(config)
    assert result["status"] == "blocked_missing_raw_snapshot_metadata"
