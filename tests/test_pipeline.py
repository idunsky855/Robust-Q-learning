from __future__ import annotations

import json
from pathlib import Path

from ears_q_learning.config import load_config
from ears_q_learning.reproducibility import sha256_file
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


def test_pipeline_writes_preprocessing_artifacts_when_inputs_exist(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = raw_dir / "present.csv"
    snapshot_path.write_text(
        "\n".join(
            [
                "country,year,organism,antibiotic,resistance_percentage,tested_count",
                "Aland,2015,Escherichia coli,Cefotaxime,10,100",
                "Aland,2015,Escherichia coli,Ciprofloxacin,20,100",
                "Aland,2015,Escherichia coli,Meropenem,0,100",
                "Aland,2016,Escherichia coli,Cefotaxime,30,110",
                "Aland,2016,Escherichia coli,Ciprofloxacin,40,110",
                "Aland,2016,Escherichia coli,Meropenem,1,110",
                "Aland,2017,Escherichia coli,Cefotaxime,35,115",
                "Aland,2017,Escherichia coli,Ciprofloxacin,45,115",
                "Aland,2017,Escherichia coli,Meropenem,1,115",
                "Aland,2018,Escherichia coli,Cefotaxime,40,120",
                "Aland,2018,Escherichia coli,Ciprofloxacin,50,120",
                "Aland,2018,Escherichia coli,Meropenem,2,120",
                "Aland,2019,Escherichia coli,Cefotaxime,45,125",
                "Aland,2019,Escherichia coli,Ciprofloxacin,55,125",
                "Aland,2019,Escherichia coli,Meropenem,2,125",
                "Aland,2020,Escherichia coli,Cefotaxime,50,130",
                "Aland,2020,Escherichia coli,Ciprofloxacin,60,130",
                "Aland,2020,Escherichia coli,Meropenem,2,130",
                "Aland,2021,Escherichia coli,Cefotaxime,52,131",
                "Aland,2021,Escherichia coli,Ciprofloxacin,61,131",
                "Aland,2021,Escherichia coli,Meropenem,2,131",
                "Aland,2022,Escherichia coli,Cefotaxime,54,132",
                "Aland,2022,Escherichia coli,Ciprofloxacin,62,132",
                "Aland,2022,Escherichia coli,Meropenem,2,132",
                "Aland,2023,Escherichia coli,Cefotaxime,55,133",
                "Aland,2023,Escherichia coli,Ciprofloxacin,63,133",
                "Aland,2023,Escherichia coli,Meropenem,2,133",
                "Aland,2024,Escherichia coli,Cefotaxime,56,134",
                "Aland,2024,Escherichia coli,Ciprofloxacin,64,134",
                "Aland,2024,Escherichia coli,Meropenem,2,134",
                "Borduria,2015,Escherichia coli,Cefotaxime,12,100",
                "Borduria,2015,Escherichia coli,Ciprofloxacin,22,100",
                "Borduria,2015,Escherichia coli,Meropenem,0,100",
                "Borduria,2016,Escherichia coli,Cefotaxime,14,101",
                "Borduria,2016,Escherichia coli,Ciprofloxacin,24,101",
                "Borduria,2016,Escherichia coli,Meropenem,0,101",
                "Borduria,2017,Escherichia coli,Cefotaxime,16,102",
                "Borduria,2017,Escherichia coli,Ciprofloxacin,26,102",
                "Borduria,2017,Escherichia coli,Meropenem,0,102",
                "Borduria,2018,Escherichia coli,Cefotaxime,18,103",
                "Borduria,2018,Escherichia coli,Ciprofloxacin,28,103",
                "Borduria,2018,Escherichia coli,Meropenem,0,103",
                "Borduria,2019,Escherichia coli,Cefotaxime,20,104",
                "Borduria,2019,Escherichia coli,Ciprofloxacin,30,104",
                "Borduria,2019,Escherichia coli,Meropenem,0,104",
                "Borduria,2020,Escherichia coli,Cefotaxime,21,105",
                "Borduria,2020,Escherichia coli,Ciprofloxacin,31,105",
                "Borduria,2020,Escherichia coli,Meropenem,0,105",
                "Borduria,2021,Escherichia coli,Cefotaxime,22,106",
                "Borduria,2021,Escherichia coli,Ciprofloxacin,32,106",
                "Borduria,2021,Escherichia coli,Meropenem,0,106",
                "Borduria,2022,Escherichia coli,Cefotaxime,23,107",
                "Borduria,2022,Escherichia coli,Ciprofloxacin,33,107",
                "Borduria,2022,Escherichia coli,Meropenem,0,107",
                "Borduria,2023,Escherichia coli,Cefotaxime,24,108",
                "Borduria,2023,Escherichia coli,Ciprofloxacin,34,108",
                "Borduria,2023,Escherichia coli,Meropenem,0,108",
                "Borduria,2024,Escherichia coli,Cefotaxime,25,109",
                "Borduria,2024,Escherichia coli,Ciprofloxacin,35,109",
                "Borduria,2024,Escherichia coli,Meropenem,0,109",
            ]
        ),
        encoding="utf-8",
    )
    metadata_path = raw_dir / "present.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_url": "https://example.test",
                "retrieval_date": "2026-06-27",
                "selected_filters": {"pathogen": "Escherichia coli"},
            }
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
    assert result["status"] == "scaffold_completed"
    processed_dir = tmp_path / "data" / "processed"
    assert (processed_dir / "preprocessing_report.json").exists()
    assert (processed_dir / "state_assignments.json").exists()
    assert (processed_dir / "training_summary.json").exists()
    assert (processed_dir / "evaluation_metrics.json").exists()
    assert (processed_dir / "penalty_sensitivity.json").exists()
    assert (processed_dir / "penalty_sensitivity.csv").exists()
    assert (processed_dir / "penalty_vs_carbapenem_use.svg").exists()
    assert (processed_dir / "susceptibility_vs_carbapenem_use.svg").exists()
    assert (processed_dir / "stewardship_reward_scenario.json").exists()
    assert (processed_dir / "stewardship_reward_scenario.csv").exists()
    assert (processed_dir / "stewardship_reward_carbapenem_use.svg").exists()
    assert (processed_dir / "stewardship_full_training.json").exists()
    assert (processed_dir / "stewardship_full_training_summary.csv").exists()
    assert (processed_dir / "final_results_table.csv").exists()
    transition_model = json.loads(
        (processed_dir / "transition_model.json").read_text(encoding="utf-8")
    )
    assert transition_model["action_independent"] is True
    assert len(transition_model["annual_distances"]) == 4
    assert all(
        abs(sum(row) - 1.0) < 1e-12
        for row in transition_model["reference_kernel"]
    )


def test_pipeline_accepts_three_atlas_exports(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    antibiotic_exports = {
        "carb": ("Carbapenems", 0.0),
        "fq": ("Fluoroquinolones", 20.0),
        "3gc": ("Third-generation cephalosporins", 10.0),
    }
    snapshot_entries: list[str] = []
    for index, (code, (label, base_resistance)) in enumerate(antibiotic_exports.items()):
        path = raw_dir / f"{code}.csv"
        lines = [
            '"HealthTopic","Population","Indicator","Unit","Time","RegionCode","RegionName","NumValue","TxtValue"'
        ]
        for country_code, country_offset in {"AA": 0.0, "BB": 2.0}.items():
            country = "Aland" if country_code == "AA" else "Borduria"
            for year in range(2015, 2025):
                resistance = base_resistance + country_offset + (year - 2015)
                tested = 100 + index + (year - 2015)
                lines.extend(
                    [
                        f'"Antimicrobial resistance","Escherichia coli|{label}","R - resistant isolates, percentage","%","{year}","{country_code}","{country}",{resistance:.1f},""',
                        f'"Antimicrobial resistance","Escherichia coli|{label}","Total tested isolates","N","{year}","{country_code}","{country}",{tested:.1f},""',
                    ]
                )
        path.write_text("\n".join(lines), encoding="utf-8")
        metadata_path = raw_dir / f"{code}.metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "source_url": "https://atlas.ecdc.europa.eu/public/",
                    "retrieval_date": "2026-07-01",
                    "selected_filters": {"subpopulation": label},
                    "sha256": sha256_file(path),
                }
            ),
            encoding="utf-8",
        )
        snapshot_entries.extend(
            [
                f"    - path: data/raw/{code}.csv",
                f"      metadata: data/raw/{code}.metadata.json",
            ]
        )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  name: Test",
                "  random_seed: 1",
                "paths:",
                "  raw_snapshots:",
                *snapshot_entries,
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

    result = run_pipeline(load_config(config_path))

    assert result["status"] == "scaffold_completed"
    report = json.loads(
        (tmp_path / "data" / "processed" / "raw_snapshot_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["record_count"] == 60
    assert report["metadata"]["source_format"] == "ecdc_atlas_long"
    assert len(report["metadata"]["snapshots"]) == 3
    assert (tmp_path / "data" / "processed" / "training_summary.json").exists()
