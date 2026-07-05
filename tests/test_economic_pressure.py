from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ears_q_learning.config import LearningConfig
from ears_q_learning.economic_pressure import (
    load_normalized_cost_scores,
    run_full_economic_training,
    run_economic_pressure_scenario,
    write_economic_pressure_artifacts,
)
from ears_q_learning.mdp import normalized_hamming_cost
from ears_q_learning.state_space import encode_state, fit_thresholds
from ears_q_learning.types import CountryYearRow


def _row(year: int) -> CountryYearRow:
    return CountryYearRow(
        country="Aland",
        year=year,
        resistance_3gc=10.0,
        resistance_fq=20.0,
        resistance_carb=0.0,
        tested_3gc=100,
        tested_fq=100,
        tested_carb=100,
    )


def _cost_file(path: Path) -> None:
    fields = (
        "action",
        "representative_product",
        "atc_code",
        "route",
        "ddd_grams",
        "cost_per_ddd_gbp",
        "source_url",
        "source_period",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for action, cost in (("3gc", 1.0), ("fq", 2.0), ("carb", 5.0)):
            writer.writerow(
                {
                    "action": action,
                    "representative_product": action,
                    "atc_code": action,
                    "route": "parenteral",
                    "ddd_grams": 1.0,
                    "cost_per_ddd_gbp": cost,
                    "source_url": "https://example.test",
                    "source_period": "test",
                }
            )


def test_economic_pressure_scenario_uses_normalized_costs(tmp_path: Path) -> None:
    cost_path = tmp_path / "costs.csv"
    _cost_file(cost_path)
    costs = load_normalized_cost_scores(cost_path)
    assert costs["normalized_cost_scores"]["carb"] == 1.0

    rows = [_row(year) for year in range(2018, 2022)]
    lookup = {(row.country, row.year): 0 for row in rows}
    training_summary = {
        "classical": {"configuration": {"discount": 0.3}},
        "robust": {
            "selected_configuration": {"discount": 0.3},
            "radii": [{"robust_epsilon": 0.05}],
        },
    }
    analysis = run_economic_pressure_scenario(
        rows=rows,
        training_rows=rows[:2],
        state_lookup=lookup,
        kernel=np.eye(8),
        cost_matrix=normalized_hamming_cost(),
        training_summary=training_summary,
        breadth_scores={"3gc": 0.4, "fq": 0.6, "carb": 1.0},
        beta=0.15,
        delta=0.1,
        gamma_grid=(0.0, 0.1),
        cost_input=cost_path,
        decision_year_start=2020,
        outcome_year_end=2021,
    )
    assert len(analysis["results"]) == 4
    paths = write_economic_pressure_artifacts(tmp_path, analysis)
    assert all(Path(path).exists() for path in paths.values())


def test_full_economic_training_records_cost_scenario(tmp_path: Path) -> None:
    cost_path = tmp_path / "costs.csv"
    _cost_file(cost_path)
    rows = [_row(year) for year in range(2015, 2025)]
    thresholds = fit_thresholds(rows[:5])
    lookup = {(row.country, row.year): encode_state(row, thresholds) for row in rows}
    learning = LearningConfig(
        discount_grid=(0.3,),
        exploration_grid=(0.1,),
        updates=20,
        q_norm=1,
        epsilon_multipliers=(1.0,),
        tuning_seeds=(1,),
        final_seeds=(2,),
    )

    result = run_full_economic_training(
        rows=rows,
        evaluation_state_lookup=lookup,
        training_year_end=2019,
        decision_year_start=2020,
        outcome_year_end=2024,
        learning=learning,
        smoothing_gamma=0.1,
        weighting="equal",
        epsilon_star=0.05,
        breadth_scores={"3gc": 0.4, "fq": 0.6, "carb": 1.0},
        beta=0.15,
        gamma=0.025,
        delta=0.1,
        cost_input=cost_path,
    )

    assert result["status"] == "full_economic_training_completed"
    assert result["gamma"] == 0.025
    assert result["cost_data"]["normalized_cost_scores"]["carb"] == 1.0
