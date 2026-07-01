from __future__ import annotations

import numpy as np

from ears_q_learning.evaluation import evaluate_policy, run_policy_evaluation
from ears_q_learning.types import CountryYearRow


def _rows() -> list[CountryYearRow]:
    rows: list[CountryYearRow] = []
    for country, offset in (("Aland", 0.0), ("Borduria", 5.0)):
        for year in range(2020, 2025):
            rows.append(
                CountryYearRow(
                    country=country,
                    year=year,
                    resistance_3gc=20.0 + offset + year - 2020,
                    resistance_fq=30.0 + offset + year - 2020,
                    resistance_carb=1.0,
                    tested_3gc=100,
                    tested_fq=100,
                    tested_carb=100,
                )
            )
    return rows


def _training_output(policy: list[int]) -> dict[str, object]:
    return {
        "robust_epsilon": 0.0,
        "seed_results": [{"seed": 7, "policy": policy}],
        "state_summaries": [
            {"state": state, "modal_action": action}
            for state, action in enumerate(policy)
        ],
    }


def test_evaluation_uses_four_one_year_ahead_transitions_per_country() -> None:
    rows = _rows()
    lookup = {(row.country, row.year): 0 for row in rows}
    result = evaluate_policy(
        name="fixed_3gc",
        policy=np.zeros(8, dtype=int),
        rows=rows,
        state_lookup=lookup,
        carbapenem_penalty=0.1,
    )

    assert result["transition_count"] == 8
    assert [item["decision_year"] for item in result["yearly_performance"]] == [
        2020,
        2021,
        2022,
        2023,
    ]
    assert result["carbapenem_use_rate"] == 0.0
    assert result["mean_raw_susceptibility"] == result["mean_adjusted_coverage"]
    assert result["worst_country"] == "Borduria"


def test_complete_evaluation_contains_learned_policies_and_baselines() -> None:
    rows = _rows()
    lookup = {(row.country, row.year): 0 for row in rows}
    classical = _training_output([0] * 8)
    robust = _training_output([2] * 8)
    robust["robust_epsilon"] = 0.05
    result = run_policy_evaluation(
        rows=rows,
        state_lookup=lookup,
        training_summary={
            "classical": classical,
            "robust": {"radii": [robust]},
        },
        myopic_policy=np.zeros(8, dtype=int),
        carbapenem_penalty=0.1,
    )

    assert result["decision_years"] == [2020, 2021, 2022, 2023]
    assert result["outcome_years"] == [2021, 2022, 2023, 2024]
    assert len(result["learned_policies"]) == 2
    assert {item["name"] for item in result["baselines"]} == {
        "myopic",
        "fixed_3gc",
        "fixed_fq",
        "fixed_carb",
    }
    assert result["hindsight_oracle"]["deployable_policy"] is False
    assert "standard_deviation" in result["learned_policies"][0][
        "endpoint_summaries"
    ]["mean_adjusted_coverage"]
