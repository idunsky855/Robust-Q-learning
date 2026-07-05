from __future__ import annotations

import numpy as np

from ears_q_learning.config import LearningConfig
from ears_q_learning.learning import solve_bellman_reward_matrix
from ears_q_learning.mdp import normalized_hamming_cost
from ears_q_learning.state_space import encode_state, fit_thresholds
from ears_q_learning.stewardship_reward import build_stewardship_reward_matrix
from ears_q_learning.stewardship_training import (
    build_stewardship_training_context,
    run_full_stewardship_training,
)
from ears_q_learning.types import CountryYearRow


def _row(year: int) -> CountryYearRow:
    return CountryYearRow(
        country="Aland",
        year=year,
        resistance_3gc=20.0,
        resistance_fq=30.0,
        resistance_carb=0.0,
        tested_3gc=100,
        tested_fq=100,
        tested_carb=100,
    )


def test_stewardship_penalty_can_switch_policy_from_carbapenem() -> None:
    rows = [_row(2018), _row(2019)]
    lookup = {(row.country, row.year): 0 for row in rows}
    scores = {"3gc": 0.4, "fq": 0.6, "carb": 1.0}
    nominal = build_stewardship_reward_matrix(rows, lookup, scores, 0.0, 0.0)
    penalized = build_stewardship_reward_matrix(rows, lookup, scores, 0.5, 0.0)
    kernel = np.eye(8)
    cost = normalized_hamming_cost()

    nominal_policy = solve_bellman_reward_matrix(
        kernel, nominal, 0.3, cost
    ).greedy_policy
    penalized_policy = solve_bellman_reward_matrix(
        kernel, penalized, 0.3, cost
    ).greedy_policy

    assert np.all(nominal_policy == 2)
    assert np.all(penalized_policy == 0)


def test_full_stewardship_training_reports_convergence_and_evaluation() -> None:
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

    result = run_full_stewardship_training(
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
        scenarios=((0.15, 0.1),),
    )

    scenario = result["scenarios"][0]
    assert "convergence" in scenario["classical"]["training"]
    assert scenario["classical"]["evaluation"]["exact_policy_metrics"][
        "transition_count"
    ] == 4


def test_training_context_includes_cost_penalty() -> None:
    rows = [_row(year) for year in range(2015, 2020)]
    arguments = {
        "rows": rows,
        "training_year_end": 2019,
        "smoothing_gamma": 0.1,
        "weighting": "equal",
        "breadth_scores": {"3gc": 0.4, "fq": 0.6, "carb": 1.0},
        "beta": 0.15,
        "delta": 0.1,
    }
    no_cost = build_stewardship_training_context(**arguments)
    with_cost = build_stewardship_training_context(
        **arguments,
        cost_scores={"3gc": 0.4, "fq": 0.6, "carb": 1.0},
        gamma=0.025,
    )

    expected = np.asarray([0.01, 0.015, 0.025])
    assert np.allclose(no_cost.reward_matrix - with_cost.reward_matrix, expected)
