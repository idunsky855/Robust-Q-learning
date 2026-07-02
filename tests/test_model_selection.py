from __future__ import annotations

from ears_q_learning.config import LearningConfig
from ears_q_learning.model_selection import (
    build_training_context,
    final_seeded_training,
    tune_configuration,
)
from ears_q_learning.types import CountryYearRow


def _stable_rows() -> list[CountryYearRow]:
    rows: list[CountryYearRow] = []
    for country in ("Aland", "Borduria"):
        for year in range(2015, 2020):
            rows.append(
                CountryYearRow(
                    country=country,
                    year=year,
                    resistance_3gc=10.0,
                    resistance_fq=10.0,
                    resistance_carb=0.0,
                    tested_3gc=100,
                    tested_fq=100,
                    tested_carb=100,
                )
            )
    return rows


def _learning_config() -> LearningConfig:
    return LearningConfig(
        discount_grid=(0.3, 0.45),
        exploration_grid=(0.1, 0.2),
        updates=1,
        q_norm=1,
        epsilon_multipliers=(1.0,),
        tuning_seeds=(1, 2),
        final_seeds=(3, 4, 5),
    )


def test_tuning_tie_breaks_toward_paper_defaults() -> None:
    selected = tune_configuration(
        rows=_stable_rows(),
        learning=_learning_config(),
        smoothing_gamma=0.1,
        carbapenem_penalty=0.1,
        weighting="equal",
        robust=False,
    )

    assert selected.discount == 0.45
    assert selected.exploration_rate == 0.1
    assert len(selected.selection_trace) == 4
    assert selected.selection_trace[0]["rank"] == 1


def test_final_seeded_training_reports_seed_and_state_summaries() -> None:
    learning = _learning_config()
    context = build_training_context(
        rows=_stable_rows(),
        training_year_end=2019,
        smoothing_gamma=0.1,
        carbapenem_penalty=0.1,
        weighting="equal",
    )
    selected = tune_configuration(
        rows=_stable_rows(),
        learning=learning,
        smoothing_gamma=0.1,
        carbapenem_penalty=0.1,
        weighting="equal",
        robust=False,
    )

    output = final_seeded_training(
        context=context,
        selected=selected,
        learning=learning,
        robust_epsilon=0.0,
    )

    assert len(output["seed_results"]) == 3
    assert len(output["state_summaries"]) == 8
    assert all("modal_action" in row for row in output["state_summaries"])
    assert len(output["exact_bellman_reference"]["policy"]) == 8
    assert all(
        "bellman_sup_norm_error" in result for result in output["seed_results"]
    )
