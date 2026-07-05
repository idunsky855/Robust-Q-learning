from __future__ import annotations

from pathlib import Path

from ears_q_learning.config import LearningConfig
from ears_q_learning.state_space import encode_state, fit_thresholds
from ears_q_learning.types import CountryYearRow
from ears_q_learning.weighting_sensitivity import (
    run_weighting_sensitivity,
    write_weighting_sensitivity_artifacts,
)


def _row(year: int) -> CountryYearRow:
    return CountryYearRow(
        country="Aland",
        year=year,
        resistance_3gc=10.0 + year - 2015,
        resistance_fq=20.0 + year - 2015,
        resistance_carb=0.0,
        tested_3gc=100 + year,
        tested_fq=110 + year,
        tested_carb=120 + year,
    )


def test_tested_weighting_runs_full_protocol_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    rows = [_row(year) for year in range(2015, 2025)]
    training_rows = rows[:5]
    thresholds = fit_thresholds(training_rows)
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

    result = run_weighting_sensitivity(
        rows=rows,
        training_rows=training_rows,
        state_lookup=lookup,
        learning=learning,
        training_year_end=2019,
        decision_year_start=2020,
        outcome_year_end=2024,
        smoothing_gamma=0.1,
        carbapenem_penalty=0.1,
        weighting="tested",
    )

    assert result["weighting"] == "tested"
    assert result["evaluation"]["weighting"] == "tested"
    paths = write_weighting_sensitivity_artifacts(tmp_path, result)
    assert all(Path(path).exists() for path in paths.values())
