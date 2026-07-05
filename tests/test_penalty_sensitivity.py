from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ears_q_learning.penalty_sensitivity import (
    run_penalty_sensitivity,
    write_penalty_sensitivity_artifacts,
)
from ears_q_learning.types import CountryYearRow


def _row(country: str, year: int, carbapenem_resistance: float) -> CountryYearRow:
    return CountryYearRow(
        country=country,
        year=year,
        resistance_3gc=10.0,
        tested_3gc=100,
        resistance_fq=20.0,
        tested_fq=100,
        resistance_carb=carbapenem_resistance,
        tested_carb=100,
    )


def _training_summary() -> dict[str, object]:
    return {
        "classical": {"configuration": {"discount": 0.3}},
        "robust": {
            "selected_configuration": {"discount": 0.3},
            "radii": [{"robust_epsilon": 0.05}],
        },
    }


def test_penalty_sensitivity_preserves_primary_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    rows = [
        _row("Aland", 2018, 0.0),
        _row("Aland", 2019, 0.0),
        _row("Aland", 2020, 0.0),
        _row("Aland", 2021, 0.0),
    ]
    state_lookup = {(row.country, row.year): 0 for row in rows}
    kernel = np.eye(8)
    analysis = run_penalty_sensitivity(
        rows=rows,
        training_rows=rows[:2],
        state_lookup=state_lookup,
        kernel=kernel,
        cost_matrix=np.ones((8, 8)) - np.eye(8),
        training_summary=_training_summary(),
        primary_penalty=0.1,
        penalties=(0.2, 0.1),
        decision_year_start=2020,
        outcome_year_end=2021,
    )

    assert analysis["penalty_grid"] == [0.1, 0.2]
    assert len(analysis["results"]) == 4
    assert sum(row["is_primary"] for row in analysis["results"]) == 2
    paths = write_penalty_sensitivity_artifacts(tmp_path, analysis)
    assert all(Path(path).exists() for path in paths.values())


def test_penalty_sensitivity_requires_primary_in_grid() -> None:
    with pytest.raises(ValueError, match="must contain the primary"):
        run_penalty_sensitivity(
            rows=[],
            training_rows=[],
            state_lookup={},
            kernel=np.eye(8),
            cost_matrix=np.ones((8, 8)) - np.eye(8),
            training_summary=_training_summary(),
            primary_penalty=0.1,
            penalties=(0.2,),
            decision_year_start=2020,
            outcome_year_end=2021,
        )
