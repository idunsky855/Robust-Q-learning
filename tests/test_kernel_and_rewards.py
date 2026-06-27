from __future__ import annotations

from pathlib import Path

import numpy as np

from ears_q_learning.data import validate_raw_snapshot
from ears_q_learning.mdp import estimate_reward_bands, transition_kernel
from ears_q_learning.preprocessing import (
    build_country_year_panel,
    build_transition_records,
    eligible_countries,
    filter_rows_by_country,
)
from ears_q_learning.state_space import encode_state, fit_thresholds


def test_kernel_is_stochastic_and_action_independent(sample_raw_csv: Path) -> None:
    records = validate_raw_snapshot(sample_raw_csv, "Escherichia coli", 2015, 2024)
    rows = build_country_year_panel(records)
    countries = eligible_countries(rows, 2019, 2020, 3, 2)
    rows = filter_rows_by_country(rows, countries)
    thresholds = fit_thresholds([row for row in rows if row.year <= 2019])
    lookup = {(row.country, row.year): encode_state(row, thresholds) for row in rows}
    transitions = build_transition_records(rows, lookup, "equal")
    kernel = transition_kernel(transitions, 0.1)
    assert kernel.shape == (8, 8)
    assert np.allclose(kernel.sum(axis=1), np.ones(8))


def test_reward_bands_include_carbapenem_penalty(sample_raw_csv: Path) -> None:
    records = validate_raw_snapshot(sample_raw_csv, "Escherichia coli", 2015, 2024)
    rows = build_country_year_panel(records)
    thresholds = fit_thresholds([row for row in rows if row.year <= 2019])
    lookup = {(row.country, row.year): encode_state(row, thresholds) for row in rows}
    rewards = estimate_reward_bands(
        training_rows=[row for row in rows if row.year <= 2019],
        state_lookup=lookup,
        carbapenem_penalty=0.1,
    )
    assert rewards["carb"][0] < 1.0
