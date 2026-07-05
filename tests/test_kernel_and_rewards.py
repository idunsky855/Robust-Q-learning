from __future__ import annotations

from pathlib import Path

import numpy as np

from ears_q_learning.data import validate_raw_snapshot
from ears_q_learning.mdp import (
    calibrate_wasserstein_radius,
    estimate_reward_bands,
    normalized_hamming_cost,
    transition_kernel,
)
from ears_q_learning.types import CountryYearRow, TransitionRecord
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


def test_kernel_distributes_total_smoothing_mass_uniformly() -> None:
    row = CountryYearRow("Aland", 2016, 0, 0, 0, 1, 1, 1)
    transitions = [TransitionRecord("Aland", 2015, 0, 3, row, 1.0)]

    kernel = transition_kernel(transitions, smoothing_gamma=0.1)

    expected = np.full(8, 0.1 / 8)
    expected[3] += 0.9
    assert np.allclose(kernel[0], expected)
    assert np.allclose(kernel[1], np.full(8, 1 / 8))


def test_drift_calibration_uses_median_consecutive_year_distance() -> None:
    distributions = {
        2015: np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        2016: np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=float),
        2017: np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=float),
    }

    calibration = calibrate_wasserstein_radius(
        distributions,
        normalized_hamming_cost(),
    )

    assert calibration["annual_distances"] == [
        {"from_year": 2015, "to_year": 2016, "distance": 1 / 3},
        {"from_year": 2016, "to_year": 2017, "distance": 2 / 3},
    ]
    assert calibration["epsilon_star"] == 0.5


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


def test_tested_weighting_changes_reward_means() -> None:
    rows = [
        CountryYearRow("A", 2018, 0.0, 0.0, 0.0, 10, 10, 10),
        CountryYearRow("B", 2018, 100.0, 100.0, 100.0, 90, 90, 90),
    ]
    lookup = {("A", 2018): 0, ("B", 2018): 0}

    equal = estimate_reward_bands(rows, lookup, 0.0, "equal")
    tested = estimate_reward_bands(rows, lookup, 0.0, "tested")

    assert equal["3gc"][0] == 0.5
    assert tested["3gc"][0] == 0.1
