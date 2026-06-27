from __future__ import annotations

from pathlib import Path

from ears_q_learning.data import validate_raw_snapshot
from ears_q_learning.preprocessing import build_country_year_panel, eligible_countries
from ears_q_learning.state_space import encode_state, fit_thresholds


def test_thresholds_use_training_data_only(sample_raw_csv: Path) -> None:
    records = validate_raw_snapshot(sample_raw_csv, "Escherichia coli", 2015, 2024)
    rows = build_country_year_panel(records)
    training_rows = [row for row in rows if row.year <= 2019]
    thresholds = fit_thresholds(training_rows)
    assert thresholds.resistance_3gc_median == 19.0
    assert thresholds.resistance_fq_median == 29.0


def test_state_encoding_is_deterministic(sample_raw_csv: Path) -> None:
    records = validate_raw_snapshot(sample_raw_csv, "Escherichia coli", 2015, 2024)
    rows = build_country_year_panel(records)
    thresholds = fit_thresholds([row for row in rows if row.year <= 2019])
    encoded_once = [encode_state(row, thresholds) for row in rows]
    encoded_twice = [encode_state(row, thresholds) for row in rows]
    assert encoded_once == encoded_twice


def test_country_eligibility_uses_transition_rules(sample_raw_csv: Path) -> None:
    records = validate_raw_snapshot(sample_raw_csv, "Escherichia coli", 2015, 2024)
    rows = build_country_year_panel(records)
    countries = eligible_countries(rows, 2019, 2020, 3, 2)
    assert countries == {"Aland", "Borduria"}
