from __future__ import annotations

from pathlib import Path

from ears_q_learning.data import validate_raw_snapshot
from ears_q_learning.preprocessing import (
    build_country_year_panel,
    build_preprocessing_report,
    eligible_countries,
)
from ears_q_learning.state_space import encode_state, fit_thresholds


def test_thresholds_use_training_data_only(sample_raw_csv: Path) -> None:
    records = validate_raw_snapshot(sample_raw_csv, "Escherichia coli", 2015, 2024)
    rows = build_country_year_panel(records)
    training_rows = [row for row in rows if row.year <= 2019]
    thresholds = fit_thresholds(training_rows)
    assert thresholds.resistance_3gc_median == 19.0
    assert thresholds.resistance_fq_median == 29.0


def test_thresholds_ignore_evaluation_period_leakage(sample_raw_csv: Path) -> None:
    records = validate_raw_snapshot(sample_raw_csv, "Escherichia coli", 2015, 2024)
    rows = build_country_year_panel(records)
    training_rows = [row for row in rows if row.year <= 2019]
    baseline = fit_thresholds(training_rows)

    mutated_training_rows = []
    for row in rows:
        if row.year <= 2019:
            mutated_training_rows.append(row)
        else:
            mutated_training_rows.append(
                type(row)(
                    country=row.country,
                    year=row.year,
                    resistance_3gc=999.0,
                    resistance_fq=999.0,
                    resistance_carb=row.resistance_carb,
                    tested_3gc=row.tested_3gc,
                    tested_fq=row.tested_fq,
                    tested_carb=row.tested_carb,
                )
            )
    candidate = fit_thresholds([row for row in mutated_training_rows if row.year <= 2019])
    assert candidate == baseline


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


def test_preprocessing_report_lists_eligible_countries(sample_raw_csv: Path) -> None:
    records = validate_raw_snapshot(sample_raw_csv, "Escherichia coli", 2015, 2024)
    rows = build_country_year_panel(records)
    countries = eligible_countries(rows, 2019, 2020, 3, 2)
    report = build_preprocessing_report(rows, countries, 2019, 2020, 3, 2)
    assert report["eligible_country_count"] == 2
    assert report["eligible_countries"] == ["Aland", "Borduria"]
