"""Country-year preprocessing for the EARS-Net Q-learning project."""

from __future__ import annotations

from collections import defaultdict

from ears_q_learning.types import CountryYearRow, RawRecord, TransitionRecord


def build_country_year_panel(records: list[RawRecord]) -> list[CountryYearRow]:
    """Pivot validated raw records into one row per country and year."""
    grouped: dict[tuple[str, int], dict[str, RawRecord]] = defaultdict(dict)
    for record in records:
        grouped[(record.country, record.year)][record.action_code] = record

    rows: list[CountryYearRow] = []
    for (country, year), by_action in sorted(grouped.items()):
        if set(by_action) != {"3gc", "fq", "carb"}:
            continue
        rows.append(
            CountryYearRow(
                country=country,
                year=year,
                resistance_3gc=by_action["3gc"].resistance_percentage,
                resistance_fq=by_action["fq"].resistance_percentage,
                resistance_carb=by_action["carb"].resistance_percentage,
                tested_3gc=by_action["3gc"].tested_count,
                tested_fq=by_action["fq"].tested_count,
                tested_carb=by_action["carb"].tested_count,
            )
        )
    return rows


def split_rows_by_period(
    rows: list[CountryYearRow],
    training_year_end: int,
    evaluation_year_end: int,
) -> tuple[list[CountryYearRow], list[CountryYearRow]]:
    """Split country-year rows into training and evaluation periods."""
    training = [row for row in rows if row.year <= training_year_end]
    evaluation = [
        row
        for row in rows
        if training_year_end < row.year <= evaluation_year_end
    ]
    return training, evaluation


def eligible_countries(
    rows: list[CountryYearRow],
    training_year_end: int,
    evaluation_year_start: int,
    minimum_training_transitions: int,
    minimum_evaluation_transitions: int,
) -> set[str]:
    """Return countries satisfying the transition completeness rules."""
    by_country: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_country[row.country].append(row.year)
    eligible: set[str] = set()
    for country, years in by_country.items():
        present = set(years)
        training_transitions = sum(
            1
            for year in range(min(present), training_year_end)
            if year in present and (year + 1) in present
        )
        evaluation_transitions = sum(
            1
            for year in range(evaluation_year_start, max(present))
            if year in present and (year + 1) in present
        )
        if (
            training_transitions >= minimum_training_transitions
            and evaluation_transitions >= minimum_evaluation_transitions
        ):
            eligible.add(country)
    return eligible


def filter_rows_by_country(rows: list[CountryYearRow], countries: set[str]) -> list[CountryYearRow]:
    """Filter country-year rows to the chosen countries."""
    return [row for row in rows if row.country in countries]


def build_transition_records(
    rows: list[CountryYearRow],
    state_lookup: dict[tuple[str, int], int],
    weighting: str,
) -> list[TransitionRecord]:
    """Create transition records from encoded country-year rows."""
    by_country: dict[str, dict[int, CountryYearRow]] = defaultdict(dict)
    for row in rows:
        by_country[row.country][row.year] = row

    transitions: list[TransitionRecord] = []
    for country, by_year in sorted(by_country.items()):
        for year in sorted(by_year):
            if year + 1 not in by_year:
                continue
            next_row = by_year[year + 1]
            if weighting == "equal":
                weight = 1.0
            elif weighting == "tested":
                weight = (
                    next_row.tested_3gc + next_row.tested_fq + next_row.tested_carb
                ) / 3.0
            else:
                raise ValueError(f"Unsupported weighting scheme: {weighting}")
            transitions.append(
                TransitionRecord(
                    country=country,
                    year=year,
                    current_state=state_lookup[(country, year)],
                    next_state=state_lookup[(country, year + 1)],
                    next_row=next_row,
                    weight=weight,
                )
            )
    return transitions
