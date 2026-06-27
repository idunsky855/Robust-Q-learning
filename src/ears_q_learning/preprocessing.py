"""Country-year preprocessing for the EARS-Net Q-learning project."""

from __future__ import annotations

from dataclasses import asdict
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


def count_complete_transitions(
    rows: list[CountryYearRow],
    training_year_end: int,
    evaluation_year_start: int,
) -> dict[str, dict[str, int]]:
    """Count complete training and evaluation transitions for each country."""
    by_country: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_country[row.country].append(row.year)

    counts: dict[str, dict[str, int]] = {}
    for country, years in sorted(by_country.items()):
        present = set(years)
        counts[country] = {
            "training_transitions": sum(
                1
                for year in range(min(present), training_year_end)
                if year in present and (year + 1) in present
            ),
            "evaluation_transitions": sum(
                1
                for year in range(evaluation_year_start, max(present))
                if year in present and (year + 1) in present
            ),
        }
    return counts


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
    transition_counts = count_complete_transitions(
        rows=rows,
        training_year_end=training_year_end,
        evaluation_year_start=evaluation_year_start,
    )
    eligible: set[str] = set()
    for country, counts in transition_counts.items():
        if (
            counts["training_transitions"] >= minimum_training_transitions
            and counts["evaluation_transitions"] >= minimum_evaluation_transitions
        ):
            eligible.add(country)
    return eligible


def filter_rows_by_country(rows: list[CountryYearRow], countries: set[str]) -> list[CountryYearRow]:
    """Filter country-year rows to the chosen countries."""
    return [row for row in rows if row.country in countries]


def build_preprocessing_report(
    rows: list[CountryYearRow],
    eligible: set[str],
    training_year_end: int,
    evaluation_year_start: int,
    minimum_training_transitions: int,
    minimum_evaluation_transitions: int,
) -> dict[str, object]:
    """Build a machine-readable report for country eligibility and coverage."""
    transition_counts = count_complete_transitions(
        rows=rows,
        training_year_end=training_year_end,
        evaluation_year_start=evaluation_year_start,
    )
    return {
        "status": "country_year_panel_built",
        "country_year_row_count": len(rows),
        "country_count": len({row.country for row in rows}),
        "eligible_country_count": len(eligible),
        "eligible_countries": sorted(eligible),
        "minimum_training_transitions": minimum_training_transitions,
        "minimum_evaluation_transitions": minimum_evaluation_transitions,
        "transition_counts": transition_counts,
    }


def build_state_assignment_rows(
    rows: list[CountryYearRow],
    state_lookup: dict[tuple[str, int], int],
) -> list[dict[str, object]]:
    """Build export-ready state-assignment rows."""
    assignments: list[dict[str, object]] = []
    for row in rows:
        assignments.append(
            {
                **asdict(row),
                "state": state_lookup[(row.country, row.year)],
            }
        )
    return assignments


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
