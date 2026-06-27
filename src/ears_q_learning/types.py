"""Typed records used by the project pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RawRecord:
    """One validated row from the raw EARS-Net snapshot."""

    country: str
    year: int
    organism: str
    action_code: str
    resistance_percentage: float
    tested_count: int


@dataclass(frozen=True)
class CountryYearRow:
    """One country-year row containing all three antibiotic classes."""

    country: str
    year: int
    resistance_3gc: float
    resistance_fq: float
    resistance_carb: float
    tested_3gc: int
    tested_fq: int
    tested_carb: int


@dataclass(frozen=True)
class TransitionRecord:
    """One country-year transition for the MDP."""

    country: str
    year: int
    current_state: int
    next_state: int
    next_row: CountryYearRow
    weight: float
