"""State encoding for Setting 1."""

from __future__ import annotations

from dataclasses import dataclass

from ears_q_learning.types import CountryYearRow


@dataclass(frozen=True)
class Thresholds:
    """Binary discretization thresholds fitted on training data only."""

    resistance_3gc_median: float
    resistance_fq_median: float


def state_to_bits(state: int) -> tuple[int, int, int]:
    """Convert a state index into its three binary indicators."""
    if state < 0 or state > 7:
        raise ValueError(f"State must be in 0..7, received {state}.")
    return ((state >> 2) & 1, (state >> 1) & 1, state & 1)


def bits_to_state(bits: tuple[int, int, int]) -> int:
    """Convert three binary indicators into a state index."""
    return (bits[0] << 2) + (bits[1] << 1) + bits[2]


def fit_thresholds(training_rows: list[CountryYearRow]) -> Thresholds:
    """Fit training-only thresholds for Setting 1."""
    if not training_rows:
        raise ValueError("At least one training row is required.")
    resistance_3gc = sorted(row.resistance_3gc for row in training_rows)
    resistance_fq = sorted(row.resistance_fq for row in training_rows)
    midpoint = len(training_rows) // 2
    if len(training_rows) % 2:
        median_3gc = resistance_3gc[midpoint]
        median_fq = resistance_fq[midpoint]
    else:
        median_3gc = (resistance_3gc[midpoint - 1] + resistance_3gc[midpoint]) / 2.0
        median_fq = (resistance_fq[midpoint - 1] + resistance_fq[midpoint]) / 2.0
    return Thresholds(
        resistance_3gc_median=median_3gc,
        resistance_fq_median=median_fq,
    )


def encode_state(row: CountryYearRow, thresholds: Thresholds) -> int:
    """Encode one country-year row into a Setting 1 state."""
    bits = (
        int(row.resistance_3gc > thresholds.resistance_3gc_median),
        int(row.resistance_fq > thresholds.resistance_fq_median),
        int(row.resistance_carb > 0.0),
    )
    return bits_to_state(bits)
