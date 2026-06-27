"""MDP utilities for the EARS-Net Q-learning project."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ears_q_learning.constants import ACTION_INDEX, STATE_COUNT
from ears_q_learning.state_space import state_to_bits
from ears_q_learning.types import CountryYearRow, TransitionRecord
from ears_q_learning.wasserstein import wasserstein_distance


def normalized_hamming_cost() -> np.ndarray:
    """Build the normalized Hamming distance matrix for the eight states."""
    matrix = np.zeros((STATE_COUNT, STATE_COUNT), dtype=float)
    for i in range(STATE_COUNT):
        bits_i = state_to_bits(i)
        for j in range(STATE_COUNT):
            bits_j = state_to_bits(j)
            matrix[i, j] = sum(a != b for a, b in zip(bits_i, bits_j)) / 3.0
    return matrix


def estimate_reward_bands(
    training_rows: list[CountryYearRow],
    state_lookup: dict[tuple[str, int], int],
    carbapenem_penalty: float,
) -> dict[str, dict[int, float]]:
    """Estimate low-band and high-band rewards from training rows."""
    grouped: dict[str, dict[int, list[float]]] = {
        "3gc": {0: [], 1: []},
        "fq": {0: [], 1: []},
        "carb": {0: [], 1: []},
    }
    for row in training_rows:
        state = state_lookup[(row.country, row.year)]
        bits = state_to_bits(state)
        grouped["3gc"][bits[0]].append(1.0 - row.resistance_3gc / 100.0)
        grouped["fq"][bits[1]].append(1.0 - row.resistance_fq / 100.0)
        grouped["carb"][bits[2]].append(1.0 - row.resistance_carb / 100.0)
    rewards: dict[str, dict[int, float]] = {}
    for action, bands in grouped.items():
        overall = [value for values in bands.values() for value in values]
        overall_mean = float(np.mean(overall)) if overall else 0.0
        rewards[action] = {}
        for band, values in bands.items():
            reward = float(np.mean(values)) if values else overall_mean
            if action == "carb":
                reward -= carbapenem_penalty
            rewards[action][band] = reward
    return rewards


def transition_kernel(
    transitions: list[TransitionRecord],
    smoothing_gamma: float,
) -> np.ndarray:
    """Estimate and uniformly smooth the action-independent transition kernel."""
    if not 0.0 <= smoothing_gamma <= 1.0:
        raise ValueError("Smoothing gamma must be between zero and one.")
    counts = np.zeros((STATE_COUNT, STATE_COUNT), dtype=float)
    totals = np.zeros(STATE_COUNT, dtype=float)
    for transition in transitions:
        if transition.weight < 0:
            raise ValueError("Transition weights must be non-negative.")
        counts[transition.current_state, transition.next_state] += transition.weight
        totals[transition.current_state] += transition.weight
    kernel = np.zeros_like(counts)
    for state in range(STATE_COUNT):
        if totals[state] == 0:
            kernel[state, :] = 1.0 / STATE_COUNT
        else:
            empirical = counts[state, :] / totals[state]
            kernel[state, :] = (
                (1.0 - smoothing_gamma) * empirical
                + smoothing_gamma / STATE_COUNT
            )
    return kernel


def annual_state_distributions(rows: list[CountryYearRow], state_lookup: dict[tuple[str, int], int]) -> dict[int, np.ndarray]:
    """Compute the empirical state distribution for each year."""
    by_year: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        by_year[row.year].append(state_lookup[(row.country, row.year)])
    distributions: dict[int, np.ndarray] = {}
    for year, states in sorted(by_year.items()):
        values = np.zeros(STATE_COUNT, dtype=float)
        for state in states:
            values[state] += 1.0
        distributions[year] = values / values.sum()
    return distributions


def calibrate_wasserstein_radius(
    annual_distributions: dict[int, np.ndarray],
    cost_matrix: np.ndarray,
) -> dict[str, object]:
    """Calibrate epsilon from consecutive training-year state distributions."""
    annual_distances: list[dict[str, int | float]] = []
    years = sorted(annual_distributions)
    for from_year, to_year in zip(years, years[1:]):
        if to_year != from_year + 1:
            continue
        distance = wasserstein_distance(
            annual_distributions[from_year],
            annual_distributions[to_year],
            cost_matrix,
        )
        annual_distances.append(
            {
                "from_year": from_year,
                "to_year": to_year,
                "distance": distance,
            }
        )
    if not annual_distances:
        raise ValueError("At least two consecutive annual distributions are required.")
    epsilon_star = float(
        np.median([entry["distance"] for entry in annual_distances])
    )
    return {
        "annual_distances": annual_distances,
        "epsilon_star": epsilon_star,
    }


def state_action_reward(next_state: int, action_code: str, reward_bands: dict[str, dict[int, float]]) -> float:
    """Return the modeled reward for one action and next state."""
    bits = state_to_bits(next_state)
    band = {
        "3gc": bits[0],
        "fq": bits[1],
        "carb": bits[2],
    }[action_code]
    return reward_bands[action_code][band]


def observed_reward(next_row: CountryYearRow, action_code: str, carbapenem_penalty: float) -> float:
    """Return the observed next-year reward for evaluation."""
    if action_code == "3gc":
        reward = 1.0 - next_row.resistance_3gc / 100.0
    elif action_code == "fq":
        reward = 1.0 - next_row.resistance_fq / 100.0
    elif action_code == "carb":
        reward = 1.0 - next_row.resistance_carb / 100.0 - carbapenem_penalty
    else:
        raise ValueError(f"Unsupported action code: {action_code}")
    return reward


def myopic_policy(kernel: np.ndarray, reward_bands: dict[str, dict[int, float]]) -> np.ndarray:
    """Compute the state-wise myopic policy under the learned model."""
    values = np.zeros((STATE_COUNT, len(ACTION_INDEX)), dtype=float)
    action_codes = list(ACTION_INDEX)
    for state in range(STATE_COUNT):
        for action_index, action_code in enumerate(action_codes):
            rewards = np.array(
                [state_action_reward(next_state, action_code, reward_bands) for next_state in range(STATE_COUNT)],
                dtype=float,
            )
            values[state, action_index] = float(kernel[state, :] @ rewards)
    return np.argmax(values, axis=1)
