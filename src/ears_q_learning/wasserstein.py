"""Finite-state Wasserstein utilities."""

from __future__ import annotations

import itertools

import numpy as np


def _candidate_lambdas(values: np.ndarray, cost_row: np.ndarray) -> list[float]:
    candidates = {0.0}
    for i, j in itertools.combinations(range(len(values)), 2):
        denominator = cost_row[i] - cost_row[j]
        if abs(denominator) < 1e-12:
            continue
        candidate = (values[j] - values[i]) / denominator
        if candidate >= 0:
            candidates.add(float(candidate))
    return sorted(candidates)


def robust_lower_expectation_dual(
    reference_distribution: np.ndarray,
    values: np.ndarray,
    epsilon: float,
    cost_matrix: np.ndarray,
) -> float:
    """Compute the Wasserstein-robust lower expectation by dual search."""
    candidates = {0.0}
    for row in range(cost_matrix.shape[0]):
        candidates.update(_candidate_lambdas(values, cost_matrix[row, :]))
    best = float("-inf")
    for lam in sorted(candidates):
        row_terms = []
        for row in range(cost_matrix.shape[0]):
            row_terms.append(np.min(values + lam * cost_matrix[row, :]))
        objective = float(reference_distribution @ np.array(row_terms) - lam * epsilon)
        if objective > best:
            best = objective
    return best


def robust_lower_expectation_primal_binary(
    reference_distribution: np.ndarray,
    values: np.ndarray,
    epsilon: float,
) -> float:
    """Closed-form two-state primal used in tests."""
    if len(reference_distribution) != 2 or len(values) != 2:
        raise ValueError("This helper only supports two-state examples.")
    if values[0] <= values[1]:
        low, high = 0, 1
    else:
        low, high = 1, 0
    shift = min(reference_distribution[high], epsilon)
    stressed = reference_distribution.copy()
    stressed[low] += shift
    stressed[high] -= shift
    return float(stressed @ values)
