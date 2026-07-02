from __future__ import annotations

import numpy as np

from ears_q_learning.wasserstein import (
    robust_lower_expectation_dual,
    robust_lower_expectation_dual_solution,
    robust_lower_expectation_primal_binary,
    robust_lower_expectation_primal_lp,
    wasserstein_distance,
)


def _loop_dual(
    reference_distribution: np.ndarray,
    values: np.ndarray,
    epsilon: float,
    cost_matrix: np.ndarray,
) -> float:
    candidates = {0.0}
    for row in range(len(values)):
        for first in range(len(values)):
            for second in range(first + 1, len(values)):
                denominator = cost_matrix[row, first] - cost_matrix[row, second]
                if abs(denominator) < 1e-12:
                    continue
                candidate = (values[second] - values[first]) / denominator
                if candidate >= 0:
                    candidates.add(float(candidate))
    return max(
        float(
            reference_distribution
            @ np.array(
                [
                    np.min(values + candidate * cost_matrix[row, :])
                    for row in range(len(values))
                ]
            )
            - candidate * epsilon
        )
        for candidate in candidates
    )


def test_dual_matches_binary_primal_example() -> None:
    reference = np.array([0.7, 0.3], dtype=float)
    values = np.array([1.0, 0.2], dtype=float)
    epsilon = 0.15
    cost = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    dual = robust_lower_expectation_dual(reference, values, epsilon, cost)
    primal = robust_lower_expectation_primal_binary(reference, values, epsilon)
    assert abs(dual - primal) < 1e-9


def test_dual_matches_finite_lp_primal_example() -> None:
    reference = np.array([0.5, 0.3, 0.2], dtype=float)
    values = np.array([0.9, 0.4, 0.1], dtype=float)
    epsilon = 0.2
    cost = np.array(
        [
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 0.5],
            [1.0, 0.5, 0.0],
        ],
        dtype=float,
    )

    dual = robust_lower_expectation_dual(reference, values, epsilon, cost)
    primal = robust_lower_expectation_primal_lp(reference, values, epsilon, cost)

    assert abs(dual - primal) < 1e-9


def test_vectorized_dual_matches_loop_search_for_random_eight_state_inputs() -> None:
    rng = np.random.default_rng(27)
    bits = np.array(
        [[(state >> shift) & 1 for shift in (2, 1, 0)] for state in range(8)]
    )
    cost = np.mean(bits[:, None, :] != bits[None, :, :], axis=2)

    for _ in range(20):
        reference = rng.dirichlet(np.ones(8))
        values = rng.normal(size=8)
        epsilon = float(rng.uniform(0.001, 0.5))

        vectorized = robust_lower_expectation_dual(
            reference, values, epsilon, cost
        )
        loop = _loop_dual(reference, values, epsilon, cost)

        assert abs(vectorized - loop) < 1e-12


def test_dual_solution_transform_has_the_reported_expectation() -> None:
    reference = np.array([0.5, 0.3, 0.2], dtype=float)
    values = np.array([0.9, 0.4, 0.1], dtype=float)
    epsilon = 0.2
    cost = np.array(
        [[0.0, 0.5, 1.0], [0.5, 0.0, 0.5], [1.0, 0.5, 0.0]],
        dtype=float,
    )

    solution = robust_lower_expectation_dual_solution(
        reference, values, epsilon, cost
    )

    sampled_expectation = (
        reference @ solution.transformed_values
        - epsilon * solution.multiplier
    )
    assert abs(sampled_expectation - solution.lower_expectation) < 1e-12


def test_wasserstein_distance_solves_finite_transport_problem() -> None:
    source = np.array([0.5, 0.5, 0.0], dtype=float)
    target = np.array([0.0, 0.5, 0.5], dtype=float)
    cost = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )

    assert wasserstein_distance(source, target, cost) == 1.0


def test_wasserstein_distance_rejects_non_probability_vectors() -> None:
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])

    with np.testing.assert_raises(ValueError):
        wasserstein_distance(np.array([0.8, 0.8]), np.array([0.5, 0.5]), cost)
