from __future__ import annotations

import numpy as np

from ears_q_learning.wasserstein import (
    robust_lower_expectation_dual,
    robust_lower_expectation_primal_binary,
    wasserstein_distance,
)


def test_dual_matches_binary_primal_example() -> None:
    reference = np.array([0.7, 0.3], dtype=float)
    values = np.array([1.0, 0.2], dtype=float)
    epsilon = 0.15
    cost = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    dual = robust_lower_expectation_dual(reference, values, epsilon, cost)
    primal = robust_lower_expectation_primal_binary(reference, values, epsilon)
    assert abs(dual - primal) < 1e-9


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
