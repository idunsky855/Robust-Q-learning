from __future__ import annotations

import numpy as np

from ears_q_learning.learning import (
    solve_bellman_optimum,
    train_classical_q_learning,
    train_q_learning,
    train_robust_q_learning,
)
from ears_q_learning.mdp import myopic_policy, normalized_hamming_cost


def test_classical_q_learning_matches_myopic_policy_in_action_independent_case() -> None:
    kernel = np.zeros((8, 8), dtype=float)
    kernel[:, 0] = 0.25
    kernel[:, 1] = 0.75
    reward_bands = {
        "3gc": {0: 0.9, 1: 0.2},
        "fq": {0: 0.7, 1: 0.4},
        "carb": {0: 0.5, 1: 0.3},
    }
    learned = train_q_learning(
        kernel=kernel,
        reward_bands=reward_bands,
        discount=0.45,
        exploration_rate=0.1,
        updates=30000,
        seed=7,
        robust_epsilon=0.0,
        cost_matrix=normalized_hamming_cost(),
    )
    expected = myopic_policy(kernel, reward_bands)
    assert np.array_equal(learned.greedy_policy, expected)


def test_training_is_deterministic_for_identical_seed() -> None:
    kernel = np.full((8, 8), 1.0 / 8.0, dtype=float)
    reward_bands = {
        "3gc": {0: 0.8, 1: 0.1},
        "fq": {0: 0.7, 1: 0.2},
        "carb": {0: 0.6, 1: 0.3},
    }
    kwargs = dict(
        kernel=kernel,
        reward_bands=reward_bands,
        discount=0.3,
        exploration_rate=0.1,
        updates=5000,
        seed=19,
        robust_epsilon=0.05,
        cost_matrix=normalized_hamming_cost(),
    )
    first = train_q_learning(**kwargs)
    second = train_q_learning(**kwargs)
    assert np.allclose(first.q_values, second.q_values)


def test_first_visit_learning_rate_uses_full_target() -> None:
    kernel = np.zeros((8, 8), dtype=float)
    kernel[:, 0] = 1.0
    reward_bands = {
        "3gc": {0: 1.0, 1: 0.0},
        "fq": {0: 0.5, 1: 0.5},
        "carb": {0: 0.4, 1: 0.4},
    }

    result = train_classical_q_learning(
        kernel=kernel,
        reward_bands=reward_bands,
        discount=0.0,
        exploration_rate=0.0,
        updates=1,
        seed=3,
        cost_matrix=normalized_hamming_cost(),
    )

    assert result.visits.sum() == 1.0
    assert result.q_values.max() == 1.0


def test_robust_q_learning_reproduces_coin_toss_style_caution() -> None:
    kernel = np.zeros((8, 8), dtype=float)
    kernel[:, 0] = 0.75
    kernel[:, 4] = 0.25
    reward_bands = {
        "3gc": {0: 1.0, 1: 0.0},
        "fq": {0: 0.65, 1: 0.65},
        "carb": {0: 0.4, 1: 0.4},
    }
    cost_matrix = normalized_hamming_cost()

    classical = train_classical_q_learning(
        kernel=kernel,
        reward_bands=reward_bands,
        discount=0.0,
        exploration_rate=0.1,
        updates=50000,
        seed=11,
        cost_matrix=cost_matrix,
    )
    robust = train_robust_q_learning(
        kernel=kernel,
        reward_bands=reward_bands,
        discount=0.0,
        exploration_rate=0.1,
        updates=50000,
        seed=11,
        robust_epsilon=0.1,
        cost_matrix=cost_matrix,
    )

    recurrent_states = [0, 4]
    assert set(classical.greedy_policy[recurrent_states]) == {0}
    assert set(robust.greedy_policy[recurrent_states]) == {1}


def test_training_follows_the_sampled_markov_trajectory() -> None:
    kernel = np.zeros((8, 8), dtype=float)
    for state in range(8):
        kernel[state, (state + 1) % 8] = 1.0
    reward_bands = {
        "3gc": {0: 1.0, 1: 1.0},
        "fq": {0: 0.0, 1: 0.0},
        "carb": {0: 0.0, 1: 0.0},
    }

    result = train_classical_q_learning(
        kernel=kernel,
        reward_bands=reward_bands,
        discount=0.0,
        exploration_rate=0.0,
        updates=3,
        seed=5,
        cost_matrix=normalized_hamming_cost(),
        starting_state=0,
    )

    assert result.visits[:, 0].tolist() == [
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_exact_bellman_solution_is_a_reference_for_sampled_learning() -> None:
    kernel = np.full((8, 8), 1.0 / 8.0, dtype=float)
    reward_bands = {
        "3gc": {0: 0.9, 1: 0.2},
        "fq": {0: 0.6, 1: 0.4},
        "carb": {0: 0.5, 1: 0.3},
    }
    cost = normalized_hamming_cost()
    exact = solve_bellman_optimum(
        kernel=kernel,
        reward_bands=reward_bands,
        discount=0.45,
        cost_matrix=cost,
    )
    learned = train_classical_q_learning(
        kernel=kernel,
        reward_bands=reward_bands,
        discount=0.45,
        exploration_rate=0.2,
        updates=100000,
        seed=17,
        cost_matrix=cost,
    )

    assert exact.residual <= 1e-12
    assert np.array_equal(learned.greedy_policy, exact.greedy_policy)
