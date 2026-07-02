"""Classical and robust tabular Q-learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ears_q_learning.constants import ACTION_INDEX, ACTIONS, STATE_COUNT
from ears_q_learning.mdp import state_action_reward
from ears_q_learning.wasserstein import robust_lower_expectation_dual_solution


@dataclass(frozen=True)
class TrainingResult:
    """Outputs from one policy-training run."""

    q_values: np.ndarray
    greedy_policy: np.ndarray
    visits: np.ndarray


@dataclass(frozen=True)
class BellmanSolution:
    """Exact fixed-point solution for the finite model."""

    q_values: np.ndarray
    greedy_policy: np.ndarray
    iterations: int
    residual: float


def _validate_training_inputs(
    kernel: np.ndarray,
    discount: float,
    exploration_rate: float,
    updates: int,
    robust_epsilon: float,
    cost_matrix: np.ndarray,
    q_norm: int,
) -> None:
    """Validate tabular Q-learning inputs before training starts."""
    if kernel.shape != (STATE_COUNT, STATE_COUNT):
        raise ValueError("Kernel must have shape (8, 8).")
    if np.any(kernel < 0) or not np.allclose(kernel.sum(axis=1), 1.0):
        raise ValueError("Kernel rows must be stochastic probability vectors.")
    if cost_matrix.shape != (STATE_COUNT, STATE_COUNT):
        raise ValueError("Cost matrix must have shape (8, 8).")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("Discount must be between zero and one.")
    if not 0.0 <= exploration_rate <= 1.0:
        raise ValueError("Exploration rate must be between zero and one.")
    if updates <= 0:
        raise ValueError("Updates must be positive.")
    if robust_epsilon < 0:
        raise ValueError("Robust epsilon must be non-negative.")
    if q_norm != 1:
        raise ValueError("Only q=1 Wasserstein uncertainty is supported.")


def epsilon_greedy(
    q_values: np.ndarray,
    state: int,
    exploration_rate: float,
    rng: np.random.Generator,
) -> int:
    """Select one action using an epsilon-greedy rule."""
    if rng.random() < exploration_rate:
        return int(rng.integers(len(ACTION_INDEX)))
    return int(np.argmax(q_values[state, :]))


def solve_bellman_optimum(
    kernel: np.ndarray,
    reward_bands: dict[str, dict[int, float]],
    discount: float,
    cost_matrix: np.ndarray,
    robust_epsilon: float = 0.0,
    tolerance: float = 1e-12,
    max_iterations: int = 10000,
) -> BellmanSolution:
    """Solve the finite classical or Wasserstein-robust Bellman equation."""
    _validate_training_inputs(
        kernel=kernel,
        discount=discount,
        exploration_rate=0.0,
        updates=max_iterations,
        robust_epsilon=robust_epsilon,
        cost_matrix=cost_matrix,
        q_norm=1,
    )
    if tolerance <= 0:
        raise ValueError("Bellman tolerance must be positive.")

    q_values = np.zeros((STATE_COUNT, len(ACTIONS)), dtype=float)
    action_codes = [action.code for action in ACTIONS]
    residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        future_values = np.max(q_values, axis=1)
        updated = np.zeros_like(q_values)
        for state in range(STATE_COUNT):
            for action, action_code in enumerate(action_codes):
                outcomes = np.array(
                    [
                        state_action_reward(next_state, action_code, reward_bands)
                        + discount * future_values[next_state]
                        for next_state in range(STATE_COUNT)
                    ],
                    dtype=float,
                )
                if robust_epsilon > 0:
                    updated[state, action] = robust_lower_expectation_dual_solution(
                        reference_distribution=kernel[state, :],
                        values=outcomes,
                        epsilon=robust_epsilon,
                        cost_matrix=cost_matrix,
                    ).lower_expectation
                else:
                    updated[state, action] = float(kernel[state, :] @ outcomes)
        residual = float(np.max(np.abs(updated - q_values)))
        q_values = updated
        if residual <= tolerance:
            break
    else:
        raise RuntimeError("Bellman iteration did not converge.")

    return BellmanSolution(
        q_values=q_values,
        greedy_policy=np.argmax(q_values, axis=1),
        iterations=iteration,
        residual=residual,
    )


def train_q_learning(
    kernel: np.ndarray,
    reward_bands: dict[str, dict[int, float]],
    discount: float,
    exploration_rate: float,
    updates: int,
    seed: int,
    robust_epsilon: float,
    cost_matrix: np.ndarray,
    q_norm: int = 1,
    starting_state: int = 0,
) -> TrainingResult:
    """Train classical or robust Q-learning along one Markov trajectory."""
    _validate_training_inputs(
        kernel=kernel,
        discount=discount,
        exploration_rate=exploration_rate,
        updates=updates,
        robust_epsilon=robust_epsilon,
        cost_matrix=cost_matrix,
        q_norm=q_norm,
    )
    if not 0 <= starting_state < STATE_COUNT:
        raise ValueError(f"Starting state must be between 0 and {STATE_COUNT - 1}.")
    action_codes = [action.code for action in ACTIONS]
    rng = np.random.default_rng(seed)
    q_values = np.zeros((STATE_COUNT, len(action_codes)), dtype=float)
    visits = np.zeros_like(q_values)
    state = starting_state

    for _ in range(updates):
        action = epsilon_greedy(q_values, state, exploration_rate, rng)
        next_state = int(rng.choice(np.arange(STATE_COUNT), p=kernel[state, :]))
        action_code = action_codes[action]
        future_values = np.max(q_values, axis=1)
        outcome_values = np.array(
            [
                state_action_reward(candidate_state, action_code, reward_bands)
                + discount * future_values[candidate_state]
                for candidate_state in range(STATE_COUNT)
            ],
            dtype=float,
        )
        if robust_epsilon > 0:
            dual = robust_lower_expectation_dual_solution(
                reference_distribution=kernel[state, :],
                values=outcome_values,
                epsilon=robust_epsilon,
                cost_matrix=cost_matrix,
            )
            target = (
                dual.transformed_values[next_state]
                - robust_epsilon * dual.multiplier
            )
        else:
            target = outcome_values[next_state]
        learning_rate = 1.0 / (1.0 + visits[state, action])
        q_values[state, action] += learning_rate * (target - q_values[state, action])
        visits[state, action] += 1.0
        state = next_state

    return TrainingResult(
        q_values=q_values,
        greedy_policy=np.argmax(q_values, axis=1),
        visits=visits,
    )


def train_classical_q_learning(
    kernel: np.ndarray,
    reward_bands: dict[str, dict[int, float]],
    discount: float,
    exploration_rate: float,
    updates: int,
    seed: int,
    cost_matrix: np.ndarray,
    q_norm: int = 1,
    starting_state: int = 0,
) -> TrainingResult:
    """Train classical tabular Q-learning as the epsilon-zero comparator."""
    return train_q_learning(
        kernel=kernel,
        reward_bands=reward_bands,
        discount=discount,
        exploration_rate=exploration_rate,
        updates=updates,
        seed=seed,
        robust_epsilon=0.0,
        cost_matrix=cost_matrix,
        q_norm=q_norm,
        starting_state=starting_state,
    )


def train_robust_q_learning(
    kernel: np.ndarray,
    reward_bands: dict[str, dict[int, float]],
    discount: float,
    exploration_rate: float,
    updates: int,
    seed: int,
    robust_epsilon: float,
    cost_matrix: np.ndarray,
    q_norm: int = 1,
    starting_state: int = 0,
) -> TrainingResult:
    """Train Wasserstein-robust tabular Q-learning for q=1 uncertainty."""
    if robust_epsilon <= 0:
        raise ValueError("Robust Q-learning requires a positive epsilon.")
    return train_q_learning(
        kernel=kernel,
        reward_bands=reward_bands,
        discount=discount,
        exploration_rate=exploration_rate,
        updates=updates,
        seed=seed,
        robust_epsilon=robust_epsilon,
        cost_matrix=cost_matrix,
        q_norm=q_norm,
        starting_state=starting_state,
    )
