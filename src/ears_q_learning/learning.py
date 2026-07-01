"""Classical and robust tabular Q-learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ears_q_learning.constants import ACTION_INDEX, ACTIONS, STATE_COUNT
from ears_q_learning.mdp import state_action_reward
from ears_q_learning.wasserstein import robust_lower_expectation_dual


@dataclass(frozen=True)
class TrainingResult:
    """Outputs from one policy-training run."""

    q_values: np.ndarray
    greedy_policy: np.ndarray
    visits: np.ndarray


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
) -> TrainingResult:
    """Train classical or robust tabular Q-learning."""
    _validate_training_inputs(
        kernel=kernel,
        discount=discount,
        exploration_rate=exploration_rate,
        updates=updates,
        robust_epsilon=robust_epsilon,
        cost_matrix=cost_matrix,
        q_norm=q_norm,
    )
    action_codes = [action.code for action in ACTIONS]
    rng = np.random.default_rng(seed)
    q_values = np.zeros((STATE_COUNT, len(action_codes)), dtype=float)
    visits = np.zeros_like(q_values)
    initial_distribution = np.full(STATE_COUNT, 1.0 / STATE_COUNT, dtype=float)

    for _ in range(updates):
        state = int(rng.choice(np.arange(STATE_COUNT), p=initial_distribution))
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
            target = robust_lower_expectation_dual(
                reference_distribution=kernel[state, :],
                values=outcome_values,
                epsilon=robust_epsilon,
                cost_matrix=cost_matrix,
            )
        else:
            # The sampled state keeps the update path seed-dependent while the
            # model-estimated reward/continuation uses the learned kernel.
            reward = state_action_reward(next_state, action_code, reward_bands)
            continuation = float(kernel[state, :] @ future_values)
            target = reward + discount * continuation
        learning_rate = 1.0 / (1.0 + visits[state, action])
        q_values[state, action] += learning_rate * (target - q_values[state, action])
        visits[state, action] += 1.0

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
    )
