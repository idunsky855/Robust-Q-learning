"""Classical and robust tabular Q-learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ears_q_learning.constants import ACTION_INDEX, STATE_COUNT
from ears_q_learning.mdp import state_action_reward
from ears_q_learning.wasserstein import robust_lower_expectation_dual


@dataclass(frozen=True)
class TrainingResult:
    """Outputs from one policy-training run."""

    q_values: np.ndarray
    greedy_policy: np.ndarray


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
) -> TrainingResult:
    """Train classical or robust tabular Q-learning."""
    action_codes = list(ACTION_INDEX)
    rng = np.random.default_rng(seed)
    q_values = np.zeros((STATE_COUNT, len(action_codes)), dtype=float)
    visits = np.zeros_like(q_values)
    initial_distribution = np.full(STATE_COUNT, 1.0 / STATE_COUNT, dtype=float)

    for _ in range(updates):
        state = int(rng.choice(np.arange(STATE_COUNT), p=initial_distribution))
        action = epsilon_greedy(q_values, state, exploration_rate, rng)
        next_state = int(rng.choice(np.arange(STATE_COUNT), p=kernel[state, :]))
        action_code = action_codes[action]
        reward = state_action_reward(next_state, action_code, reward_bands)
        future_values = np.max(q_values, axis=1)
        if robust_epsilon > 0:
            continuation = robust_lower_expectation_dual(
                reference_distribution=kernel[state, :],
                values=future_values,
                epsilon=robust_epsilon,
                cost_matrix=cost_matrix,
            )
        else:
            continuation = float(kernel[state, :] @ future_values)
        target = reward + discount * continuation
        visits[state, action] += 1.0
        learning_rate = 1.0 / (1.0 + visits[state, action])
        q_values[state, action] += learning_rate * (target - q_values[state, action])

    return TrainingResult(
        q_values=q_values,
        greedy_policy=np.argmax(q_values, axis=1),
    )
