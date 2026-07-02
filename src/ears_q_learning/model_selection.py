"""Rolling-origin model selection and seeded policy training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean, pstdev

import numpy as np

from ears_q_learning.config import LearningConfig
from ears_q_learning.constants import ACTIONS, STATE_COUNT
from ears_q_learning.learning import (
    BellmanSolution,
    TrainingResult,
    solve_bellman_optimum,
    train_classical_q_learning,
    train_robust_q_learning,
)
from ears_q_learning.mdp import (
    annual_state_distributions,
    calibrate_wasserstein_radius,
    estimate_reward_bands,
    normalized_hamming_cost,
    observed_reward,
    transition_kernel,
)
from ears_q_learning.preprocessing import build_transition_records
from ears_q_learning.state_space import encode_state, fit_thresholds
from ears_q_learning.types import CountryYearRow

PAPER_DEFAULTS = (0.45, 0.1)


@dataclass(frozen=True)
class TrainingContext:
    """Model components estimated from one training window."""

    rows: list[CountryYearRow]
    state_lookup: dict[tuple[str, int], int]
    kernel: np.ndarray
    reward_bands: dict[str, dict[int, float]]
    cost_matrix: np.ndarray
    epsilon_star: float


@dataclass(frozen=True)
class SelectedConfiguration:
    """Chosen hyperparameter configuration and validation trace."""

    discount: float
    exploration_rate: float
    mean_reward: float
    seed_std: float
    default_distance: float
    validation_scores: list[float]
    selection_trace: list[dict[str, object]]


def _state_lookup_for_rows(
    rows: list[CountryYearRow],
    training_year_end: int,
) -> dict[tuple[str, int], int]:
    training_rows = [row for row in rows if row.year <= training_year_end]
    thresholds = fit_thresholds(training_rows)
    return {
        (row.country, row.year): encode_state(row, thresholds)
        for row in rows
        if row.year <= training_year_end + 1
    }


def build_training_context(
    rows: list[CountryYearRow],
    training_year_end: int,
    smoothing_gamma: float,
    carbapenem_penalty: float,
    weighting: str,
) -> TrainingContext:
    """Estimate model components for one training cutoff year."""
    training_rows = [row for row in rows if row.year <= training_year_end]
    if not training_rows:
        raise ValueError("At least one training row is required.")
    state_lookup = _state_lookup_for_rows(rows, training_year_end)
    transitions = build_transition_records(training_rows, state_lookup, weighting)
    kernel = transition_kernel(transitions, smoothing_gamma)
    reward_bands = estimate_reward_bands(training_rows, state_lookup, carbapenem_penalty)
    cost_matrix = normalized_hamming_cost()
    distributions = annual_state_distributions(training_rows, state_lookup)
    try:
        epsilon_star = float(
            calibrate_wasserstein_radius(distributions, cost_matrix)["epsilon_star"]
        )
    except ValueError:
        epsilon_star = 0.0
    return TrainingContext(
        rows=training_rows,
        state_lookup=state_lookup,
        kernel=kernel,
        reward_bands=reward_bands,
        cost_matrix=cost_matrix,
        epsilon_star=epsilon_star,
    )


def validation_reward(
    rows: list[CountryYearRow],
    state_lookup: dict[tuple[str, int], int],
    policy: np.ndarray,
    decision_year: int,
    carbapenem_penalty: float,
) -> float:
    """Score one policy on one rolling-origin validation transition."""
    by_country_year = {(row.country, row.year): row for row in rows}
    action_codes = [action.code for action in ACTIONS]
    rewards: list[float] = []
    for current in rows:
        if current.year != decision_year:
            continue
        next_row = by_country_year.get((current.country, decision_year + 1))
        if next_row is None:
            continue
        state = state_lookup.get((current.country, decision_year))
        if state is None:
            continue
        action_code = action_codes[int(policy[state])]
        rewards.append(observed_reward(next_row, action_code, carbapenem_penalty))
    if not rewards:
        raise ValueError(f"No validation transitions found for {decision_year}.")
    return float(mean(rewards))


def _train_candidate(
    context: TrainingContext,
    discount: float,
    exploration_rate: float,
    updates: int,
    seed: int,
    robust_epsilon: float,
    q_norm: int,
) -> TrainingResult:
    if robust_epsilon > 0:
        return train_robust_q_learning(
            kernel=context.kernel,
            reward_bands=context.reward_bands,
            discount=discount,
            exploration_rate=exploration_rate,
            updates=updates,
            seed=seed,
            robust_epsilon=robust_epsilon,
            cost_matrix=context.cost_matrix,
            q_norm=q_norm,
        )
    return train_classical_q_learning(
        kernel=context.kernel,
        reward_bands=context.reward_bands,
        discount=discount,
        exploration_rate=exploration_rate,
        updates=updates,
        seed=seed,
        cost_matrix=context.cost_matrix,
        q_norm=q_norm,
    )


def tune_configuration(
    rows: list[CountryYearRow],
    learning: LearningConfig,
    smoothing_gamma: float,
    carbapenem_penalty: float,
    weighting: str,
    robust: bool,
) -> SelectedConfiguration:
    """Select hyperparameters using two rolling-origin validation folds."""
    candidates: list[SelectedConfiguration] = []
    for discount in learning.discount_grid:
        for exploration_rate in learning.exploration_grid:
            seed_fold_scores: list[float] = []
            for seed in learning.tuning_seeds:
                fold_scores: list[float] = []
                for decision_year in (2017, 2018):
                    context = build_training_context(
                        rows=rows,
                        training_year_end=decision_year,
                        smoothing_gamma=smoothing_gamma,
                        carbapenem_penalty=carbapenem_penalty,
                        weighting=weighting,
                    )
                    robust_epsilon = context.epsilon_star if robust else 0.0
                    result = _train_candidate(
                        context=context,
                        discount=float(discount),
                        exploration_rate=float(exploration_rate),
                        updates=learning.updates,
                        seed=int(seed),
                        robust_epsilon=robust_epsilon,
                        q_norm=learning.q_norm,
                    )
                    fold_scores.append(
                        validation_reward(
                            rows=rows,
                            state_lookup=context.state_lookup,
                            policy=result.greedy_policy,
                            decision_year=decision_year,
                            carbapenem_penalty=carbapenem_penalty,
                        )
                    )
                seed_fold_scores.append(float(mean(fold_scores)))
            candidates.append(
                SelectedConfiguration(
                    discount=float(discount),
                    exploration_rate=float(exploration_rate),
                    mean_reward=float(mean(seed_fold_scores)),
                    seed_std=float(pstdev(seed_fold_scores))
                    if len(seed_fold_scores) > 1
                    else 0.0,
                    default_distance=abs(float(discount) - PAPER_DEFAULTS[0])
                    + abs(float(exploration_rate) - PAPER_DEFAULTS[1]),
                    validation_scores=seed_fold_scores,
                    selection_trace=[],
                )
            )
    ordered = sorted(
        candidates,
        key=lambda item: (-item.mean_reward, item.seed_std, item.default_distance),
    )
    selected = ordered[0]
    trace = [
        {
            "discount": item.discount,
            "exploration_rate": item.exploration_rate,
            "mean_reward": item.mean_reward,
            "seed_std": item.seed_std,
            "default_distance": item.default_distance,
            "validation_scores": item.validation_scores,
            "rank": rank,
        }
        for rank, item in enumerate(ordered, start=1)
    ]
    return SelectedConfiguration(
        discount=selected.discount,
        exploration_rate=selected.exploration_rate,
        mean_reward=selected.mean_reward,
        seed_std=selected.seed_std,
        default_distance=selected.default_distance,
        validation_scores=selected.validation_scores,
        selection_trace=trace,
    )


def summarize_seeded_policies(
    seed_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize modal action and agreement for each state."""
    summaries: list[dict[str, object]] = []
    for state in range(STATE_COUNT):
        actions = [int(result["policy"][state]) for result in seed_results]
        counts = Counter(actions)
        modal_action, modal_count = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        summaries.append(
            {
                "state": state,
                "modal_action": modal_action,
                "modal_action_label": ACTIONS[modal_action].label,
                "agreement": modal_count / len(actions),
                "action_counts": dict(sorted(counts.items())),
            }
        )
    return summaries


def final_seeded_training(
    context: TrainingContext,
    selected: SelectedConfiguration,
    learning: LearningConfig,
    robust_epsilon: float,
) -> dict[str, object]:
    """Train final policies for all configured final seeds."""
    exact = solve_bellman_optimum(
        kernel=context.kernel,
        reward_bands=context.reward_bands,
        discount=selected.discount,
        cost_matrix=context.cost_matrix,
        robust_epsilon=robust_epsilon,
    )
    seed_results: list[dict[str, object]] = []
    for seed in learning.final_seeds:
        result = _train_candidate(
            context=context,
            discount=selected.discount,
            exploration_rate=selected.exploration_rate,
            updates=learning.updates,
            seed=int(seed),
            robust_epsilon=robust_epsilon,
            q_norm=learning.q_norm,
        )
        seed_results.append(
            {
                "seed": int(seed),
                "policy": result.greedy_policy.astype(int).tolist(),
                "q_values": result.q_values.tolist(),
                "visits": result.visits.tolist(),
                "bellman_sup_norm_error": float(
                    np.max(np.abs(result.q_values - exact.q_values))
                ),
                "policy_agreement_with_bellman": float(
                    np.mean(result.greedy_policy == exact.greedy_policy)
                ),
            }
        )
    return {
        "configuration": selected.__dict__,
        "robust_epsilon": robust_epsilon,
        "seed_results": seed_results,
        "state_summaries": summarize_seeded_policies(seed_results),
        "exact_bellman_reference": _bellman_summary(exact),
    }


def _bellman_summary(solution: BellmanSolution) -> dict[str, object]:
    """Convert an exact Bellman solution to a machine-readable summary."""
    return {
        "policy": solution.greedy_policy.astype(int).tolist(),
        "q_values": solution.q_values.tolist(),
        "iterations": solution.iterations,
        "residual": solution.residual,
    }


def run_model_selection(
    rows: list[CountryYearRow],
    training_year_end: int,
    learning: LearningConfig,
    smoothing_gamma: float,
    carbapenem_penalty: float,
    weighting: str,
    epsilon_star: float,
) -> dict[str, object]:
    """Run tuning and final seeded training for classical and robust policies."""
    final_context = build_training_context(
        rows=rows,
        training_year_end=training_year_end,
        smoothing_gamma=smoothing_gamma,
        carbapenem_penalty=carbapenem_penalty,
        weighting=weighting,
    )
    classical_selected = tune_configuration(
        rows=rows,
        learning=learning,
        smoothing_gamma=smoothing_gamma,
        carbapenem_penalty=carbapenem_penalty,
        weighting=weighting,
        robust=False,
    )
    robust_selected = tune_configuration(
        rows=rows,
        learning=learning,
        smoothing_gamma=smoothing_gamma,
        carbapenem_penalty=carbapenem_penalty,
        weighting=weighting,
        robust=True,
    )
    robust_radii = [multiplier * epsilon_star for multiplier in learning.epsilon_multipliers]
    return {
        "status": "training_completed",
        "folds": [
            {"train_through": 2017, "validate_transition": "2017->2018"},
            {"train_through": 2018, "validate_transition": "2018->2019"},
        ],
        "classical": final_seeded_training(
            context=final_context,
            selected=classical_selected,
            learning=learning,
            robust_epsilon=0.0,
        ),
        "robust": {
            "selected_configuration": robust_selected.__dict__,
            "radii": [
                final_seeded_training(
                    context=final_context,
                    selected=robust_selected,
                    learning=learning,
                    robust_epsilon=float(radius),
                )
                for radius in robust_radii
            ],
        },
    }
