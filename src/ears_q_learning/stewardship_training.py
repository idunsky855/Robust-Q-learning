"""Full model selection and seeded training for stewardship rewards."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from ears_q_learning.config import LearningConfig
from ears_q_learning.learning import (
    BellmanSolution,
    solve_bellman_reward_matrix,
    train_q_learning_reward_matrix,
)
from ears_q_learning.model_selection import (
    PAPER_DEFAULTS,
    SelectedConfiguration,
    _bellman_summary,
    _convergence_summary,
    build_training_context,
    summarize_seeded_policies,
)
from ears_q_learning.stewardship_reward import (
    build_stewardship_reward_matrix,
    evaluate_stewardship_policy,
)
from ears_q_learning.types import CountryYearRow


@dataclass(frozen=True)
class StewardshipTrainingContext:
    """One training window with its explicit stewardship reward matrix."""

    rows: list[CountryYearRow]
    state_lookup: dict[tuple[str, int], int]
    kernel: np.ndarray
    reward_matrix: np.ndarray
    cost_matrix: np.ndarray
    epsilon_star: float


def build_stewardship_training_context(
    *,
    rows: list[CountryYearRow],
    training_year_end: int,
    smoothing_gamma: float,
    weighting: str,
    breadth_scores: dict[str, float],
    beta: float,
    delta: float,
    cost_scores: dict[str, float] | None = None,
    gamma: float = 0.0,
) -> StewardshipTrainingContext:
    """Estimate one fold while fitting rewards only on its training period."""
    base = build_training_context(
        rows=rows,
        training_year_end=training_year_end,
        smoothing_gamma=smoothing_gamma,
        carbapenem_penalty=0.0,
        weighting=weighting,
    )
    reward_matrix = build_stewardship_reward_matrix(
        training_rows=base.rows,
        state_lookup=base.state_lookup,
        breadth_scores=breadth_scores,
        beta=beta,
        delta=delta,
        cost_scores=cost_scores,
        gamma=gamma,
    )
    return StewardshipTrainingContext(
        rows=base.rows,
        state_lookup=base.state_lookup,
        kernel=base.kernel,
        reward_matrix=reward_matrix,
        cost_matrix=base.cost_matrix,
        epsilon_star=base.epsilon_star,
    )


def _train_candidate(
    context: StewardshipTrainingContext,
    *,
    discount: float,
    exploration_rate: float,
    updates: int,
    seed: int,
    robust_epsilon: float,
    q_norm: int,
):
    return train_q_learning_reward_matrix(
        kernel=context.kernel,
        reward_matrix=context.reward_matrix,
        discount=discount,
        exploration_rate=exploration_rate,
        updates=updates,
        seed=seed,
        robust_epsilon=robust_epsilon,
        cost_matrix=context.cost_matrix,
        q_norm=q_norm,
    )


def _validation_reward(
    *,
    rows: list[CountryYearRow],
    context: StewardshipTrainingContext,
    policy: np.ndarray,
    breadth_scores: dict[str, float],
    beta: float,
    delta: float,
    decision_year: int,
    cost_scores: dict[str, float] | None = None,
    gamma: float = 0.0,
) -> float:
    metrics = evaluate_stewardship_policy(
        policy=policy,
        rows=rows,
        state_lookup=context.state_lookup,
        breadth_scores=breadth_scores,
        beta=beta,
        delta=delta,
        decision_year_start=decision_year,
        outcome_year_end=decision_year + 1,
        cost_scores=cost_scores,
        gamma=gamma,
    )
    return float(metrics["mean_reward"])


def tune_stewardship_configuration(
    *,
    rows: list[CountryYearRow],
    learning: LearningConfig,
    smoothing_gamma: float,
    weighting: str,
    breadth_scores: dict[str, float],
    beta: float,
    delta: float,
    robust: bool,
    cost_scores: dict[str, float] | None = None,
    gamma: float = 0.0,
) -> SelectedConfiguration:
    """Tune one algorithm using the original rolling-origin protocol."""
    candidates: list[SelectedConfiguration] = []
    for discount in learning.discount_grid:
        for exploration_rate in learning.exploration_grid:
            seed_scores: list[float] = []
            for seed in learning.tuning_seeds:
                fold_scores: list[float] = []
                for decision_year in (2017, 2018):
                    context = build_stewardship_training_context(
                        rows=rows,
                        training_year_end=decision_year,
                        smoothing_gamma=smoothing_gamma,
                        weighting=weighting,
                        breadth_scores=breadth_scores,
                        beta=beta,
                        delta=delta,
                        cost_scores=cost_scores,
                        gamma=gamma,
                    )
                    result = _train_candidate(
                        context,
                        discount=float(discount),
                        exploration_rate=float(exploration_rate),
                        updates=learning.updates,
                        seed=int(seed),
                        robust_epsilon=context.epsilon_star if robust else 0.0,
                        q_norm=learning.q_norm,
                    )
                    fold_scores.append(
                        _validation_reward(
                            rows=rows,
                            context=context,
                            policy=result.greedy_policy,
                            breadth_scores=breadth_scores,
                            beta=beta,
                            delta=delta,
                            decision_year=decision_year,
                            cost_scores=cost_scores,
                            gamma=gamma,
                        )
                    )
                seed_scores.append(float(mean(fold_scores)))
            candidates.append(
                SelectedConfiguration(
                    discount=float(discount),
                    exploration_rate=float(exploration_rate),
                    mean_reward=float(mean(seed_scores)),
                    seed_std=float(pstdev(seed_scores)),
                    default_distance=(
                        abs(float(discount) - PAPER_DEFAULTS[0])
                        + abs(float(exploration_rate) - PAPER_DEFAULTS[1])
                    ),
                    validation_scores=seed_scores,
                    selection_trace=[],
                )
            )
    ordered = sorted(
        candidates,
        key=lambda item: (-item.mean_reward, item.seed_std, item.default_distance),
    )
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
    selected = ordered[0]
    return SelectedConfiguration(
        discount=selected.discount,
        exploration_rate=selected.exploration_rate,
        mean_reward=selected.mean_reward,
        seed_std=selected.seed_std,
        default_distance=selected.default_distance,
        validation_scores=selected.validation_scores,
        selection_trace=trace,
    )


def _final_seeded_training(
    *,
    context: StewardshipTrainingContext,
    selected: SelectedConfiguration,
    learning: LearningConfig,
    robust_epsilon: float,
) -> dict[str, object]:
    exact: BellmanSolution = solve_bellman_reward_matrix(
        kernel=context.kernel,
        reward_matrix=context.reward_matrix,
        discount=selected.discount,
        cost_matrix=context.cost_matrix,
        robust_epsilon=robust_epsilon,
    )
    seed_results: list[dict[str, object]] = []
    for seed in learning.final_seeds:
        result = _train_candidate(
            context,
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
        "convergence": _convergence_summary(seed_results, exact),
    }


def _evaluate_training_output(
    *,
    output: dict[str, object],
    rows: list[CountryYearRow],
    state_lookup: dict[tuple[str, int], int],
    breadth_scores: dict[str, float],
    beta: float,
    delta: float,
    decision_year_start: int,
    outcome_year_end: int,
    cost_scores: dict[str, float] | None = None,
    gamma: float = 0.0,
) -> dict[str, object]:
    exact_policy = np.asarray(output["exact_bellman_reference"]["policy"], dtype=int)
    modal_policy = np.asarray(
        [row["modal_action"] for row in output["state_summaries"]], dtype=int
    )
    seed_metrics = [
        {
            "seed": row["seed"],
            **evaluate_stewardship_policy(
                policy=np.asarray(row["policy"], dtype=int),
                rows=rows,
                state_lookup=state_lookup,
                breadth_scores=breadth_scores,
                beta=beta,
                delta=delta,
                decision_year_start=decision_year_start,
                outcome_year_end=outcome_year_end,
                cost_scores=cost_scores,
                gamma=gamma,
            ),
        }
        for row in output["seed_results"]
    ]
    return {
        "exact_policy_metrics": evaluate_stewardship_policy(
            policy=exact_policy,
            rows=rows,
            state_lookup=state_lookup,
            breadth_scores=breadth_scores,
            beta=beta,
            delta=delta,
            decision_year_start=decision_year_start,
            outcome_year_end=outcome_year_end,
            cost_scores=cost_scores,
            gamma=gamma,
        ),
        "modal_policy_metrics": evaluate_stewardship_policy(
            policy=modal_policy,
            rows=rows,
            state_lookup=state_lookup,
            breadth_scores=breadth_scores,
            beta=beta,
            delta=delta,
            decision_year_start=decision_year_start,
            outcome_year_end=outcome_year_end,
            cost_scores=cost_scores,
            gamma=gamma,
        ),
        "seed_metrics": seed_metrics,
    }


def run_full_stewardship_training(
    *,
    rows: list[CountryYearRow],
    evaluation_state_lookup: dict[tuple[str, int], int],
    training_year_end: int,
    decision_year_start: int,
    outcome_year_end: int,
    learning: LearningConfig,
    smoothing_gamma: float,
    weighting: str,
    epsilon_star: float,
    breadth_scores: dict[str, float],
    scenarios: tuple[tuple[float, float], ...],
    cost_scores: dict[str, float] | None = None,
    gamma: float = 0.0,
) -> dict[str, object]:
    """Fully tune, train, and evaluate every prespecified scenario."""
    scenario_outputs: list[dict[str, object]] = []
    for beta, delta in scenarios:
        final_context = build_stewardship_training_context(
            rows=rows,
            training_year_end=training_year_end,
            smoothing_gamma=smoothing_gamma,
            weighting=weighting,
            breadth_scores=breadth_scores,
            beta=beta,
            delta=delta,
            cost_scores=cost_scores,
            gamma=gamma,
        )
        classical_selected = tune_stewardship_configuration(
            rows=rows,
            learning=learning,
            smoothing_gamma=smoothing_gamma,
            weighting=weighting,
            breadth_scores=breadth_scores,
            beta=beta,
            delta=delta,
            robust=False,
            cost_scores=cost_scores,
            gamma=gamma,
        )
        robust_selected = tune_stewardship_configuration(
            rows=rows,
            learning=learning,
            smoothing_gamma=smoothing_gamma,
            weighting=weighting,
            breadth_scores=breadth_scores,
            beta=beta,
            delta=delta,
            robust=True,
            cost_scores=cost_scores,
            gamma=gamma,
        )
        classical = _final_seeded_training(
            context=final_context,
            selected=classical_selected,
            learning=learning,
            robust_epsilon=0.0,
        )
        robust = [
            _final_seeded_training(
                context=final_context,
                selected=robust_selected,
                learning=learning,
                robust_epsilon=float(multiplier * epsilon_star),
            )
            for multiplier in learning.epsilon_multipliers
        ]
        scenario_outputs.append(
            {
                "beta": beta,
                "delta": delta,
                "classical": {
                    "training": classical,
                    "evaluation": _evaluate_training_output(
                        output=classical,
                        rows=rows,
                        state_lookup=evaluation_state_lookup,
                        breadth_scores=breadth_scores,
                        beta=beta,
                        delta=delta,
                        decision_year_start=decision_year_start,
                        outcome_year_end=outcome_year_end,
                        cost_scores=cost_scores,
                        gamma=gamma,
                    ),
                },
                "robust": [
                    {
                        "training": output,
                        "evaluation": _evaluate_training_output(
                            output=output,
                            rows=rows,
                            state_lookup=evaluation_state_lookup,
                            breadth_scores=breadth_scores,
                            beta=beta,
                            delta=delta,
                            decision_year_start=decision_year_start,
                            outcome_year_end=outcome_year_end,
                            cost_scores=cost_scores,
                            gamma=gamma,
                        ),
                    }
                    for output in robust
                ],
            }
        )
    return {
        "status": "full_stewardship_training_completed",
        "interpretation": "secondary_normative_scenarios_non_causal",
        "breadth_scores": breadth_scores,
        "gamma": gamma,
        "cost_scores": cost_scores,
        "scenarios": scenario_outputs,
    }


def write_full_stewardship_summary(
    path: Path, analysis: dict[str, object]
) -> None:
    """Write one compact row per scenario, algorithm, and robustness radius."""
    fields = (
        "beta",
        "delta",
        "gamma",
        "algorithm",
        "robust_epsilon",
        "discount",
        "exploration_rate",
        "exact_policy",
        "modal_policy",
        "convergence_status",
        "mean_policy_agreement",
        "minimum_policy_agreement",
        "mean_bellman_sup_norm_error",
        "mean_raw_susceptibility",
        "mean_reward",
        "use_3gc",
        "use_fq",
        "use_carb",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scenario in analysis["scenarios"]:
            outputs = [("classical", scenario["classical"])]
            outputs.extend(("wasserstein_robust", item) for item in scenario["robust"])
            for algorithm, output in outputs:
                training = output["training"]
                metrics = output["evaluation"]["exact_policy_metrics"]
                convergence = training["convergence"]
                writer.writerow(
                    {
                        "beta": scenario["beta"],
                        "delta": scenario["delta"],
                        "gamma": analysis.get("gamma", 0.0),
                        "algorithm": algorithm,
                        "robust_epsilon": training["robust_epsilon"],
                        "discount": training["configuration"]["discount"],
                        "exploration_rate": training["configuration"][
                            "exploration_rate"
                        ],
                        "exact_policy": " ".join(
                            str(value)
                            for value in training["exact_bellman_reference"]["policy"]
                        ),
                        "modal_policy": " ".join(
                            str(row["modal_action"])
                            for row in training["state_summaries"]
                        ),
                        "convergence_status": convergence["status"],
                        "mean_policy_agreement": convergence[
                            "mean_policy_agreement"
                        ],
                        "minimum_policy_agreement": convergence[
                            "minimum_policy_agreement"
                        ],
                        "mean_bellman_sup_norm_error": convergence[
                            "bellman_sup_norm_error"
                        ]["mean"],
                        "mean_raw_susceptibility": metrics[
                            "mean_raw_susceptibility"
                        ],
                        "mean_reward": metrics["mean_reward"],
                        "use_3gc": metrics["action_use_rates"]["3gc"],
                        "use_fq": metrics["action_use_rates"]["fq"],
                        "use_carb": metrics["action_use_rates"]["carb"],
                    }
                )
