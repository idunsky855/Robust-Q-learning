"""Representative acquisition-cost and future-pressure reward scenarios."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from ears_q_learning.constants import ACTIONS
from ears_q_learning.learning import solve_bellman_reward_matrix
from ears_q_learning.stewardship_reward import (
    build_stewardship_reward_matrix,
    evaluate_stewardship_policy,
)
from ears_q_learning.stewardship_training import run_full_stewardship_training
from ears_q_learning.config import LearningConfig
from ears_q_learning.types import CountryYearRow


def load_normalized_cost_scores(path: Path) -> dict[str, object]:
    """Load representative one-DDD costs and normalize log costs to [0, 1]."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {action.code for action in ACTIONS}
    if {row["action"] for row in rows} != expected:
        raise ValueError(f"Cost input must define exactly {sorted(expected)}.")
    costs = {row["action"]: float(row["cost_per_ddd_gbp"]) for row in rows}
    if any(value <= 0 for value in costs.values()):
        raise ValueError("Representative costs must be positive.")
    denominator = max(math.log1p(value) for value in costs.values())
    scores = {
        action: math.log1p(value) / denominator for action, value in costs.items()
    }
    return {
        "cost_per_ddd_gbp": costs,
        "normalized_cost_scores": scores,
        "normalization": "log1p(cost) divided by maximum log1p(cost)",
        "representative_products": {
            row["action"]: {
                "product": row["representative_product"],
                "atc_code": row["atc_code"],
                "route": row["route"],
                "ddd_grams": float(row["ddd_grams"]),
                "source_url": row["source_url"],
                "source_period": row["source_period"],
            }
            for row in rows
        },
    }


def run_economic_pressure_scenario(
    *,
    rows: list[CountryYearRow],
    training_rows: list[CountryYearRow],
    state_lookup: dict[tuple[str, int], int],
    kernel: np.ndarray,
    cost_matrix: np.ndarray,
    training_summary: dict[str, object],
    breadth_scores: dict[str, float],
    beta: float,
    delta: float,
    gamma_grid: tuple[float, ...],
    cost_input: Path,
    decision_year_start: int,
    outcome_year_end: int,
) -> dict[str, object]:
    """Evaluate cost weights while retaining the pressure interaction."""
    cost_data = load_normalized_cost_scores(cost_input)
    cost_scores = cost_data["normalized_cost_scores"]
    classical_discount = float(
        training_summary["classical"]["configuration"]["discount"]
    )
    robust_discount = float(
        training_summary["robust"]["selected_configuration"]["discount"]
    )
    model_specs = [("classical", 0.0, classical_discount)]
    model_specs.extend(
        ("wasserstein_robust", float(item["robust_epsilon"]), robust_discount)
        for item in training_summary["robust"]["radii"]
    )
    results: list[dict[str, object]] = []
    for gamma in sorted(set(gamma_grid)):
        reward_matrix = build_stewardship_reward_matrix(
            training_rows,
            state_lookup,
            breadth_scores,
            beta,
            delta,
            cost_scores,
            gamma,
        )
        for algorithm, epsilon, discount in model_specs:
            solution = solve_bellman_reward_matrix(
                kernel=kernel,
                reward_matrix=reward_matrix,
                discount=discount,
                cost_matrix=cost_matrix,
                robust_epsilon=epsilon,
            )
            metrics = evaluate_stewardship_policy(
                policy=solution.greedy_policy,
                rows=rows,
                state_lookup=state_lookup,
                breadth_scores=breadth_scores,
                beta=beta,
                delta=delta,
                decision_year_start=decision_year_start,
                outcome_year_end=outcome_year_end,
                cost_scores=cost_scores,
                gamma=gamma,
            )
            results.append(
                {
                    "algorithm": algorithm,
                    "robust_epsilon": epsilon,
                    "beta": beta,
                    "gamma": gamma,
                    "delta": delta,
                    "policy": solution.greedy_policy.astype(int).tolist(),
                    "policy_labels": [
                        ACTIONS[int(action)].label for action in solution.greedy_policy
                    ],
                    **metrics,
                }
            )
    return {
        "status": "economic_pressure_scenario_completed",
        "interpretation": "secondary_normative_non_causal_cost_scenario",
        "reward": (
            "susceptibility - beta*breadth - gamma*normalized_acquisition_cost "
            "- delta*breadth*next_state_severity"
        ),
        "beta": beta,
        "delta": delta,
        "gamma_grid": sorted(set(gamma_grid)),
        "cost_data": cost_data,
        "future_pressure_interpretation": (
            "The delta term is a normative action-severity interaction. Published "
            "European panel evidence reports usage-associated resistance persisting "
            "for at least four years, but this model does not estimate a causal effect."
        ),
        "results": results,
    }


def write_economic_pressure_artifacts(
    output_dir: Path, analysis: dict[str, object]
) -> dict[str, str]:
    """Write full and compact economic-pressure outputs."""
    from ears_q_learning.reproducibility import write_json

    json_path = output_dir / "economic_pressure_scenario.json"
    csv_path = output_dir / "economic_pressure_summary.csv"
    write_json(json_path, analysis)
    fields = (
        "algorithm",
        "robust_epsilon",
        "beta",
        "gamma",
        "delta",
        "policy",
        "mean_reward",
        "mean_raw_susceptibility",
        "mean_cost_score",
        "mean_cost_penalty",
        "mean_pressure_penalty",
        "use_3gc",
        "use_fq",
        "use_carb",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in analysis["results"]:
            writer.writerow(
                {
                    "algorithm": row["algorithm"],
                    "robust_epsilon": row["robust_epsilon"],
                    "beta": row["beta"],
                    "gamma": row["gamma"],
                    "delta": row["delta"],
                    "policy": " ".join(str(value) for value in row["policy"]),
                    "mean_reward": row["mean_reward"],
                    "mean_raw_susceptibility": row["mean_raw_susceptibility"],
                    "mean_cost_score": row["mean_cost_score"],
                    "mean_cost_penalty": row["mean_cost_penalty"],
                    "mean_pressure_penalty": row["mean_pressure_penalty"],
                    "use_3gc": row["action_use_rates"]["3gc"],
                    "use_fq": row["action_use_rates"]["fq"],
                    "use_carb": row["action_use_rates"]["carb"],
                }
            )
    return {"json": str(json_path), "csv": str(csv_path)}


def run_full_economic_training(
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
    beta: float,
    gamma: float,
    delta: float,
    cost_input: Path,
) -> dict[str, object]:
    """Fully tune and train one prespecified cost-adjusted reward scenario."""
    cost_data = load_normalized_cost_scores(cost_input)
    analysis = run_full_stewardship_training(
        rows=rows,
        evaluation_state_lookup=evaluation_state_lookup,
        training_year_end=training_year_end,
        decision_year_start=decision_year_start,
        outcome_year_end=outcome_year_end,
        learning=learning,
        smoothing_gamma=smoothing_gamma,
        weighting=weighting,
        epsilon_star=epsilon_star,
        breadth_scores=breadth_scores,
        scenarios=((beta, delta),),
        cost_scores=cost_data["normalized_cost_scores"],
        gamma=gamma,
    )
    analysis["status"] = "full_economic_training_completed"
    analysis["interpretation"] = "secondary_normative_non_causal_cost_scenario"
    analysis["cost_data"] = cost_data
    return analysis
