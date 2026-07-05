"""Full tested-isolate weighting sensitivity analysis."""

from __future__ import annotations

import csv
from pathlib import Path

from ears_q_learning.config import LearningConfig
from ears_q_learning.evaluation import run_policy_evaluation
from ears_q_learning.mdp import (
    annual_state_distributions,
    calibrate_wasserstein_radius,
    estimate_reward_bands,
    myopic_policy,
    normalized_hamming_cost,
    transition_kernel,
)
from ears_q_learning.model_selection import run_model_selection
from ears_q_learning.preprocessing import build_transition_records
from ears_q_learning.reproducibility import write_json
from ears_q_learning.types import CountryYearRow


def run_weighting_sensitivity(
    *,
    rows: list[CountryYearRow],
    training_rows: list[CountryYearRow],
    state_lookup: dict[tuple[str, int], int],
    learning: LearningConfig,
    training_year_end: int,
    decision_year_start: int,
    outcome_year_end: int,
    smoothing_gamma: float,
    carbapenem_penalty: float,
    weighting: str,
) -> dict[str, object]:
    """Retune, train, and evaluate under one alternative weighting scheme."""
    if weighting != "tested":
        raise ValueError("The supported sensitivity weighting is 'tested'.")
    cost_matrix = normalized_hamming_cost()
    transitions = build_transition_records(training_rows, state_lookup, weighting)
    kernel = transition_kernel(transitions, smoothing_gamma)
    reward_bands = estimate_reward_bands(
        training_rows,
        state_lookup,
        carbapenem_penalty,
        weighting,
    )
    distributions = annual_state_distributions(
        training_rows, state_lookup, weighting
    )
    drift = calibrate_wasserstein_radius(distributions, cost_matrix)
    epsilon_star = float(drift["epsilon_star"])
    training_summary = run_model_selection(
        rows=rows,
        training_year_end=training_year_end,
        learning=learning,
        smoothing_gamma=smoothing_gamma,
        carbapenem_penalty=carbapenem_penalty,
        weighting=weighting,
        epsilon_star=epsilon_star,
    )
    learned_myopic_policy = myopic_policy(kernel, reward_bands)
    evaluation = run_policy_evaluation(
        rows=rows,
        state_lookup=state_lookup,
        training_summary=training_summary,
        myopic_policy=learned_myopic_policy,
        carbapenem_penalty=carbapenem_penalty,
        decision_year_start=decision_year_start,
        outcome_year_end=outcome_year_end,
        weighting=weighting,
    )
    return {
        "status": "weighting_sensitivity_completed",
        "interpretation": "secondary_tested_isolate_weighting",
        "weighting": weighting,
        "weight_definition": (
            "Mean next-year tested isolates across the three classes for transitions "
            "and evaluation; action-specific tested isolates for reward means."
        ),
        "epsilon_star": epsilon_star,
        "annual_distances": drift["annual_distances"],
        "reference_kernel": kernel.tolist(),
        "reward_bands": reward_bands,
        "myopic_policy": learned_myopic_policy.tolist(),
        "training": training_summary,
        "evaluation": evaluation,
    }


def write_weighting_sensitivity_artifacts(
    output_dir: Path, analysis: dict[str, object]
) -> dict[str, str]:
    """Write complete and compact tested-weighting artifacts."""
    json_path = output_dir / "tested_weighting_sensitivity.json"
    csv_path = output_dir / "tested_weighting_summary.csv"
    write_json(json_path, analysis)
    fields = (
        "algorithm",
        "robust_epsilon",
        "exact_policy",
        "modal_policy",
        "convergence_status",
        "mean_policy_agreement",
        "mean_adjusted_coverage",
        "mean_raw_susceptibility",
        "carbapenem_use_rate",
        "mean_regret_to_oracle",
    )
    rows: list[tuple[str, dict[str, object], dict[str, object]]] = []
    training = analysis["training"]
    evaluation_lookup = {
        (row["algorithm"], float(row["robust_epsilon"])): row
        for row in analysis["evaluation"]["learned_policies"]
    }
    rows.append(
        (
            "classical",
            training["classical"],
            evaluation_lookup[("classical", 0.0)],
        )
    )
    rows.extend(
        (
            "wasserstein_robust",
            output,
            evaluation_lookup[("wasserstein_robust", float(output["robust_epsilon"]))],
        )
        for output in training["robust"]["radii"]
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for algorithm, output, evaluated in rows:
            metrics = evaluated["exact_bellman_policy_metrics"]
            writer.writerow(
                {
                    "algorithm": algorithm,
                    "robust_epsilon": output["robust_epsilon"],
                    "exact_policy": " ".join(
                        str(value)
                        for value in output["exact_bellman_reference"]["policy"]
                    ),
                    "modal_policy": " ".join(
                        str(row["modal_action"])
                        for row in output["state_summaries"]
                    ),
                    "convergence_status": output["convergence"]["status"],
                    "mean_policy_agreement": output["convergence"][
                        "mean_policy_agreement"
                    ],
                    "mean_adjusted_coverage": metrics["mean_adjusted_coverage"],
                    "mean_raw_susceptibility": metrics["mean_raw_susceptibility"],
                    "carbapenem_use_rate": metrics["carbapenem_use_rate"],
                    "mean_regret_to_oracle": metrics["mean_regret_to_oracle"],
                }
            )
    return {"json": str(json_path), "csv": str(csv_path)}
