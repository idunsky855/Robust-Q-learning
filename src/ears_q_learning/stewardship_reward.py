"""Secondary stewardship-aware reward scenario."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from ears_q_learning.constants import ACTIONS, STATE_COUNT
from ears_q_learning.learning import solve_bellman_reward_matrix
from ears_q_learning.mdp import estimate_reward_bands, state_action_reward
from ears_q_learning.reproducibility import write_json
from ears_q_learning.state_space import state_to_bits
from ears_q_learning.types import CountryYearRow


def _validate_scenario(
    breadth_scores: dict[str, float],
    beta_grid: tuple[float, ...],
    delta_grid: tuple[float, ...],
) -> None:
    expected = {action.code for action in ACTIONS}
    if set(breadth_scores) != expected:
        raise ValueError(f"Breadth scores must define exactly {sorted(expected)}.")
    if any(not 0.0 <= score <= 1.0 for score in breadth_scores.values()):
        raise ValueError("Breadth scores must be between zero and one.")
    if not beta_grid or not delta_grid:
        raise ValueError("Stewardship coefficient grids must not be empty.")
    if any(value < 0 for value in (*beta_grid, *delta_grid)):
        raise ValueError("Stewardship coefficients must be non-negative.")


def build_stewardship_reward_matrix(
    training_rows: list[CountryYearRow],
    state_lookup: dict[tuple[str, int], int],
    breadth_scores: dict[str, float],
    beta: float,
    delta: float,
    cost_scores: dict[str, float] | None = None,
    gamma: float = 0.0,
) -> np.ndarray:
    """Build rewards from efficacy, breadth, and action-severity pressure."""
    efficacy_bands = estimate_reward_bands(
        training_rows=training_rows,
        state_lookup=state_lookup,
        carbapenem_penalty=0.0,
    )
    rewards = np.zeros((STATE_COUNT, len(ACTIONS)), dtype=float)
    costs = cost_scores or {action.code: 0.0 for action in ACTIONS}
    for next_state in range(STATE_COUNT):
        severity = sum(state_to_bits(next_state)) / 3.0
        for action_index, action in enumerate(ACTIONS):
            breadth = breadth_scores[action.code]
            efficacy = state_action_reward(next_state, action.code, efficacy_bands)
            rewards[next_state, action_index] = (
                efficacy
                - beta * breadth
                - gamma * costs[action.code]
                - delta * breadth * severity
            )
    return rewards


def _actual_susceptibility(row: CountryYearRow, action_code: str) -> float:
    resistance = {
        "3gc": row.resistance_3gc,
        "fq": row.resistance_fq,
        "carb": row.resistance_carb,
    }[action_code]
    return 1.0 - resistance / 100.0


def evaluate_stewardship_policy(
    *,
    policy: np.ndarray,
    rows: list[CountryYearRow],
    state_lookup: dict[tuple[str, int], int],
    breadth_scores: dict[str, float],
    beta: float,
    delta: float,
    decision_year_start: int,
    outcome_year_end: int,
    cost_scores: dict[str, float] | None = None,
    gamma: float = 0.0,
) -> dict[str, float | dict[str, float]]:
    by_key = {(row.country, row.year): row for row in rows}
    observations: list[dict[str, float | str]] = []
    action_counts: defaultdict[str, int] = defaultdict(int)
    costs = cost_scores or {action.code: 0.0 for action in ACTIONS}
    for current in rows:
        if not decision_year_start <= current.year < outcome_year_end:
            continue
        next_row = by_key.get((current.country, current.year + 1))
        if next_row is None:
            continue
        current_state = state_lookup[(current.country, current.year)]
        next_state = state_lookup[(next_row.country, next_row.year)]
        action = ACTIONS[int(policy[current_state])]
        susceptibility = _actual_susceptibility(next_row, action.code)
        breadth = breadth_scores[action.code]
        severity = sum(state_to_bits(next_state)) / 3.0
        breadth_penalty = beta * breadth
        pressure_penalty = delta * breadth * severity
        cost_score = costs[action.code]
        cost_penalty = gamma * cost_score
        action_counts[action.code] += 1
        observations.append(
            {
                "susceptibility": susceptibility,
                "breadth": breadth,
                "severity": severity,
                "breadth_penalty": breadth_penalty,
                "pressure_penalty": pressure_penalty,
                "cost_score": cost_score,
                "cost_penalty": cost_penalty,
                "reward": (
                    susceptibility
                    - breadth_penalty
                    - cost_penalty
                    - pressure_penalty
                ),
            }
        )
    if not observations:
        raise ValueError("No complete transitions are available for evaluation.")
    count = len(observations)
    return {
        "transition_count": count,
        "mean_reward": mean(float(row["reward"]) for row in observations),
        "mean_raw_susceptibility": mean(
            float(row["susceptibility"]) for row in observations
        ),
        "mean_breadth_score": mean(float(row["breadth"]) for row in observations),
        "mean_next_state_severity": mean(
            float(row["severity"]) for row in observations
        ),
        "mean_breadth_penalty": mean(
            float(row["breadth_penalty"]) for row in observations
        ),
        "mean_pressure_penalty": mean(
            float(row["pressure_penalty"]) for row in observations
        ),
        "mean_cost_score": mean(float(row["cost_score"]) for row in observations),
        "mean_cost_penalty": mean(
            float(row["cost_penalty"]) for row in observations
        ),
        "action_use_rates": {
            action.code: action_counts[action.code] / count for action in ACTIONS
        },
    }


def run_stewardship_reward_scenario(
    *,
    rows: list[CountryYearRow],
    training_rows: list[CountryYearRow],
    state_lookup: dict[tuple[str, int], int],
    kernel: np.ndarray,
    cost_matrix: np.ndarray,
    training_summary: dict[str, object],
    breadth_scores: dict[str, float],
    beta_grid: tuple[float, ...],
    delta_grid: tuple[float, ...],
    decision_year_start: int,
    outcome_year_end: int,
) -> dict[str, object]:
    """Solve and evaluate the stewardship reward over prespecified coefficients."""
    _validate_scenario(breadth_scores, beta_grid, delta_grid)
    classical_discount = float(
        training_summary["classical"]["configuration"]["discount"]
    )
    robust_discount = float(
        training_summary["robust"]["selected_configuration"]["discount"]
    )
    model_specs = [("classical", 0.0, classical_discount)]
    model_specs.extend(
        ("wasserstein_robust", float(row["robust_epsilon"]), robust_discount)
        for row in training_summary["robust"]["radii"]
    )
    results: list[dict[str, object]] = []
    for beta in sorted(set(beta_grid)):
        for delta in sorted(set(delta_grid)):
            reward_matrix = build_stewardship_reward_matrix(
                training_rows,
                state_lookup,
                breadth_scores,
                beta,
                delta,
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
                )
                results.append(
                    {
                        "beta": beta,
                        "delta": delta,
                        "cost_coefficient_gamma": 0.0,
                        "cost_component_status": "not_estimated",
                        "algorithm": algorithm,
                        "robust_epsilon": epsilon,
                        "discount": discount,
                        "policy": solution.greedy_policy.astype(int).tolist(),
                        "policy_labels": [
                            ACTIONS[int(action)].label
                            for action in solution.greedy_policy
                        ],
                        "bellman_residual": solution.residual,
                        **metrics,
                    }
                )
    return {
        "status": "stewardship_reward_scenario_completed",
        "interpretation": "secondary_normative_scenario_non_causal",
        "reward": (
            "susceptibility(next_state, action) - beta*breadth(action) "
            "- delta*breadth(action)*severity(next_state)"
        ),
        "breadth_scores": breadth_scores,
        "beta_grid": sorted(set(beta_grid)),
        "delta_grid": sorted(set(delta_grid)),
        "severity_definition": "mean of three binary next-state resistance indicators",
        "cost_component": (
            "Not estimated because class-level actions do not specify a comparable "
            "drug, dose, route, duration, country, or procurement price."
        ),
        "fixed_settings": {
            "transition_kernel": True,
            "state_thresholds": True,
            "robust_radii": [spec[1] for spec in model_specs[1:]],
            "policy_solution": "exact_bellman",
        },
        "results": results,
    }


def write_stewardship_reward_artifacts(
    output_dir: Path, analysis: dict[str, object]
) -> dict[str, str]:
    """Write the complete scenario and a flat analysis table."""
    json_path = output_dir / "stewardship_reward_scenario.json"
    csv_path = output_dir / "stewardship_reward_scenario.csv"
    figure_path = output_dir / "stewardship_reward_carbapenem_use.svg"
    write_json(json_path, analysis)
    fields = (
        "beta",
        "delta",
        "algorithm",
        "robust_epsilon",
        "policy",
        "mean_reward",
        "mean_raw_susceptibility",
        "mean_breadth_score",
        "mean_breadth_penalty",
        "mean_pressure_penalty",
        "use_3gc",
        "use_fq",
        "use_carb",
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in analysis["results"]:
            writer.writerow(
                {
                    "beta": result["beta"],
                    "delta": result["delta"],
                    "algorithm": result["algorithm"],
                    "robust_epsilon": result["robust_epsilon"],
                    "policy": " ".join(str(value) for value in result["policy"]),
                    "mean_reward": result["mean_reward"],
                    "mean_raw_susceptibility": result["mean_raw_susceptibility"],
                    "mean_breadth_score": result["mean_breadth_score"],
                    "mean_breadth_penalty": result["mean_breadth_penalty"],
                    "mean_pressure_penalty": result["mean_pressure_penalty"],
                    "use_3gc": result["action_use_rates"]["3gc"],
                    "use_fq": result["action_use_rates"]["fq"],
                    "use_carb": result["action_use_rates"]["carb"],
                }
            )
    _write_classical_heatmap(figure_path, analysis)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "figure": str(figure_path),
    }


def _write_classical_heatmap(path: Path, analysis: dict[str, object]) -> None:
    """Visualize classical carbapenem use across beta and delta values."""
    rows = [
        row for row in analysis["results"] if row["algorithm"] == "classical"
    ]
    betas = sorted({float(row["beta"]) for row in rows})
    deltas = sorted({float(row["delta"]) for row in rows})
    lookup = {
        (float(row["beta"]), float(row["delta"])): float(
            row["action_use_rates"]["carb"]
        )
        for row in rows
    }
    width, height = 760, 500
    left, top, cell_width, cell_height = 115, 85, 105, 75
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbf7ed"/>',
        '<text x="40" y="38" font-family="Georgia, serif" font-size="22" fill="#172c3c">Classical policy: carbapenem-use rate</text>',
        '<text x="40" y="62" font-family="Georgia, serif" font-size="13" fill="#4b5962">Secondary stewardship scenario; darker cells indicate greater use</text>',
    ]
    for column, delta in enumerate(deltas):
        x = left + column * cell_width
        elements.append(
            f'<text x="{x + cell_width / 2:.1f}" y="{top - 12}" text-anchor="middle" font-family="Georgia, serif" font-size="13">delta={delta:.2f}</text>'
        )
    for row_index, beta in enumerate(betas):
        y = top + row_index * cell_height
        elements.append(
            f'<text x="{left - 12}" y="{y + cell_height / 2 + 5:.1f}" text-anchor="end" font-family="Georgia, serif" font-size="13">beta={beta:.2f}</text>'
        )
        for column, delta in enumerate(deltas):
            value = lookup[(beta, delta)]
            red = round(235 - 165 * value)
            green = round(225 - 145 * value)
            blue = round(202 - 135 * value)
            x = left + column * cell_width
            text_color = "#ffffff" if value >= 0.65 else "#172c3c"
            elements.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="rgb({red},{green},{blue})" stroke="#fbf7ed" stroke-width="3"/>',
                    f'<text x="{x + cell_width / 2:.1f}" y="{y + cell_height / 2 + 6:.1f}" text-anchor="middle" font-family="Georgia, serif" font-size="17" fill="{text_color}">{value:.1%}</text>',
                ]
            )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="ascii")
