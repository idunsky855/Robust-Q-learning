"""Carbapenem-penalty sensitivity analysis with fixed model settings."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ears_q_learning.constants import ACTIONS, STATE_COUNT
from ears_q_learning.evaluation import evaluate_policy
from ears_q_learning.learning import solve_bellman_optimum
from ears_q_learning.mdp import estimate_reward_bands
from ears_q_learning.reproducibility import write_json
from ears_q_learning.types import CountryYearRow


def _validate_penalties(primary: float, penalties: tuple[float, ...]) -> tuple[float, ...]:
    """Return a sorted, unique sensitivity grid containing the primary value."""
    if primary < 0 or any(value < 0 for value in penalties):
        raise ValueError("Carbapenem penalties must be non-negative.")
    values = tuple(sorted(set(float(value) for value in penalties)))
    if primary not in values:
        raise ValueError("The sensitivity grid must contain the primary penalty.")
    return values


def run_penalty_sensitivity(
    *,
    rows: list[CountryYearRow],
    training_rows: list[CountryYearRow],
    state_lookup: dict[tuple[str, int], int],
    kernel: np.ndarray,
    cost_matrix: np.ndarray,
    training_summary: dict[str, object],
    primary_penalty: float,
    penalties: tuple[float, ...],
    decision_year_start: int,
    outcome_year_end: int,
) -> dict[str, object]:
    """Evaluate exact policies across a prespecified carbapenem-penalty grid."""
    penalty_grid = _validate_penalties(primary_penalty, penalties)
    classical_discount = float(
        training_summary["classical"]["configuration"]["discount"]
    )
    robust_discount = float(
        training_summary["robust"]["selected_configuration"]["discount"]
    )
    model_specs = [("classical", 0.0, classical_discount)]
    model_specs.extend(
        (
            "wasserstein_robust",
            float(radius["robust_epsilon"]),
            robust_discount,
        )
        for radius in training_summary["robust"]["radii"]
    )

    results: list[dict[str, object]] = []
    for penalty in penalty_grid:
        reward_bands = estimate_reward_bands(
            training_rows=training_rows,
            state_lookup=state_lookup,
            carbapenem_penalty=penalty,
        )
        for algorithm, epsilon, discount in model_specs:
            solution = solve_bellman_optimum(
                kernel=kernel,
                reward_bands=reward_bands,
                discount=discount,
                cost_matrix=cost_matrix,
                robust_epsilon=epsilon,
            )
            metrics = evaluate_policy(
                name=f"{algorithm}_penalty_{penalty:g}_epsilon_{epsilon:g}",
                policy=solution.greedy_policy,
                rows=rows,
                state_lookup=state_lookup,
                carbapenem_penalty=penalty,
                decision_year_start=decision_year_start,
                outcome_year_end=outcome_year_end,
            )
            results.append(
                {
                    "penalty": penalty,
                    "is_primary": penalty == primary_penalty,
                    "algorithm": algorithm,
                    "robust_epsilon": epsilon,
                    "discount": discount,
                    "policy": solution.greedy_policy.astype(int).tolist(),
                    "policy_labels": [
                        ACTIONS[int(action)].label
                        for action in solution.greedy_policy
                    ],
                    "bellman_iterations": solution.iterations,
                    "bellman_residual": solution.residual,
                    "mean_adjusted_coverage": metrics["mean_adjusted_coverage"],
                    "mean_raw_susceptibility": metrics["mean_raw_susceptibility"],
                    "carbapenem_use_rate": metrics["carbapenem_use_rate"],
                    "mean_regret_to_oracle": metrics["mean_regret_to_oracle"],
                    "worst_country": metrics["worst_country"],
                    "worst_country_reward": metrics["worst_country_reward"],
                }
            )

    return {
        "status": "penalty_sensitivity_completed",
        "interpretation": "sensitivity_only_not_penalty_optimization",
        "primary_penalty": primary_penalty,
        "penalty_grid": list(penalty_grid),
        "fixed_settings": {
            "transition_kernel": True,
            "state_thresholds": True,
            "classical_discount": classical_discount,
            "robust_discount": robust_discount,
            "robust_radii": [spec[1] for spec in model_specs[1:]],
            "policy_solution": "exact_bellman",
        },
        "state_count": STATE_COUNT,
        "results": results,
    }


def _write_csv(path: Path, results: list[dict[str, object]]) -> None:
    """Write a flat sensitivity table for external analysis."""
    fields = (
        "penalty",
        "is_primary",
        "algorithm",
        "robust_epsilon",
        "discount",
        "policy",
        "mean_adjusted_coverage",
        "mean_raw_susceptibility",
        "carbapenem_use_rate",
        "mean_regret_to_oracle",
        "worst_country",
        "worst_country_reward",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {field: result[field] for field in fields}
            row["policy"] = " ".join(str(action) for action in result["policy"])
            writer.writerow(row)


def _series_label(result: dict[str, object]) -> str:
    if result["algorithm"] == "classical":
        return "Classical"
    return f"Robust epsilon={float(result['robust_epsilon']):.4f}"


def _write_svg(
    path: Path,
    results: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    """Write a compact dependency-free line chart as SVG."""
    width, height = 900, 560
    left, right, top, bottom = 90, 30, 65, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [float(row[x_key]) for row in results]
    y_values = [float(row[y_key]) for row in results]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0
    y_padding = max((y_max - y_min) * 0.08, 0.002)
    y_min -= y_padding
    y_max += y_padding

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in results:
        grouped.setdefault(_series_label(row), []).append(row)
    colors = ("#17324d", "#b55233", "#48745f", "#b58a2a", "#6b5b73")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbf7ed"/>',
        f'<text x="{left}" y="34" font-family="Georgia, serif" font-size="22" fill="#172c3c">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#172c3c"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#172c3c"/>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        x = left + fraction * plot_width
        y = top + (1.0 - fraction) * plot_height
        elements.extend(
            [
                f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="#172c3c"/>',
                f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="Georgia, serif" font-size="12">{x_value:.3f}</text>',
                f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#172c3c"/>',
                f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Georgia, serif" font-size="12">{y_value:.3f}</text>',
            ]
        )
    for index, (label, rows) in enumerate(grouped.items()):
        color = colors[index % len(colors)]
        ordered = sorted(rows, key=lambda row: float(row[x_key]))
        points = " ".join(
            f"{x_position(float(row[x_key])):.1f},{y_position(float(row[y_key])):.1f}"
            for row in ordered
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for row in ordered:
            elements.append(
                f'<circle cx="{x_position(float(row[x_key])):.1f}" cy="{y_position(float(row[y_key])):.1f}" r="4" fill="{color}"/>'
            )
        legend_y = 52 + index * 18
        elements.extend(
            [
                f'<line x1="{width - 245}" y1="{legend_y}" x2="{width - 220}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{width - 212}" y="{legend_y + 4}" font-family="Georgia, serif" font-size="12">{label}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 24}" text-anchor="middle" font-family="Georgia, serif" font-size="15">{x_label}</text>',
            f'<text x="24" y="{top + plot_height / 2}" transform="rotate(-90 24 {top + plot_height / 2})" text-anchor="middle" font-family="Georgia, serif" font-size="15">{y_label}</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="ascii")


def write_penalty_sensitivity_artifacts(
    output_dir: Path, analysis: dict[str, object]
) -> dict[str, str]:
    """Persist sensitivity data and figures and return their paths."""
    results = analysis["results"]
    json_path = output_dir / "penalty_sensitivity.json"
    csv_path = output_dir / "penalty_sensitivity.csv"
    penalty_figure = output_dir / "penalty_vs_carbapenem_use.svg"
    tradeoff_figure = output_dir / "susceptibility_vs_carbapenem_use.svg"
    write_json(json_path, analysis)
    _write_csv(csv_path, results)
    _write_svg(
        penalty_figure,
        results,
        x_key="penalty",
        y_key="carbapenem_use_rate",
        x_label="Carbapenem penalty",
        y_label="Carbapenem-use rate",
        title="Penalty sensitivity of carbapenem use",
    )
    _write_svg(
        tradeoff_figure,
        results,
        x_key="carbapenem_use_rate",
        y_key="mean_raw_susceptibility",
        x_label="Carbapenem-use rate",
        y_label="Mean raw susceptibility",
        title="Susceptibility and carbapenem-use trade-off",
    )
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "penalty_figure": str(penalty_figure),
        "tradeoff_figure": str(tradeoff_figure),
    }
