"""Consolidate final experiment artifacts into one reporting table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


FINAL_RESULT_FIELDS = (
    "analysis",
    "reward_definition",
    "weighting",
    "beta",
    "gamma",
    "delta",
    "algorithm",
    "robust_epsilon",
    "radius_multiplier",
    "exact_policy",
    "modal_policy",
    "convergence_status",
    "mean_policy_agreement",
    "endpoint_name",
    "endpoint_value",
    "mean_raw_susceptibility",
    "carbapenem_use_rate",
    "mean_regret_to_oracle",
)


def _policy(values: list[int] | str) -> str:
    if isinstance(values, str):
        return "".join(values.split())
    return "".join(str(value) for value in values)


def _radius_multipliers(rows: list[dict[str, object]]) -> dict[float, float]:
    positive = sorted(
        {float(row["robust_epsilon"]) for row in rows if float(row["robust_epsilon"]) > 0}
    )
    return {
        epsilon: multiplier
        for epsilon, multiplier in zip(positive, (0.5, 1.0, 1.5, 2.0), strict=False)
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_final_results_rows(processed_dir: Path) -> list[dict[str, object]]:
    """Read completed artifacts and return consistently named result rows."""
    evaluation = json.loads(
        (processed_dir / "evaluation_metrics.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, object]] = []
    primary = evaluation["learned_policies"]
    primary_multipliers = _radius_multipliers(primary)
    for item in primary:
        metrics = item["exact_bellman_policy_metrics"]
        epsilon = float(item["robust_epsilon"])
        rows.append(
            {
                "analysis": "primary",
                "reward_definition": "susceptibility_minus_0.10_if_carbapenem",
                "weighting": evaluation["weighting"],
                "beta": "",
                "gamma": "",
                "delta": "",
                "algorithm": item["algorithm"],
                "robust_epsilon": epsilon,
                "radius_multiplier": primary_multipliers.get(epsilon, 0.0),
                "exact_policy": _policy(metrics["policy"]),
                "modal_policy": _policy(item["modal_policy_metrics"]["policy"]),
                "convergence_status": item["convergence"]["status"],
                "mean_policy_agreement": item["convergence"][
                    "mean_policy_agreement"
                ],
                "endpoint_name": "mean_adjusted_coverage",
                "endpoint_value": metrics["mean_adjusted_coverage"],
                "mean_raw_susceptibility": metrics["mean_raw_susceptibility"],
                "carbapenem_use_rate": metrics["carbapenem_use_rate"],
                "mean_regret_to_oracle": metrics["mean_regret_to_oracle"],
            }
        )

    specifications = (
        (
            "tested_weighting",
            "tested_weighting_summary.csv",
            "susceptibility_minus_0.10_if_carbapenem",
            "tested",
            "mean_adjusted_coverage",
        ),
        (
            "stewardship",
            "stewardship_full_training_summary.csv",
            "susceptibility_minus_breadth_and_pressure",
            "equal",
            "mean_reward",
        ),
        (
            "economic",
            "economic_full_training_summary.csv",
            "susceptibility_minus_breadth_cost_and_pressure",
            "equal",
            "mean_reward",
        ),
    )
    for analysis, filename, reward, weighting, endpoint_name in specifications:
        source_path = processed_dir / filename
        if not source_path.exists():
            continue
        source_rows = _read_csv(source_path)
        multipliers = _radius_multipliers(source_rows)
        for source in source_rows:
            epsilon = float(source["robust_epsilon"])
            rows.append(
                {
                    "analysis": analysis,
                    "reward_definition": reward,
                    "weighting": weighting,
                    "beta": source.get("beta", ""),
                    "gamma": source.get("gamma", ""),
                    "delta": source.get("delta", ""),
                    "algorithm": source["algorithm"],
                    "robust_epsilon": epsilon,
                    "radius_multiplier": multipliers.get(epsilon, 0.0),
                    "exact_policy": _policy(source["exact_policy"]),
                    "modal_policy": _policy(source["modal_policy"]),
                    "convergence_status": source["convergence_status"],
                    "mean_policy_agreement": source["mean_policy_agreement"],
                    "endpoint_name": endpoint_name,
                    "endpoint_value": source[endpoint_name],
                    "mean_raw_susceptibility": source[
                        "mean_raw_susceptibility"
                    ],
                    "carbapenem_use_rate": source.get(
                        "carbapenem_use_rate", source.get("use_carb", "")
                    ),
                    "mean_regret_to_oracle": source.get(
                        "mean_regret_to_oracle", ""
                    ),
                }
            )
    return rows


def write_final_results_table(processed_dir: Path) -> Path:
    """Write the consolidated result rows as a stable CSV artifact."""
    path = processed_dir / "final_results_table.csv"
    rows = build_final_results_rows(processed_dir)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path
