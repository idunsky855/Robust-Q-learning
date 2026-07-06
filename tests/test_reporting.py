from __future__ import annotations

import csv
import json
from pathlib import Path

from ears_q_learning.reporting import write_final_results_table


def _write_summary(path: Path, *, tested: bool = False) -> None:
    fields = [
        "algorithm",
        "robust_epsilon",
        "exact_policy",
        "modal_policy",
        "convergence_status",
        "mean_policy_agreement",
        "mean_raw_susceptibility",
        "mean_adjusted_coverage" if tested else "mean_reward",
        "carbapenem_use_rate" if tested else "use_carb",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "algorithm": "classical",
                "robust_epsilon": 0.0,
                "exact_policy": "0 0 2 2 2 2 2 2",
                "modal_policy": "0 0 2 2 2 2 2 2",
                "convergence_status": "policy_converged",
                "mean_policy_agreement": 1.0,
                "mean_raw_susceptibility": 0.95,
                "mean_adjusted_coverage" if tested else "mean_reward": 0.8,
                "carbapenem_use_rate" if tested else "use_carb": 0.5,
            }
        )


def test_final_results_table_consolidates_completed_artifacts(tmp_path: Path) -> None:
    evaluation = {
        "weighting": "equal",
        "learned_policies": [
            {
                "algorithm": "classical",
                "robust_epsilon": 0.0,
                "convergence": {
                    "status": "not_converged",
                    "mean_policy_agreement": 0.9,
                },
                "exact_bellman_policy_metrics": {
                    "policy": [2] * 8,
                    "mean_adjusted_coverage": 0.9,
                    "mean_raw_susceptibility": 1.0,
                    "carbapenem_use_rate": 1.0,
                    "mean_regret_to_oracle": 0.01,
                },
                "modal_policy_metrics": {"policy": [2] * 8},
            }
        ],
    }
    (tmp_path / "evaluation_metrics.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    _write_summary(tmp_path / "tested_weighting_summary.csv", tested=True)
    _write_summary(tmp_path / "stewardship_full_training_summary.csv")
    _write_summary(tmp_path / "economic_full_training_summary.csv")

    path = write_final_results_table(tmp_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert rows[0]["analysis"] == "primary"
    assert rows[0]["exact_policy"] == "22222222"
    assert rows[-1]["analysis"] == "economic"
