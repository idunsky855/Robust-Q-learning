from pathlib import Path

from ears_q_learning.types import CountryYearRow
from ears_q_learning.walkthrough import write_walkthrough


def _rows() -> list[CountryYearRow]:
    return [
        CountryYearRow(
            country="Aland",
            year=year,
            resistance_3gc=10.0 + year - 2015,
            resistance_fq=20.0 + year - 2015,
            resistance_carb=0.1,
            tested_3gc=100,
            tested_fq=100,
            tested_carb=100,
        )
        for year in range(2015, 2025)
    ]


def _policy(algorithm: str, epsilon: float, action: int) -> dict[str, object]:
    yearly = [
        {
            "outcome_year": year,
            "mean_adjusted_coverage": 0.90,
        }
        for year in range(2021, 2025)
    ]
    metrics = {
        "policy": [action] * 8,
        "mean_adjusted_coverage": 0.90,
        "mean_raw_susceptibility": 0.95 if action == 0 else 0.99,
        "carbapenem_use_rate": 0.0 if action == 0 else 1.0,
        "yearly_performance": yearly,
    }
    return {
        "algorithm": algorithm,
        "robust_epsilon": epsilon,
        "exact_bellman_policy_metrics": metrics,
    }


def test_walkthrough_writes_six_figures_and_accessible_html(tmp_path: Path) -> None:
    transition_model = {
        "epsilon_star": 0.05,
        "annual_distances": [
            {"from_year": year - 1, "to_year": year, "distance": 0.04}
            for year in range(2016, 2020)
        ],
    }
    evaluation = {
        "transition_count": 4,
        "learned_policies": [
            _policy("classical", 0.0, 0),
            _policy("wasserstein_robust", 0.05, 2),
        ],
    }

    result = write_walkthrough(
        site_dir=tmp_path,
        rows=_rows(),
        transition_model=transition_model,
        evaluation=evaluation,
    )

    assert result["figure_count"] == "6"
    assert len(list((tmp_path / "assets").glob("*.svg"))) == 6
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert 'id="results"' in html
    assert "not a clinical recommendation" in html
