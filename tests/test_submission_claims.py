from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def _docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    return " ".join(
        element.text or "" for element in root.findall(".//w:t", namespace)
    )


def test_final_deliverables_match_primary_metrics() -> None:
    evaluation = json.loads(
        (ROOT / "data/processed/evaluation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    classical = next(
        row for row in evaluation["learned_policies"] if row["algorithm"] == "classical"
    )
    epsilon_star = json.loads(
        (ROOT / "data/processed/transition_model.json").read_text(encoding="utf-8")
    )["epsilon_star"]
    robust = min(
        (
            row
            for row in evaluation["learned_policies"]
            if row["algorithm"] == "wasserstein_robust"
        ),
        key=lambda row: abs(row["robust_epsilon"] - epsilon_star),
    )
    assert round(
        100 * classical["exact_bellman_policy_metrics"]["mean_adjusted_coverage"], 2
    ) == 90.13
    assert round(
        100 * robust["exact_bellman_policy_metrics"]["mean_adjusted_coverage"], 2
    ) == 89.76

    documents = (
        _docx_text(
            ROOT / "output/documents/EARS-Net_Q-Learning_Abstract.docx"
        ),
        (ROOT / "docs/final_results.md").read_text(encoding="utf-8"),
        (ROOT / "site/index.html").read_text(encoding="utf-8"),
    )
    for text in documents:
        assert "90.13%" in text
        assert "89.76%" in text
        assert "64.55%" in text
        assert "100%" in text or "100.00%" in text
