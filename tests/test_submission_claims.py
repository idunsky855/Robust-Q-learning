from __future__ import annotations

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
    documents = (
        _docx_text(
            ROOT / "output/documents/EARS-Net_Q-Learning_Abstract.docx"
        ),
        (ROOT / "docs/final_results.md").read_text(encoding="utf-8"),
    )
    for text in documents:
        assert "90.13%" in text
        assert "89.76%" in text
        assert "64.55%" in text
        assert "100%" in text or "100.00%" in text
