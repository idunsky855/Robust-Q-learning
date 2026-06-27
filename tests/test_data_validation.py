from __future__ import annotations

from pathlib import Path

import pytest

from ears_q_learning.data import validate_raw_snapshot


def test_validate_raw_snapshot_accepts_expected_schema(sample_raw_csv: Path) -> None:
    records = validate_raw_snapshot(sample_raw_csv, "Escherichia coli", 2015, 2024)
    assert len(records) == 60
    assert {record.action_code for record in records} == {"3gc", "fq", "carb"}


def test_validate_raw_snapshot_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.csv"
    path.write_text(
        "\n".join(
            [
                "country,year,organism,antibiotic,resistance_percentage,tested_count",
                "Aland,2015,Escherichia coli,Cefotaxime,10,100",
                "Aland,2015,Escherichia coli,Cefotaxime,11,100",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_raw_snapshot(path, "Escherichia coli", 2015, 2024)
