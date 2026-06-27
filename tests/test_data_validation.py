from __future__ import annotations

from pathlib import Path

import pytest

from ears_q_learning.data import validate_raw_snapshot, validate_snapshot_metadata


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


def test_validate_raw_snapshot_rejects_invalid_tested_count(tmp_path: Path) -> None:
    path = tmp_path / "bad_count.csv"
    path.write_text(
        "\n".join(
            [
                "country,year,organism,antibiotic,resistance_percentage,tested_count",
                "Aland,2015,Escherichia coli,Cefotaxime,10,0",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tested count"):
        validate_raw_snapshot(path, "Escherichia coli", 2015, 2024)


def test_validate_raw_snapshot_rejects_invalid_antibiotic_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad_antibiotic.csv"
    path.write_text(
        "\n".join(
            [
                "country,year,organism,antibiotic,resistance_percentage,tested_count",
                "Aland,2015,Escherichia coli,UnknownDrug,10,100",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported antibiotic mapping"):
        validate_raw_snapshot(path, "Escherichia coli", 2015, 2024)


def test_validate_snapshot_metadata_computes_checksum(
    sample_raw_csv: Path,
    sample_metadata_json: Path,
) -> None:
    metadata = validate_snapshot_metadata(sample_raw_csv, sample_metadata_json)
    assert metadata["source_url"] == "https://example.test"
    assert metadata["sha256"]


def test_validate_snapshot_metadata_rejects_checksum_mismatch(
    sample_raw_csv: Path,
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(
        """
        {
          "source_url": "https://example.test",
          "retrieval_date": "2026-06-27",
          "selected_filters": {"pathogen": "Escherichia coli"},
          "sha256": "wrong"
        }
        """.strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="checksum"):
        validate_snapshot_metadata(sample_raw_csv, path)
