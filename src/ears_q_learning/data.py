"""Raw EARS-Net snapshot intake and validation."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from ears_q_learning.constants import ACTIONS
from ears_q_learning.reproducibility import sha256_file, write_json
from ears_q_learning.types import RawRecord

CANONICAL_COLUMNS = {
    "country": ("country", "reporting country", "geo"),
    "year": ("year", "period", "time"),
    "organism": ("organism", "pathogen", "bacteria"),
    "antibiotic": ("antibiotic", "antibiotic class", "antimicrobial"),
    "resistance_percentage": (
        "resistance_percentage",
        "resistance percentage",
        "resistant (%)",
        "resistance (%)",
        "percent resistant",
    ),
    "tested_count": (
        "tested_count",
        "tested count",
        "number tested",
        "isolates tested",
        "n",
    ),
}

REQUIRED_METADATA_FIELDS = (
    "source_url",
    "retrieval_date",
    "selected_filters",
)


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().replace("_", " ").split())


def _resolve_columns(fieldnames: Iterable[str]) -> dict[str, str]:
    if fieldnames is None:
        raise ValueError("The raw CSV is missing a header row.")
    normalized = {_normalize(name): name for name in fieldnames}
    resolved: dict[str, str] = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        for alias in aliases:
            if _normalize(alias) in normalized:
                resolved[canonical] = normalized[_normalize(alias)]
                break
        else:
            raise ValueError(f"Missing required column for '{canonical}'.")
    return resolved


def canonical_action(antibiotic: str) -> str:
    """Map one antibiotic label to the project's canonical action code."""
    normalized = _normalize(antibiotic)
    for action in ACTIONS:
        if normalized in {_normalize(alias) for alias in action.aliases}:
            return action.code
    raise ValueError(f"Unsupported antibiotic mapping: {antibiotic}")


def validate_raw_snapshot(
    path: Path,
    organism: str,
    year_start: int,
    year_end: int,
) -> list[RawRecord]:
    """Validate and load the raw EARS-Net snapshot."""
    records: list[RawRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = _resolve_columns(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            row_organism = row[columns["organism"]].strip()
            if row_organism != organism:
                continue
            year = int(row[columns["year"]])
            if year < year_start or year > year_end:
                raise ValueError(
                    f"Row {row_number} has year {year}, outside {year_start}-{year_end}."
                )
            resistance_percentage = float(row[columns["resistance_percentage"]])
            if resistance_percentage < 0 or resistance_percentage > 100:
                raise ValueError(
                    f"Row {row_number} has invalid resistance percentage "
                    f"{resistance_percentage}."
                )
            tested_count = int(float(row[columns["tested_count"]]))
            if tested_count <= 0:
                raise ValueError(
                    f"Row {row_number} has invalid tested count {tested_count}."
                )
            records.append(
                RawRecord(
                    country=row[columns["country"]].strip(),
                    year=year,
                    organism=row_organism,
                    action_code=canonical_action(row[columns["antibiotic"]]),
                    resistance_percentage=resistance_percentage,
                    tested_count=tested_count,
                )
            )
    duplicate_counts = Counter(
        (record.country, record.year, record.action_code) for record in records
    )
    duplicates = [key for key, count in duplicate_counts.items() if count > 1]
    if duplicates:
        raise ValueError(
            "The raw snapshot contains duplicate country-year-action rows. "
            f"Examples: {duplicates[:3]}"
        )
    return records


def load_snapshot_metadata(path: Path) -> dict[str, object]:
    """Load raw-snapshot metadata from a JSON sidecar."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Snapshot metadata must be a JSON object.")
    return payload


def validate_snapshot_metadata(
    snapshot_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    """Validate raw-snapshot provenance metadata."""
    if not metadata_path.exists():
        raise FileNotFoundError(
            "The raw snapshot metadata file is missing. "
            f"Expected: {metadata_path}"
        )
    payload = load_snapshot_metadata(metadata_path)
    for field in REQUIRED_METADATA_FIELDS:
        if field not in payload:
            raise ValueError(f"Snapshot metadata is missing '{field}'.")
    if not isinstance(payload["selected_filters"], dict):
        raise ValueError("Snapshot metadata field 'selected_filters' must be an object.")
    checksum = sha256_file(snapshot_path)
    recorded_checksum = payload.get("sha256")
    if recorded_checksum is not None and recorded_checksum != checksum:
        raise ValueError(
            "Snapshot metadata checksum does not match the current raw file. "
            f"Recorded: {recorded_checksum}; current: {checksum}."
        )
    validated = dict(payload)
    validated["sha256"] = checksum
    validated["snapshot_path"] = str(snapshot_path)
    return validated


def build_snapshot_validation_report(
    records: list[RawRecord],
    metadata: dict[str, object],
    year_start: int,
    year_end: int,
) -> dict[str, object]:
    """Build a machine-readable report for the raw snapshot intake stage."""
    years = sorted({record.year for record in records})
    countries = sorted({record.country for record in records})
    action_counts = Counter(record.action_code for record in records)
    return {
        "status": "raw_snapshot_validated",
        "record_count": len(records),
        "country_count": len(countries),
        "countries": countries,
        "year_range_requested": [year_start, year_end],
        "years_present": years,
        "action_counts": dict(action_counts),
        "metadata": metadata,
    }


def write_snapshot_provenance(
    snapshot_path: Path,
    metadata_path: Path,
    source_url: str,
    retrieval_date: str,
    selected_filters: dict[str, str],
) -> None:
    """Write provenance metadata for the raw snapshot."""
    payload = {
        "source_url": source_url,
        "retrieval_date": retrieval_date,
        "selected_filters": selected_filters,
        "sha256": sha256_file(snapshot_path),
        "snapshot_path": str(snapshot_path),
    }
    write_json(metadata_path, payload)
