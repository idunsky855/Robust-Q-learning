"""Shared fixtures for the test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def sample_raw_csv(tmp_path: Path) -> Path:
    path = tmp_path / "sample.csv"
    path.write_text(
        "\n".join(
            [
                "country,year,organism,antibiotic,resistance_percentage,tested_count",
                "Aland,2015,Escherichia coli,Cefotaxime,10,100",
                "Aland,2015,Escherichia coli,Ciprofloxacin,20,100",
                "Aland,2015,Escherichia coli,Meropenem,0,100",
                "Aland,2016,Escherichia coli,Cefotaxime,30,110",
                "Aland,2016,Escherichia coli,Ciprofloxacin,40,110",
                "Aland,2016,Escherichia coli,Meropenem,1,110",
                "Aland,2017,Escherichia coli,Cefotaxime,35,115",
                "Aland,2017,Escherichia coli,Ciprofloxacin,45,115",
                "Aland,2017,Escherichia coli,Meropenem,1,115",
                "Aland,2018,Escherichia coli,Cefotaxime,40,120",
                "Aland,2018,Escherichia coli,Ciprofloxacin,50,120",
                "Aland,2018,Escherichia coli,Meropenem,2,120",
                "Aland,2019,Escherichia coli,Cefotaxime,45,125",
                "Aland,2019,Escherichia coli,Ciprofloxacin,55,125",
                "Aland,2019,Escherichia coli,Meropenem,2,125",
                "Aland,2020,Escherichia coli,Cefotaxime,50,130",
                "Aland,2020,Escherichia coli,Ciprofloxacin,60,130",
                "Aland,2020,Escherichia coli,Meropenem,2,130",
                "Aland,2021,Escherichia coli,Cefotaxime,52,131",
                "Aland,2021,Escherichia coli,Ciprofloxacin,61,131",
                "Aland,2021,Escherichia coli,Meropenem,2,131",
                "Aland,2022,Escherichia coli,Cefotaxime,54,132",
                "Aland,2022,Escherichia coli,Ciprofloxacin,62,132",
                "Aland,2022,Escherichia coli,Meropenem,2,132",
                "Aland,2023,Escherichia coli,Cefotaxime,55,133",
                "Aland,2023,Escherichia coli,Ciprofloxacin,63,133",
                "Aland,2023,Escherichia coli,Meropenem,2,133",
                "Aland,2024,Escherichia coli,Cefotaxime,56,134",
                "Aland,2024,Escherichia coli,Ciprofloxacin,64,134",
                "Aland,2024,Escherichia coli,Meropenem,2,134",
                "Borduria,2015,Escherichia coli,Cefotaxime,12,100",
                "Borduria,2015,Escherichia coli,Ciprofloxacin,22,100",
                "Borduria,2015,Escherichia coli,Meropenem,0,100",
                "Borduria,2016,Escherichia coli,Cefotaxime,14,101",
                "Borduria,2016,Escherichia coli,Ciprofloxacin,24,101",
                "Borduria,2016,Escherichia coli,Meropenem,0,101",
                "Borduria,2017,Escherichia coli,Cefotaxime,16,102",
                "Borduria,2017,Escherichia coli,Ciprofloxacin,26,102",
                "Borduria,2017,Escherichia coli,Meropenem,0,102",
                "Borduria,2018,Escherichia coli,Cefotaxime,18,103",
                "Borduria,2018,Escherichia coli,Ciprofloxacin,28,103",
                "Borduria,2018,Escherichia coli,Meropenem,0,103",
                "Borduria,2019,Escherichia coli,Cefotaxime,20,104",
                "Borduria,2019,Escherichia coli,Ciprofloxacin,30,104",
                "Borduria,2019,Escherichia coli,Meropenem,0,104",
                "Borduria,2020,Escherichia coli,Cefotaxime,21,105",
                "Borduria,2020,Escherichia coli,Ciprofloxacin,31,105",
                "Borduria,2020,Escherichia coli,Meropenem,0,105",
                "Borduria,2021,Escherichia coli,Cefotaxime,22,106",
                "Borduria,2021,Escherichia coli,Ciprofloxacin,32,106",
                "Borduria,2021,Escherichia coli,Meropenem,0,106",
                "Borduria,2022,Escherichia coli,Cefotaxime,23,107",
                "Borduria,2022,Escherichia coli,Ciprofloxacin,33,107",
                "Borduria,2022,Escherichia coli,Meropenem,0,107",
                "Borduria,2023,Escherichia coli,Cefotaxime,24,108",
                "Borduria,2023,Escherichia coli,Ciprofloxacin,34,108",
                "Borduria,2023,Escherichia coli,Meropenem,0,108",
                "Borduria,2024,Escherichia coli,Cefotaxime,25,109",
                "Borduria,2024,Escherichia coli,Ciprofloxacin,35,109",
                "Borduria,2024,Escherichia coli,Meropenem,0,109",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def sample_metadata_json(tmp_path: Path) -> Path:
    path = tmp_path / "sample.metadata.json"
    path.write_text(
        json.dumps(
            {
                "source_url": "https://example.test",
                "retrieval_date": "2026-06-27",
                "selected_filters": {
                    "pathogen": "Escherichia coli",
                    "years": "2015-2024",
                },
            }
        ),
        encoding="utf-8",
    )
    return path
