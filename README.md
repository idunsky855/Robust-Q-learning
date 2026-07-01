# EARS-Net Q-Learning: Robust Antibiotic Selection Under Resistance Drift

## Overview

This repository contains a reproducible academic project on distributionally robust reinforcement learning for population-informed antibiotic selection under resistance drift. The empirical setting uses public EARS-Net surveillance data for invasive *Escherichia coli* isolates and studies whether Wasserstein-robust Q-learning maintains stronger one-year-ahead coverage than classical Q-learning when national resistance patterns shift over time.

The project is not a patient-level prescribing system, not a causal stewardship model, and not a source of clinical guidance. It is a transparent country-year Markov decision process built from public surveillance data with documented limitations.

## Research Question

Does a Wasserstein-robust Q-learning policy maintain better one-year-ahead adjusted coverage than classical Q-learning when European *E. coli* resistance profiles drift between the training and evaluation periods?

## Current Status

The repository now includes the first reproducible implementation slice:

- a Python package with a command-line entrypoint;
- YAML configuration for the primary experiment;
- deterministic run metadata and artifact directory creation;
- raw-snapshot validation and country-year state encoding;
- action-independent transition-kernel estimation and myopic-policy summary;
- an issue tracker and implementation PRD for the approved project scope.

Full rolling-origin tuning, final policy evaluation, figures, and the static walkthrough remain in active implementation.

## Repository Principles

1. Keep the code simple, readable, and well documented.
2. Make every derived artifact reproducible from documented inputs and commands.
3. Use academic language and avoid overstated claims.
4. Preserve the raw EARS-Net snapshot unchanged and script all later stages.

## Project Layout

```text
.
|-- configs/
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|-- results/
|-- src/
`-- tests/
```

## Running The Scaffold

Install the package in a Python 3.11 or newer environment, then run:

```bash
python -m ears_q_learning run --config configs/primary.yaml
```

If the raw EARS-Net CSV exports have not yet been placed in `data/raw/`, the command writes a machine-readable blocked status instead of failing silently. Once the snapshots exist, the scaffold validates the raw files and writes summary artifacts for the first modeling slice.

## Raw Snapshot Workflow

1. In the ECDC Surveillance Atlas, select antimicrobial resistance, *Escherichia coli*, one antibiotic class, and the `R - resistant isolates, percentage` indicator.
2. Use the export dialog with all time periods, all regions, all indicators in the current table, and CSV file format.
3. Repeat the export for carbapenems, fluoroquinolones, and third-generation cephalosporins.
4. Place the unchanged CSV exports in `data/raw/` using the paths configured in `configs/primary.yaml`.
5. Create one JSON provenance sidecar beside each CSV with the source URL, retrieval date, selected filters, and checksum.
6. Run `python -m ears_q_learning run --config configs/primary.yaml`.

The pipeline requires every configured raw CSV and metadata sidecar. It consolidates the long-format Atlas exports into the canonical country-year-antibiotic schema in code, leaving the downloaded files unchanged. When the raw inputs are present, it writes a machine-readable `raw_snapshot_report.json` intake report before any modeling logic proceeds.

## Data Basis

The intended raw input is three ECDC Atlas exports covering *E. coli* resistance percentages and tested-isolate counts for:

- third-generation cephalosporins;
- fluoroquinolones;
- carbapenems.

The project uses country-year data for 2015 through 2024. Export provenance, filters, and checksums must be recorded alongside the raw file.

## Key Documents

- [Project plan](docs/project_plan.md)
- [Implementation PRD](docs/implementation_prd.md)
- [Reproducibility guidelines](docs/reproducibility.md)

## Reference

Neufeld, A., and Sester, J. (2024). Robust Q-learning algorithm for Markov decision processes under Wasserstein uncertainty. *Automatica*, 168, 111825.
