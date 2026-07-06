# EARS-Net Q-Learning: Robust Antibiotic Selection Under Resistance Drift

## Overview

This repository contains a reproducible academic project on distributionally robust reinforcement learning for population-informed antibiotic selection under resistance drift. The empirical setting uses public EARS-Net surveillance data for invasive *Escherichia coli* isolates and studies whether Wasserstein-robust Q-learning maintains stronger one-year-ahead coverage than classical Q-learning when national resistance patterns shift over time.

The project is not a patient-level prescribing system, not a causal stewardship model, and not a source of clinical guidance. It is a transparent country-year Markov decision process built from public surveillance data with documented limitations.

## Research Question

Does a Wasserstein-robust Q-learning policy maintain better one-year-ahead adjusted coverage than classical Q-learning when European *E. coli* resistance profiles drift between the training and evaluation periods?

## Current Status

The primary experiment, prespecified sensitivity analyses, final abstract,
figures, and static research walkthrough are complete. The final primary run
uses rolling-origin tuning, 30 fixed training seeds, four Wasserstein radii,
exact Bellman references, and held-out 2021-2024 outcomes.

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
|-- site/
|-- src/
`-- tests/
```

## Running The Project

Install the package in a Python 3.11 or newer environment, then run:

```bash
python -m ears_q_learning run --config configs/primary.yaml
```

The primary configuration performs the complete tuning and multi-seed analysis
used for reported results. It is computationally expensive. For a deterministic
lecturer demonstration using the selected parameters and one fixed seed, run:

```bash
python -m ears_q_learning run --config configs/demo.yaml
```

The demonstration retains 50,000 Q-learning updates and all four Wasserstein
radii, but fixes the discount factor to `0.3`, exploration rate to `0.2`, and
seed to `201`. It writes separate artifacts under `data/processed/demo/` and
`results/demo/`. Demonstration outputs illustrate reproducibility; they are not
the source of the reported 30-seed estimates or convergence conclusions. A
complete demonstration run took approximately 2 minutes 30 seconds on the
development machine; runtime depends on the lecturer's processor.

The completed static walkthrough is available at `site/index.html`. It is
dependency-free and may be opened directly in a browser or served locally with:

```bash
python -m http.server 8000
```

Then open `http://127.0.0.1:8000/site/`.

If the raw EARS-Net CSV exports have not yet been placed in `data/raw/`, the command writes a machine-readable blocked status instead of failing silently. Once the snapshots exist, the scaffold validates the raw files and writes summary artifacts for the first modeling slice.

## Raw Snapshot Workflow

1. In the ECDC Surveillance Atlas, select antimicrobial resistance, *Escherichia coli*, one antibiotic class, and the `R - resistant isolates, percentage` indicator.
2. Use the export dialog with all time periods, all regions, all indicators in the current table, and CSV file format.
3. Repeat the export for carbapenems, fluoroquinolones, and third-generation cephalosporins.
4. Place the unchanged CSV exports in `data/raw/` using the paths configured in `configs/primary.yaml`.
5. Create one JSON provenance sidecar beside each CSV with the source URL, retrieval date, selected filters, and checksum.
6. Run `python -m ears_q_learning run --config configs/primary.yaml`.

The pipeline requires every configured raw CSV and metadata sidecar. It consolidates the long-format Atlas exports into the canonical country-year-antibiotic schema in code, leaving the downloaded files unchanged. When the raw inputs are present, it writes a machine-readable `raw_snapshot_report.json` intake report before any modeling logic proceeds.

## Penalty Sensitivity

The primary analysis retains the prespecified carbapenem penalty of `0.10`. The
configuration also defines a sensitivity grid of `0.10`, `0.15`, `0.20`, `0.25`,
and `0.30`. For this analysis, state thresholds, the transition kernel, selected
discount factors, and Wasserstein radii remain fixed. Exact Bellman solutions are
used so that incomplete stochastic convergence does not obscure action-switch
points.

The pipeline writes `penalty_sensitivity.json`, `penalty_sensitivity.csv`,
`penalty_vs_carbapenem_use.svg`, and
`susceptibility_vs_carbapenem_use.svg` to `data/processed/`. Values other than
`0.10` are sensitivity analyses and are not candidates for retrospective primary
penalty selection.

The tested-isolate weighting sensitivity independently re-estimates rewards,
transition probabilities, annual drift, and the Wasserstein radius before
repeating tuning, final training, and evaluation. It writes
`tested_weighting_sensitivity.json` and `tested_weighting_summary.csv`. Equal
country-transition weighting remains the primary analysis.

## Stewardship Reward Scenario

A secondary, non-causal scenario evaluates
`susceptibility - beta*breadth - delta*breadth*next-state severity`. The
prespecified breadth scores are `0.40` for third-generation cephalosporins,
`0.60` for fluoroquinolones, and `1.00` for carbapenems. The pipeline evaluates
the configured beta and delta grids with fixed state thresholds, transition
kernel, discount factors, and Wasserstein radii. Results are written to
`stewardship_reward_scenario.json`, `stewardship_reward_scenario.csv`, and
`stewardship_reward_carbapenem_use.svg`.

The two prespecified moderate and strong scenarios are also rerun through the
complete rolling-origin tuning and 30-seed training protocol. Their seed-level
results are stored in `stewardship_full_training.json`, with a compact comparison
in `stewardship_full_training_summary.csv`.

The breadth and future-pressure components are normative stewardship assumptions
rather than measured causal effects. A separate economic scenario uses one
representative parenteral product per class and 2024-2025 English NHS hospital
purchase prices from the eMIT national database. Costs are calculated per WHO
defined daily dose and normalized after a `log1p` transformation. These prices
are acquisition-cost proxies, not Europe-wide treatment costs, and exclude
administration, monitoring, length of stay, and adverse-event costs. The scenario
writes `economic_pressure_scenario.json` and `economic_pressure_summary.csv`.
The prespecified `beta=0.15`, `gamma=0.025`, `delta=0.10` scenario additionally
receives independent rolling-origin tuning and complete 30-seed training.
Seed-level diagnostics are stored in `economic_full_training.json`, with a
compact comparison in `economic_full_training_summary.csv`.

## Data Basis

The intended raw input is three ECDC Atlas exports covering *E. coli* resistance percentages and tested-isolate counts for:

- third-generation cephalosporins;
- fluoroquinolones;
- carbapenems.

The project uses country-year data for 2015 through 2024. Export provenance, filters, and checksums must be recorded alongside the raw file.

## Key Documents

- [Final results](docs/final_results.md)
- [Project plan](docs/project_plan.md)
- [Implementation PRD](docs/implementation_prd.md)
- [Reproducibility guidelines](docs/reproducibility.md)

## Reference

Neufeld, A., and Sester, J. (2024). Robust Q-learning algorithm for Markov decision processes under Wasserstein uncertainty. *Automatica*, 168, 111825.
