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
|-- src/
`-- tests/
```

## Running The Project

### Prerequisites

- Python 3.11 or newer.
- Approximately 250 MB of free space for the environment and generated files.
- A CPU-based run is sufficient; the eight-state tabular model does not require
  a GPU.

The repository includes the three unchanged EARS-Net CSV exports, their
checksum-verified metadata sidecars, and the external cost inputs required by
both configurations. Run every command below from the repository root.

### Option A: Run With uv

[`uv`](https://docs.astral.sh/uv/) creates and manages the environment
automatically. No activation or separate installation step is required.

Run the deterministic lecturer demonstration:

```bash
uv run --python 3.11 python -m ears_q_learning run --config configs/demo.yaml
```

Run the complete reported experiment:

```bash
uv run --python 3.11 python -m ears_q_learning run --config configs/primary.yaml
```

Run the test suite:

```bash
uv run --python 3.11 --with "pytest>=8,<9" python -m pytest -q
```

### Option B: Run With Standard Python

Create an isolated environment using only Python's standard `venv` module.

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

On macOS or Linux:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

Then use the environment's Python executable for the required command.

| Task | Windows PowerShell | macOS or Linux |
|---|---|---|
| Lecturer demo | `.venv\Scripts\python -m ears_q_learning run --config configs/demo.yaml` | `.venv/bin/python -m ears_q_learning run --config configs/demo.yaml` |
| Full experiment | `.venv\Scripts\python -m ears_q_learning run --config configs/primary.yaml` | `.venv/bin/python -m ears_q_learning run --config configs/primary.yaml` |
| Tests | `.venv\Scripts\python -m pytest -q` | `.venv/bin/python -m pytest -q` |

### Demo And Full Run

`configs/demo.yaml` is the recommended classroom command. It fixes the selected
discount factor (`0.3`), exploration rate (`0.2`), and seed (`201`) while
retaining 50,000 updates and all four Wasserstein radii. It took approximately
two minutes on the development machine. Its outputs are isolated under:

- `data/processed/demo/`
- `results/demo/`
- `data/processed/demo/site/`

`configs/primary.yaml` reproduces the reported analysis, including rolling-origin
hyperparameter selection, 30 final seeds, tested-isolate weighting, stewardship
scenarios, and the cost-adjusted scenario. The verified development-machine
runtime was 2 hours 37 minutes. Its principal outputs are:

- `data/processed/training_summary.json`
- `data/processed/evaluation_metrics.json`
- `data/processed/final_results_table.csv`
- `data/processed/economic_full_training.json`
- `data/processed/stewardship_full_training.json`
- `results/<UTC timestamp>/status.json`
- locally generated walkthrough files under `site/`

The demo is an execution walkthrough, not the source of the report's 30-seed
estimates or convergence conclusions.

## Included Data

The pipeline validates the supplied raw files before preprocessing. Each
metadata sidecar records the ECDC Atlas source URL, retrieval date, selected
filters, and SHA-256 checksum. To replace an export, preserve the configured
filename and update its metadata checksum; otherwise validation will fail.

The expected paths are listed in `configs/primary.yaml` and
`configs/demo.yaml`. Derived data under `data/processed/` and timestamped runs
under `results/` are intentionally excluded from Git because the commands above
regenerate them.

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
