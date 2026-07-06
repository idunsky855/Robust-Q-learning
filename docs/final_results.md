# Final Results

## Reporting Rule

The tables report held-out 2020-2023 decisions evaluated against 2021-2024
outcomes. Policy-level outcomes use the exact Bellman policy, while convergence
diagnostics summarize 30 seeded Q-learning runs. Policies list actions for
states 0 through 7: `0` denotes third-generation cephalosporins, `1` denotes
fluoroquinolones, and `2` denotes carbapenems.

## Primary Analysis

The primary endpoint was mean adjusted coverage, defined as next-year
susceptibility minus `0.10` when the selected action was a carbapenem. The
calibrated radius was epsilon-star = `0.047619`.

| Algorithm | Radius | Exact policy | Adjusted coverage | Raw susceptibility | Carbapenem use | Regret to oracle | Mean policy agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical Q-learning | 0 | `20222222` | 90.13% | 96.58% | 64.55% | 0.35 pp | 90.00% |
| Robust Q-learning | 0.5 epsilon-star | `22222222` | 89.76% | 99.76% | 100.00% | 0.72 pp | 96.67% |
| Robust Q-learning | 1.0 epsilon-star | `22222222` | 89.76% | 99.76% | 100.00% | 0.72 pp | 96.67% |
| Robust Q-learning | 1.5 epsilon-star | `22222222` | 89.76% | 99.76% | 100.00% | 0.72 pp | 97.08% |
| Robust Q-learning | 2.0 epsilon-star | `22222222` | 89.76% | 99.76% | 100.00% | 0.72 pp | 96.25% |

At the calibrated radius, robust Q-learning did not improve the primary
endpoint. Relative to classical Q-learning, adjusted coverage was 0.36
percentage points lower, raw susceptibility was 3.18 points higher, and
carbapenem use was 35.45 points higher. The robust policy therefore exchanged
greater predicted coverage for substantially broader antibiotic use.

## Prespecified Secondary Analyses

The table below compares classical Q-learning with robust Q-learning at each
analysis-specific calibrated radius. Reward values are not comparable across
rows with different reward definitions.

| Analysis | Algorithm | Exact policy | Scalarized reward | Raw susceptibility | Carbapenem use |
|---|---|---:|---:|---:|---:|
| Tested-isolate weighting | Classical | `22222222` | 89.83% | 99.83% | 100.00% |
| Tested-isolate weighting | Robust | `22222222` | 89.83% | 99.83% | 100.00% |
| Breadth and pressure (`beta=0.15`, `delta=0.10`) | Classical | `20222222` | 79.92% | 96.58% | 64.55% |
| Breadth and pressure (`beta=0.15`, `delta=0.10`) | Robust | `22222222` | 79.28% | 99.76% | 100.00% |
| Breadth and pressure (`beta=0.15`, `delta=0.15`) | Classical | `00222222` | 77.64% | 95.41% | 51.82% |
| Breadth and pressure (`beta=0.15`, `delta=0.15`) | Robust | `20222222` | 77.50% | 96.58% | 64.55% |
| Breadth, cost, and pressure (`beta=0.15`, `gamma=0.025`, `delta=0.10`) | Classical | `00022222` | 78.12% | 94.96% | 49.09% |
| Breadth, cost, and pressure (`beta=0.15`, `gamma=0.025`, `delta=0.10`) | Robust | `20222222` | 77.94% | 96.58% | 64.55% |

The cost-adjusted reward produced the clearest reduction in carbapenem use for
the classical policy. Robust optimization partially reversed that reduction at
the calibrated radius. Fluoroquinolones were not selected by any reported exact
policy.

## Interpretation Limits

None of the final training groups satisfied the strict criterion that every
seed reproduce the exact Bellman policy. The results therefore support
policy-level comparisons with explicit seed variability, not a claim of full
numerical convergence. The reward penalties are normative design choices, and
the action-independent transition model does not estimate causal effects of
antibiotic selection on future resistance. These country-level surveillance
results are not patient-specific prescribing guidance.

The complete machine-readable table is regenerated as
`data/processed/final_results_table.csv` by the project pipeline.
