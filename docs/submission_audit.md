# Submission Audit

## Scientific Consistency

- The abstract, final-results document, consolidated CSV, and HTML walkthrough
  report the same exact-policy primary outcomes.
- Classical adjusted coverage is 90.13%, raw susceptibility is 96.58%, and
  carbapenem use is 64.55%.
- Robust adjusted coverage at epsilon-star is 89.76%, raw susceptibility is
  99.76%, and carbapenem use is 100.00%.
- The primary conclusion is negative: robustness did not improve adjusted
  coverage and increased carbapenem use.
- All reported training groups are described as not satisfying the strict
  all-seed exact-policy convergence criterion.
- Secondary reward values are not compared numerically across different reward
  definitions.

## Interpretation Boundaries

- The model is described as population-informed rather than patient-specific.
- Resistance transitions are action-independent and are not interpreted
  causally.
- Breadth, cost, and future-pressure coefficients are identified as normative
  design choices.
- The project makes no clinical-guidance, p-value, confidence-interval, or
  causal-effect claims.
- Pandemic-era evaluation years are not attributed solely to COVID-19.

## Reproducibility

- `configs/primary.yaml` defines the complete reported experiment.
- `configs/demo.yaml` defines the deterministic lecturer demonstration and
  writes to isolated output directories.
- Raw EARS-Net files and metadata sidecars are preserved with checksums.
- External acquisition-cost data include source metadata and a checksum.
- The pipeline writes configurations, seeds, policies, metrics, convergence
  diagnostics, figures, and the final consolidated results table.
- The static walkthrough and six SVG figures are generated from pipeline data.
- A fresh Python 3.11 environment passes the complete test suite.

## Deliverables

- `output/documents/EARS-Net_Q-Learning_Abstract.docx`
- `docs/final_results.md`
- `data/processed/final_results_table.csv` after pipeline execution
- `site/index.html` and `site/assets/*.svg`
- `README.md` with primary and lecturer-demonstration commands
