"""Main project pipeline."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ears_q_learning.config import Config
from ears_q_learning.data import (
    build_snapshot_validation_report,
    validate_atlas_snapshots,
    validate_raw_snapshot,
    validate_snapshot_metadata_collection,
    validate_snapshot_metadata,
)
from ears_q_learning.evaluation import run_policy_evaluation
from ears_q_learning.mdp import (
    annual_state_distributions,
    calibrate_wasserstein_radius,
    estimate_reward_bands,
    myopic_policy,
    normalized_hamming_cost,
    transition_kernel,
)
from ears_q_learning.model_selection import run_model_selection
from ears_q_learning.preprocessing import (
    build_country_year_panel,
    build_preprocessing_report,
    build_state_assignment_rows,
    build_transition_records,
    eligible_countries,
    filter_rows_by_country,
    split_rows_by_period,
)
from ears_q_learning.reproducibility import build_run_metadata, ensure_directory, set_global_seed, write_json
from ears_q_learning.state_space import encode_state, fit_thresholds


def _run_directory(results_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ensure_directory(results_dir / timestamp)


def run_pipeline(config: Config) -> dict[str, object]:
    """Run the first reproducible project slice.

    This scaffold validates the raw snapshot, derives the analysis-ready state
    space, estimates the action-independent kernel, and writes deterministic
    metadata. Full tuning and evaluation are added in later slices.
    """
    set_global_seed(config.project.random_seed)
    ensure_directory(config.paths.processed_dir)
    run_dir = _run_directory(config.paths.results_dir)
    metadata = build_run_metadata(config, run_dir)
    write_json(run_dir / "run_metadata.json", metadata)

    if config.paths.raw_snapshots:
        missing_snapshots = [
            str(snapshot_path)
            for snapshot_path, _metadata_path in config.paths.raw_snapshots
            if not snapshot_path.exists()
        ]
        if missing_snapshots:
            placeholder = {
                "status": "blocked_missing_raw_snapshot",
                "message": (
                    "One or more raw ECDC Atlas exports are missing. Place the "
                    "unchanged CSV files at the configured paths and rerun the "
                    "pipeline."
                ),
                "expected_paths": missing_snapshots,
            }
            write_json(run_dir / "status.json", placeholder)
            return placeholder

        missing_metadata = [
            str(metadata_path)
            for _snapshot_path, metadata_path in config.paths.raw_snapshots
            if not metadata_path.exists()
        ]
        if missing_metadata:
            placeholder = {
                "status": "blocked_missing_raw_snapshot_metadata",
                "message": (
                    "One or more raw ECDC Atlas metadata sidecars are missing. "
                    "Add the JSON provenance files and rerun the pipeline."
                ),
                "expected_metadata_paths": missing_metadata,
            }
            write_json(run_dir / "status.json", placeholder)
            return placeholder

        metadata_payload: dict[str, object] = {
            "source_format": "ecdc_atlas_long",
            "snapshots": validate_snapshot_metadata_collection(
                config.paths.raw_snapshots
            ),
        }
        records = validate_atlas_snapshots(
            paths=[
                snapshot_path
                for snapshot_path, _metadata_path in config.paths.raw_snapshots
            ],
            organism=config.data.organism,
            year_start=config.data.training_year_start,
            year_end=config.data.evaluation_year_end,
        )
    elif config.paths.raw_snapshot is None:
        placeholder = {
            "status": "blocked_missing_raw_snapshot",
            "message": (
                "No raw EARS-Net snapshot path is configured. Provide either "
                "'raw_snapshot' or 'raw_snapshots' in the paths section."
            ),
        }
        write_json(run_dir / "status.json", placeholder)
        return placeholder
    elif not config.paths.raw_snapshot.exists():
        placeholder = {
            "status": "blocked_missing_raw_snapshot",
            "message": (
                "The raw EARS-Net snapshot is not present yet. Place the exported CSV "
                "at the configured path and rerun the pipeline."
            ),
            "expected_path": str(config.paths.raw_snapshot),
            "expected_metadata_path": str(config.paths.raw_snapshot_metadata),
        }
        write_json(run_dir / "status.json", placeholder)
        return placeholder
    elif config.paths.raw_snapshot_metadata is None:
        placeholder = {
            "status": "blocked_missing_raw_snapshot_metadata",
            "message": (
                "The raw EARS-Net snapshot path is configured, but the required "
                "metadata sidecar path is not configured."
            ),
            "snapshot_path": str(config.paths.raw_snapshot),
        }
        write_json(run_dir / "status.json", placeholder)
        return placeholder
    elif not config.paths.raw_snapshot_metadata.exists():
        placeholder = {
            "status": "blocked_missing_raw_snapshot_metadata",
            "message": (
                "The raw EARS-Net snapshot exists, but the required metadata sidecar "
                "is missing. Add the JSON provenance file and rerun the pipeline."
            ),
            "snapshot_path": str(config.paths.raw_snapshot),
            "expected_metadata_path": str(config.paths.raw_snapshot_metadata),
        }
        write_json(run_dir / "status.json", placeholder)
        return placeholder
    else:
        metadata_payload = validate_snapshot_metadata(
            snapshot_path=config.paths.raw_snapshot,
            metadata_path=config.paths.raw_snapshot_metadata,
        )
        records = validate_raw_snapshot(
            path=config.paths.raw_snapshot,
            organism=config.data.organism,
            year_start=config.data.training_year_start,
            year_end=config.data.evaluation_year_end,
        )
    snapshot_report = build_snapshot_validation_report(
        records=records,
        metadata=metadata_payload,
        year_start=config.data.training_year_start,
        year_end=config.data.evaluation_year_end,
    )
    write_json(config.paths.processed_dir / "raw_snapshot_report.json", snapshot_report)
    rows = build_country_year_panel(records)
    countries = eligible_countries(
        rows=rows,
        training_year_end=config.data.training_year_end,
        evaluation_year_start=config.data.evaluation_year_start,
        minimum_training_transitions=config.data.minimum_training_transitions,
        minimum_evaluation_transitions=config.data.minimum_evaluation_transitions,
    )
    preprocessing_report = build_preprocessing_report(
        rows=rows,
        eligible=countries,
        training_year_end=config.data.training_year_end,
        evaluation_year_start=config.data.evaluation_year_start,
        minimum_training_transitions=config.data.minimum_training_transitions,
        minimum_evaluation_transitions=config.data.minimum_evaluation_transitions,
    )
    write_json(
        config.paths.processed_dir / "preprocessing_report.json",
        preprocessing_report,
    )
    filtered_rows = filter_rows_by_country(rows, countries)
    training_rows, evaluation_rows = split_rows_by_period(
        filtered_rows,
        training_year_end=config.data.training_year_end,
        evaluation_year_end=config.data.evaluation_year_end,
    )
    thresholds = fit_thresholds(training_rows)
    state_lookup = {
        (row.country, row.year): encode_state(row, thresholds) for row in filtered_rows
    }
    write_json(
        config.paths.processed_dir / "state_assignments.json",
        {
            "thresholds": asdict(thresholds),
            "rows": build_state_assignment_rows(filtered_rows, state_lookup),
        },
    )
    transitions = build_transition_records(
        rows=training_rows,
        state_lookup=state_lookup,
        weighting=config.data.weighting,
    )
    kernel = transition_kernel(transitions, config.data.smoothing_gamma)
    reward_bands = estimate_reward_bands(
        training_rows=training_rows,
        state_lookup=state_lookup,
        carbapenem_penalty=config.data.carbapenem_penalty,
    )
    state_distributions = annual_state_distributions(training_rows, state_lookup)
    cost_matrix = normalized_hamming_cost()
    drift_calibration = calibrate_wasserstein_radius(
        state_distributions,
        cost_matrix,
    )
    epsilon_star = float(drift_calibration["epsilon_star"])
    transition_model = {
        "action_independent": True,
        "smoothing_gamma": config.data.smoothing_gamma,
        "reference_kernel": kernel.tolist(),
        "annual_state_distributions": {
            str(year): distribution.tolist()
            for year, distribution in state_distributions.items()
        },
        **drift_calibration,
        "robustness_radii": [
            multiplier * epsilon_star
            for multiplier in config.learning.epsilon_multipliers
        ],
    }
    write_json(
        config.paths.processed_dir / "transition_model.json",
        transition_model,
    )
    learned_myopic_policy = myopic_policy(kernel, reward_bands)
    summary = {
        "status": "scaffold_completed",
        "validated_record_count": len(records),
        "country_year_count": len(filtered_rows),
        "eligible_country_count": len(countries),
        "training_country_year_count": len(training_rows),
        "evaluation_country_year_count": len(evaluation_rows),
        "thresholds": asdict(thresholds),
        "reward_bands": reward_bands,
        "myopic_policy": learned_myopic_policy.tolist(),
        "training_years": sorted({row.year for row in training_rows}),
        "transition_model": transition_model,
        "cost_matrix": cost_matrix.tolist(),
    }
    training_summary = run_model_selection(
        rows=filtered_rows,
        training_year_end=config.data.training_year_end,
        learning=config.learning,
        smoothing_gamma=config.data.smoothing_gamma,
        carbapenem_penalty=config.data.carbapenem_penalty,
        weighting=config.data.weighting,
        epsilon_star=epsilon_star,
    )
    write_json(
        config.paths.processed_dir / "training_summary.json",
        training_summary,
    )
    evaluation_metrics = run_policy_evaluation(
        rows=filtered_rows,
        state_lookup=state_lookup,
        training_summary=training_summary,
        myopic_policy=learned_myopic_policy,
        carbapenem_penalty=config.data.carbapenem_penalty,
        decision_year_start=config.data.evaluation_year_start,
        outcome_year_end=config.data.evaluation_year_end,
    )
    write_json(
        config.paths.processed_dir / "evaluation_metrics.json",
        evaluation_metrics,
    )
    summary["training_summary_path"] = str(
        config.paths.processed_dir / "training_summary.json"
    )
    summary["evaluation_metrics_path"] = str(
        config.paths.processed_dir / "evaluation_metrics.json"
    )
    write_json(config.paths.processed_dir / "scaffold_summary.json", summary)
    write_json(run_dir / "status.json", summary)
    return summary
