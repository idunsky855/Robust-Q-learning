"""Command-line interface for the EARS-Net Q-learning project."""

from __future__ import annotations

import argparse
from pathlib import Path

from ears_q_learning.config import load_config
from ears_q_learning.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="ears_q_learning",
        description=(
            "Reproducible EARS-Net Q-learning pipeline for robust antibiotic "
            "selection under resistance drift."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the configured project pipeline.",
    )
    run_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a YAML configuration file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested CLI command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        config = load_config(args.config)
        run_pipeline(config)
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2
