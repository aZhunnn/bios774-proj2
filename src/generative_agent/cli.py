"""Command-line entry point for the Phase 2 tabular VAE workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from generative_agent.state import ExperimentBudget
from generative_agent.tabular_io import ColumnRoles
from generative_agent.workflow import run_tabular_vae_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 2 tabular VAE workflow.")
    parser.add_argument("csv_path")
    parser.add_argument("--objective", default="Generate synthetic tabular data.")
    parser.add_argument("--output-dir", default="outputs/phase2_tabular_vae")
    parser.add_argument("--target")
    parser.add_argument("--identifier", action="append", default=[])
    parser.add_argument("--conditioning", action="append", default=[])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-synthetic", type=int)
    args = parser.parse_args()
    result = run_tabular_vae_workflow(
        csv_path=Path(args.csv_path),
        objective_text=args.objective,
        output_dir=Path(args.output_dir),
        roles=ColumnRoles(
            identifiers=tuple(args.identifier),
            target=args.target,
            conditioning=tuple(args.conditioning),
        ),
        budget=ExperimentBudget(max_models=1, max_epochs=args.epochs, seed=args.seed),
        n_synthetic=args.n_synthetic,
    )
    print(f"Report: {result.report_path}")
    print(f"Synthetic samples: {result.synthetic_path}")
    print(f"Checkpoint: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
