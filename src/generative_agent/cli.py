"""Unified command-line entry point for supported generative workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from generative_agent.image_data import ImagePreprocessConfig
from generative_agent.image_workflow import run_image_generation_workflow
from generative_agent.state import ExperimentBudget
from generative_agent.tabular_io import ColumnRoles
from generative_agent.workflow import run_tabular_generation_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DeEntropy Project 2 generative agent.")
    parser.add_argument("dataset_path")
    parser.add_argument("--modality", choices=("auto", "tabular", "image"), default="auto")
    parser.add_argument("--objective", default="Generate synthetic data.")
    parser.add_argument("--output-dir", default="outputs/project2_agent_run")
    parser.add_argument("--report-name", default="final_report.md")
    parser.add_argument("--target")
    parser.add_argument("--identifier", action="append", default=[])
    parser.add_argument("--conditioning", action="append", default=[])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-models", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-synthetic", type=int)
    parser.add_argument("--image-latent-dim", type=int, default=16)
    parser.add_argument("--gan-latent-dim", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--repeated-generation-seeds", type=int, default=3)
    parser.add_argument(
        "--ai-analysis",
        choices=("auto", "openai", "anthropic", "disabled"),
        default="auto",
        help="Optional final narrative analysis; auto uses a configured Anthropic or OpenAI key.",
    )
    parser.add_argument("--ai-model", default="gpt-4o-mini")
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    modality = _infer_modality(dataset_path, args.modality)
    budget = ExperimentBudget(max_models=args.max_models, max_epochs=args.epochs, seed=args.seed)
    if modality == "image":
        result = run_image_generation_workflow(
            npz_path=dataset_path,
            objective_text=args.objective,
            output_dir=Path(args.output_dir),
            report_name=args.report_name,
            budget=budget,
            preprocess=ImagePreprocessConfig(
                max_train_samples=args.max_train_samples,
                max_validation_samples=args.max_validation_samples,
                max_test_samples=args.max_test_samples,
                seed=args.seed,
            ),
            latent_dim=args.image_latent_dim,
            gan_latent_dim=args.gan_latent_dim,
            n_generated=args.n_synthetic or 32,
            repeated_generation_seeds=args.repeated_generation_seeds,
            ai_analysis_mode=args.ai_analysis,
            ai_model=args.ai_model,
        )
    else:
        result = run_tabular_generation_workflow(
            csv_path=dataset_path,
            objective_text=args.objective,
            output_dir=Path(args.output_dir),
            report_name=args.report_name,
            roles=ColumnRoles(
                identifiers=tuple(args.identifier),
                target=args.target,
                conditioning=tuple(args.conditioning),
            ),
            budget=budget,
            n_synthetic=args.n_synthetic,
            ai_analysis_mode=args.ai_analysis,
            ai_model=args.ai_model,
        )
    print(f"Report: {result.report_path}")
    if hasattr(result, "synthetic_path"):
        print(f"Synthetic samples: {result.synthetic_path}")
    if hasattr(result, "generated_grid_path"):
        print(f"Generated grid: {result.generated_grid_path}")
    print(f"Checkpoint: {result.checkpoint_path}" if result.checkpoint_path else "Checkpoint: none")


def _infer_modality(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if path.suffix.lower() == ".npz":
        return "image"
    if path.suffix.lower() in {".csv", ".tsv"}:
        return "tabular"
    raise ValueError("Could not infer modality; pass --modality tabular or --modality image.")


if __name__ == "__main__":
    main()
