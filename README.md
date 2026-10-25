# Project 2: AI Agent for Generative Modeling

This repository contains an objective-aware generative-modeling agent for continuous tabular and small image datasets. The agent inspects the data, interprets the objective, applies capability and eligibility checks, allocates a global training budget, trains feasible candidates, evaluates them on validation data, evaluates the selected candidate once on held-out data, and writes artifacts plus a factual report.

## Supported models

- Continuous tabular: VAE, RealNVP, GMM, Gaussian copula, and a training-row bootstrap baseline.
- Image: convolutional VAE and lightweight DCGAN-style GAN.

Unsupported objectives such as imputation, conditional generation, and formal privacy guarantees are rejected rather than simulated. Model outputs are not claimed to be private or clinically validated.

## Run the unified CLI

From the repository root:

```bash
PYTHONPATH=src python -m generative_agent.cli DATASET --modality auto --objective "Generate synthetic data for internal testing." --output-dir outputs/run
```

The modality is inferred from `.csv`/`.tsv` and `.npz` inputs, or can be specified explicitly with `--modality tabular` or `--modality image`. See the submission report and the two generated reports for reproducible real-data commands.

## Optional AI-assisted final analysis

The scientific workflow remains deterministic: eligibility, budget allocation, model training, validation selection, and held-out evaluation are never delegated to an LLM. The final narrative can optionally be generated from the stored profile, planning trace, evaluation results, and decision artifacts.

With the optional dependency installed and an API key configured:

```bash
pip install -e '.[ai]'
ANTHROPIC_API_KEY=... PYTHONPATH=src python -m generative_agent.cli DATASET \
  --objective "Generate synthetic data for internal testing." \
  --output-dir outputs/ai_assisted_run --ai-analysis anthropic \
  --ai-model claude-haiku-5-5
```

The default `--ai-analysis auto` uses Anthropic when `ANTHROPIC_API_KEY` is present, or OpenAI when `OPENAI_API_KEY` is present. Without credentials, it records a transparent deterministic fallback in `ai_assisted_analysis.json`; use `--ai-analysis disabled` to suppress the section explicitly. The generated report never claims that an AI call occurred unless the artifact records `status: generated`.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q
```

## Submission artifacts

- `submission/generated_report_1.md`: UCI Wine Quality demonstration report.
- `submission/generated_report_2.md`: PathMNIST demonstration report.
- `submission/report.pdf`: concise Project 2 report.
- `outputs/phase7_wine_demo/` and `outputs/phase7_pathmnist_demo/`: original demonstration artifacts.

The final recommendations are intentionally conservative: validation ranking is separated from adequacy, privacy, clinical utility, and external-release claims.
