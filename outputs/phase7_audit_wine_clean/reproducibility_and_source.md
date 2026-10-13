# Wine Quality Phase 7 Audit Reproducibility and Source

## Source and license

- Dataset: UCI Wine Quality white-wine dataset, `winequality-white.csv`.
- Local input: `data/uci_wine_quality/winequality-white.csv`.
- Source: UCI Machine Learning Repository, Wine Quality dataset.
- License recorded for this project: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Citation recorded for this project: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T.

## Modeling decision

- Objective: generate synthetic continuous tabular data for internal pipeline testing.
- Modeled variables: the 11 physicochemical variables.
- Metadata/target treatment: `quality` is excluded from unconditional generation and retained only as a target/metadata field for profiling.
- Selection split: validation only.
- Final held-out split: used only after model selection.

## Clean reproduction command

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /opt/miniconda3/bin/python -m generative_agent.cli data/uci_wine_quality/winequality-white.csv --modality tabular --objective 'Generate synthetic continuous tabular data for internal pipeline testing.' --output-dir outputs/phase7_audit_wine_clean --target quality --epochs 5 --max-models 4 --seed 2026 --n-synthetic 1000
```

## Audit note

The clean audit run reproduced the Phase 7 validation scores exactly. The scores are deterministic under the recorded seed, split manifest, preprocessing contract, candidate allocation, and model configurations. The run writes checkpoints and artifacts inside this clean output directory, and no cached validation scores from prior output directories are loaded by the workflow.
