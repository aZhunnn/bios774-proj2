# Project 2 Readiness Contracts

Project 2 exposes reusable contracts for a future Project 4 biomedical agent.
This document describes interfaces, not Project 4 integration.

## Dataset Profiles

- Tabular datasets are represented by `DatasetHandle`, `DatasetProfile`, and
  train-only `TabularVaePreprocessorMetadata`.
- Image datasets are represented by `ImageDatasetBundle` and
  `ImageDatasetProfile`.
- Split manifests and preprocessing metadata are persisted as artifacts.
- Labels and targets may be retained as metadata while excluded from
  unconditional generation.

## Objectives

- User text is interpreted into `ObjectiveSpec`.
- The current use-case dimension is `internal_testing`, `external_sharing`, or
  `formal_privacy`.
- Unsupported objectives are rejected rather than simulated.

## Capabilities

- `CapabilityRegistry` declares modality, feature types, objectives,
  operations, computational requirements, implementation status, and
  limitations.
- Implemented Project 2 families are tabular VAE, RealNVP, GMM, Gaussian
  copula, image VAE, image GAN, and bootstrap baseline.

## Planning

- Tabular planning is handled by `build_tabular_training_plan`.
- It uses sample size, dimensionality, continuous/binary feature support,
  skewness, dependence, objective, eligibility, and budget.
- Image planning records candidate priorities and budget allocation in the same
  artifact shape, while using image-specific model assumptions.
- Prior validation evidence is ignored by default unless provenance checks pass.

## Budgets

- `ExperimentBudget.max_models` means total trained candidates across model
  families.
- Bootstrap baseline does not consume a training slot.
- Held-out test data must not influence planning or validation selection.

## Evaluation

- Tabular selection uses shared synthetic-vs-validation fidelity metrics.
- Image selection uses image-specific generation metrics; it is not forced onto
  tabular scores.
- Model-specific losses and likelihoods are used only within a family.

## Recommendation

- Eligibility is a hard gate before ranking.
- Relative validation ranking is separate from adequacy and release suitability.
- No implemented method provides formal privacy guarantees.
- Reports must state limitations and unsupported capabilities.

## Artifacts

Workflows persist JSON artifacts for profile, objective, split, preprocessing,
planning/search, validation comparison, generation metadata, evaluation,
decisions, model config, and final report.
