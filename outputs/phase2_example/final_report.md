# Phase 2 Tabular VAE Report

## Scope
This report is generated from stored Phase 2 workflow results for unconditional tabular synthetic data generation.

## Modeling Assumptions
- Continuous features use a unit-variance Gaussian observation model after training-only standardization.
- Binary features use a Bernoulli likelihood parameterized by decoder logits.
- Conditional generation, class balancing, image generation, and non-VAE model families are not implemented in Phase 2.

## Configuration
```json
{
  "input_dim": 3,
  "continuous_dim": 2,
  "binary_dim": 1,
  "latent_dim": 3,
  "hidden_dims": [
    32,
    16
  ],
  "learning_rate": 0.001,
  "batch_size": 11,
  "epochs": 3,
  "kl_weight": 1.0,
  "early_stopping_patience": 5,
  "seed": 2026,
  "device": "cpu"
}
```

## Training
Best validation loss: 4.180056095123291
Epochs run: 3

## Generation
Generated samples: 6

## Evaluation
```json
{
  "vae": {
    "method": "tabular_vae",
    "feature_differences": {
      "x1": {
        "test_mean": -0.85,
        "synthetic_mean": 0.21131793657938638,
        "absolute_mean_difference": 1.0613179365793863,
        "test_std": 0.15000000000000002,
        "synthetic_std": 0.010464838740260394
      },
      "x2": {
        "test_mean": 0.07500000000000001,
        "synthetic_mean": 0.2570454527934392,
        "absolute_mean_difference": 0.1820454527934392,
        "test_std": 0.025,
        "synthetic_std": 0.00961452299338102
      },
      "flag": {
        "test_positive_rate": 0.0,
        "synthetic_positive_rate": 0.0,
        "absolute_rate_difference": 0.0
      }
    },
    "correlation_l1": 0.37112529974253594,
    "duplicate_fraction": 0.0,
    "nearest_neighbor_mean": 0.9898834918977643,
    "nearest_neighbor_min": 0.6495153731889535,
    "notes": [
      "Nearest-neighbor distances are memorization proxies, not formal privacy guarantees.",
      "Bootstrap baseline is evaluated with the same metrics for fairness."
    ]
  },
  "baseline": {
    "method": "training_row_bootstrap",
    "feature_differences": {
      "x1": {
        "test_mean": -0.85,
        "synthetic_mean": 0.3,
        "absolute_mean_difference": 1.15,
        "test_std": 0.15000000000000002,
        "synthetic_std": 0.6164414002968976
      },
      "x2": {
        "test_mean": 0.07500000000000001,
        "synthetic_mean": 0.2333333333333333,
        "absolute_mean_difference": 0.1583333333333333,
        "test_std": 0.025,
        "synthetic_std": 0.19075871903765995
      },
      "flag": {
        "test_positive_rate": 0.0,
        "synthetic_positive_rate": 0.6666666666666666,
        "absolute_rate_difference": 0.6666666666666666
      }
    },
    "correlation_l1": 0.6559068650911923,
    "duplicate_fraction": 0.0,
    "nearest_neighbor_mean": 0.1644358527333569,
    "nearest_neighbor_min": 0.09647221898511127,
    "notes": [
      "Nearest-neighbor distances are memorization proxies, not formal privacy guarantees.",
      "Bootstrap baseline is evaluated with the same metrics for fairness."
    ]
  }
}
```

## Agent Decisions
```json
{
  "feasible": true,
  "reasons": [],
  "warnings": [
    "Small sample size may limit generative model quality."
  ],
  "constraints": [
    "max_models=1",
    "max_epochs=3",
    "max_wall_minutes=30",
    "seed=2026"
  ],
  "planner": "deterministic MockPlanner; no LLM reasoning or autonomous optimization"
}
```

## Limitations
- Nearest-neighbor diagnostics are proxies, not privacy guarantees.
- The planner is deterministic and hardcoded; it does not perform LLM-based reasoning.
- No downstream predictive utility is reported unless a separate valid protocol is implemented.
