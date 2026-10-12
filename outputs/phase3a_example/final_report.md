# Phase 2.5 Tabular VAE Report

## Scope
This report is generated from stored workflow results for unconditional tabular synthetic data generation.

## Modeling Assumptions
- Continuous features use a unit-variance Gaussian observation model after training-only standardization.
- Binary features use a Bernoulli likelihood parameterized by decoder logits.
- Within-family selection uses validation loss or validation likelihood only inside that family.
- Cross-family selection uses shared validation synthetic-data fidelity; held-out test data is used only for final evaluation.
- RealNVP flow is available only for continuous tabular features with no missing values.
- Conditional generation, class balancing, image generation, GANs, and diffusion are not implemented.

## Selected Configuration
```json
{
  "input_dim": 2,
  "continuous_dim": 2,
  "binary_dim": 0,
  "latent_dim": 2,
  "hidden_dims": [
    16
  ],
  "learning_rate": 0.001,
  "batch_size": 32,
  "epochs": 3,
  "kl_weight": 1.0,
  "early_stopping_patience": 3,
  "seed": 2026,
  "device": "cpu"
}
```

## Candidate Search
```json
{
  "vae": {
    "candidates": [
      {
        "config": {
          "input_dim": 2,
          "continuous_dim": 2,
          "binary_dim": 0,
          "latent_dim": 2,
          "hidden_dims": [
            16
          ],
          "learning_rate": 0.001,
          "batch_size": 32,
          "epochs": 3,
          "kl_weight": 1.0,
          "early_stopping_patience": 3,
          "seed": 2026,
          "device": "cpu"
        },
        "validation_loss": 3.044768810272217,
        "epochs_ran": 3,
        "wall_seconds": 0.010637958068400621,
        "checkpoint_path": "outputs/phase3a_example/tabular_vae_candidate_0.pt",
        "failed": false,
        "failure_reason": null,
        "history": {
          "train_loss": [
            3.3121891021728516,
            2.5935033559799194,
            3.3223615884780884
          ],
          "validation_loss": [
            3.044768810272217,
            3.0482521057128906,
            3.066857099533081
          ]
        }
      }
    ],
    "selected_index": 0,
    "selection_reason": "Selected lowest validation loss among successful candidates; test data was not used."
  },
  "flow": {
    "candidates": [
      {
        "config": {
          "input_dim": 2,
          "num_layers": 4,
          "hidden_dim": 32,
          "learning_rate": 0.001,
          "batch_size": 32,
          "epochs": 3,
          "early_stopping_patience": 3,
          "seed": 2127,
          "device": "cpu"
        },
        "validation_nll": 2.809579610824585,
        "epochs_ran": 3,
        "wall_seconds": 0.007296833209693432,
        "checkpoint_path": "outputs/phase3a_example/tabular_flow_candidate_0.pt",
        "failed": false,
        "failure_reason": null,
        "history": {
          "train_nll": [
            3.1606215238571167,
            2.913098692893982,
            2.501747250556946
          ],
          "validation_nll": [
            2.950209856033325,
            2.8764004707336426,
            2.809579610824585
          ]
        }
      }
    ],
    "selected_index": 0,
    "selection_reason": "Selected lowest validation negative log likelihood within flow family; test data was not used."
  }
}
```

## Generation
Generated samples: 8
Method: tabular_vae

## Evaluation
```json
{
  "selected": {
    "method": "tabular_vae",
    "feature_differences": {
      "x1": {
        "test_mean": 0.08510638297872342,
        "synthetic_mean": -0.015769075602293015,
        "absolute_mean_difference": 0.10087545858101643,
        "test_std": 1.2914034813185973,
        "synthetic_std": 0.21387246196409807,
        "absolute_median_difference": 0.06842008233070362,
        "iqr_difference": 2.273881861345565
      },
      "x2": {
        "test_mean": 0.04934319667158937,
        "synthetic_mean": -0.037453629076480865,
        "absolute_mean_difference": 0.08679682574807024,
        "test_std": 1.4858811704197328,
        "synthetic_std": 0.3548020283668253,
        "absolute_median_difference": 0.06935511529445632,
        "iqr_difference": 3.0329482456360393
      }
    },
    "correlation_l1": 0.8318251427540302,
    "duplicate_fraction": 0.0,
    "unique_fraction": 1.0,
    "nearest_neighbor_mean": 0.21956683436490565,
    "nearest_neighbor_min": 0.08332949311577867,
    "marginal_fidelity_score": 0.5158061225865995,
    "objective_metrics": {
      "data_generation_fidelity": 0.5158061225865995
    },
    "notes": [
      "Nearest-neighbor distances are memorization proxies, not formal privacy guarantees.",
      "Bootstrap baseline is evaluated with the same metrics for fairness."
    ]
  },
  "vae": {
    "method": "tabular_vae",
    "feature_differences": {
      "x1": {
        "test_mean": 0.08510638297872342,
        "synthetic_mean": -0.015769075602293015,
        "absolute_mean_difference": 0.10087545858101643,
        "test_std": 1.2914034813185973,
        "synthetic_std": 0.21387246196409807,
        "absolute_median_difference": 0.06842008233070362,
        "iqr_difference": 2.273881861345565
      },
      "x2": {
        "test_mean": 0.04934319667158937,
        "synthetic_mean": -0.037453629076480865,
        "absolute_mean_difference": 0.08679682574807024,
        "test_std": 1.4858811704197328,
        "synthetic_std": 0.3548020283668253,
        "absolute_median_difference": 0.06935511529445632,
        "iqr_difference": 3.0329482456360393
      }
    },
    "correlation_l1": 0.8318251427540302,
    "duplicate_fraction": 0.0,
    "unique_fraction": 1.0,
    "nearest_neighbor_mean": 0.21956683436490565,
    "nearest_neighbor_min": 0.08332949311577867,
    "marginal_fidelity_score": 0.5158061225865995,
    "objective_metrics": {
      "data_generation_fidelity": 0.5158061225865995
    },
    "notes": [
      "Nearest-neighbor distances are memorization proxies, not formal privacy guarantees.",
      "Bootstrap baseline is evaluated with the same metrics for fairness."
    ]
  },
  "baseline": {
    "method": "training_row_bootstrap",
    "feature_differences": {
      "x1": {
        "test_mean": 0.08510638297872342,
        "synthetic_mean": -0.3085106382978723,
        "absolute_mean_difference": 0.3936170212765957,
        "test_std": 1.2914034813185973,
        "synthetic_std": 1.2460439889721386,
        "absolute_median_difference": 0.6382978723404255,
        "iqr_difference": 0.08510638297872353
      },
      "x2": {
        "test_mean": 0.04934319667158937,
        "synthetic_mean": -0.32699009644217997,
        "absolute_mean_difference": 0.37633329311376934,
        "test_std": 1.4858811704197328,
        "synthetic_std": 1.471038642854338,
        "absolute_median_difference": 0.91497824370811,
        "iqr_difference": 0.1042196745538706
      }
    },
    "correlation_l1": 0.001141278472064855,
    "duplicate_fraction": 0.125,
    "unique_fraction": 0.875,
    "nearest_neighbor_mean": 0.0,
    "nearest_neighbor_min": 0.0,
    "marginal_fidelity_score": 0.7048414689340465,
    "objective_metrics": {
      "data_generation_fidelity": 0.7048414689340465
    },
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
  "objective_decision": {
    "action": "use_vae",
    "reasons": [
      "Unconditional tabular data_generation is supported and budget permits VAE training."
    ],
    "constraints": [
      "objective=data_generation",
      "n_samples=48",
      "n_features=4",
      "max_models=1",
      "max_epochs=3",
      "max_wall_minutes=30",
      "seed=2026"
    ],
    "supported_objective": true,
    "uses_generative_model": true
  },
  "selection": "Selected vae using shared validation data_generation_fidelity; held-out test was not used.",
  "validation_comparison": {
    "decision_rule": "Select the method with the highest validation data_generation_fidelity from synthetic-vs-validation evaluation. VAE ELBO and flow NLL are used only within their own family searches.",
    "scores": {
      "baseline": {
        "data_generation_fidelity": 0.44775650481469664,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 0.0
      },
      "vae": {
        "data_generation_fidelity": 0.5036231654767201,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 0.12958793890102063
      },
      "flow": {
        "data_generation_fidelity": 0.428887410646848,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 0.8432040625418109
      }
    }
  },
  "model_limitations": {
    "vae": "Supports continuous and binary tabular features with Gaussian/Bernoulli observation assumptions.",
    "flow": "Phase 3A flow supports continuous-only tabular features with no missing values.",
    "baseline": "Bootstrap resamples training rows and is non-generative."
  },
  "planner": "deterministic rules plus within-family searches and shared validation comparison; no LLM reasoning or autonomous optimization"
}
```

## Limitations
- Nearest-neighbor diagnostics are proxies, not privacy guarantees.
- The decision layer is deterministic and does not perform LLM-based reasoning.
- No downstream predictive utility is reported unless a separate valid protocol is implemented.
- Cross-family validation scores are finite-sample heuristics, not statistical uncertainty estimates.
