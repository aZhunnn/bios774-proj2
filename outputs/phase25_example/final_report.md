# Phase 2.5 Tabular VAE Report

## Scope
This report is generated from stored workflow results for unconditional tabular synthetic data generation.

## Modeling Assumptions
- Continuous features use a unit-variance Gaussian observation model after training-only standardization.
- Binary features use a Bernoulli likelihood parameterized by decoder logits.
- Candidate selection uses validation loss only; held-out test data is used only for final evaluation.
- Conditional generation, class balancing, image generation, and non-VAE model families are not implemented.

## Selected Configuration
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
  "batch_size": 17,
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
  "candidates": [
    {
      "config": {
        "input_dim": 3,
        "continuous_dim": 2,
        "binary_dim": 1,
        "latent_dim": 3,
        "hidden_dims": [
          32,
          16
        ],
        "learning_rate": 0.001,
        "batch_size": 17,
        "epochs": 3,
        "kl_weight": 1.0,
        "early_stopping_patience": 3,
        "seed": 2026,
        "device": "cpu"
      },
      "validation_loss": 3.446455717086792,
      "epochs_ran": 3,
      "wall_seconds": 0.01141933398321271,
      "checkpoint_path": "outputs/phase25_example/tabular_vae_candidate_0.pt",
      "failed": false,
      "failure_reason": null,
      "history": {
        "train_loss": [
          3.5697460174560547,
          3.6053411960601807,
          3.5709900856018066
        ],
        "validation_loss": [
          3.474095582962036,
          3.4706923961639404,
          3.446455717086792
        ]
      }
    },
    {
      "config": {
        "input_dim": 3,
        "continuous_dim": 2,
        "binary_dim": 1,
        "latent_dim": 2,
        "hidden_dims": [
          16
        ],
        "learning_rate": 0.0005,
        "batch_size": 17,
        "epochs": 3,
        "kl_weight": 1.0,
        "early_stopping_patience": 3,
        "seed": 2043,
        "device": "cpu"
      },
      "validation_loss": 3.7476327419281006,
      "epochs_ran": 3,
      "wall_seconds": 0.00308320764452219,
      "checkpoint_path": "outputs/phase25_example/tabular_vae_candidate_1.pt",
      "failed": false,
      "failure_reason": null,
      "history": {
        "train_loss": [
          3.587759256362915,
          3.692610025405884,
          3.6776349544525146
        ],
        "validation_loss": [
          3.826096296310425,
          3.7839767932891846,
          3.7476327419281006
        ]
      }
    }
  ],
  "selected_index": 0,
  "selection_reason": "Selected lowest validation loss among successful candidates; test data was not used."
}
```

## Generation
Generated samples: 6
Method: tabular_vae

## Evaluation
```json
{
  "selected": {
    "method": "tabular_vae",
    "feature_differences": {
      "x1": {
        "test_mean": 0.08749999999999997,
        "synthetic_mean": 0.7971739669640859,
        "absolute_mean_difference": 0.7096739669640859,
        "test_std": 0.5649944689994761,
        "synthetic_std": 0.022167887284144726,
        "absolute_median_difference": 0.8212971031665802,
        "iqr_difference": 0.671814227104187
      },
      "x2": {
        "test_mean": 0.22499999999999998,
        "synthetic_mean": 0.3939974308013916,
        "absolute_mean_difference": 0.16899743080139162,
        "test_std": 0.20155644370746373,
        "synthetic_std": 0.01285950717009557,
        "absolute_median_difference": 0.1699520289897919,
        "iqr_difference": 0.24172306060791016
      },
      "flag": {
        "test_positive_rate": 0.75,
        "synthetic_positive_rate": 0.0,
        "absolute_rate_difference": 0.75
      }
    },
    "correlation_l1": 0.2443331355941375,
    "duplicate_fraction": 0.0,
    "unique_fraction": 1.0,
    "nearest_neighbor_mean": 0.6649390113126706,
    "nearest_neighbor_min": 0.5907901002581565,
    "marginal_fidelity_score": 0.6645491082976928,
    "objective_metrics": {
      "data_generation_fidelity": 0.6645491082976928
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
        "test_mean": 0.08749999999999997,
        "synthetic_mean": 0.7971739669640859,
        "absolute_mean_difference": 0.7096739669640859,
        "test_std": 0.5649944689994761,
        "synthetic_std": 0.022167887284144726,
        "absolute_median_difference": 0.8212971031665802,
        "iqr_difference": 0.671814227104187
      },
      "x2": {
        "test_mean": 0.22499999999999998,
        "synthetic_mean": 0.3939974308013916,
        "absolute_mean_difference": 0.16899743080139162,
        "test_std": 0.20155644370746373,
        "synthetic_std": 0.01285950717009557,
        "absolute_median_difference": 0.1699520289897919,
        "iqr_difference": 0.24172306060791016
      },
      "flag": {
        "test_positive_rate": 0.75,
        "synthetic_positive_rate": 0.0,
        "absolute_rate_difference": 0.75
      }
    },
    "correlation_l1": 0.2443331355941375,
    "duplicate_fraction": 0.0,
    "unique_fraction": 1.0,
    "nearest_neighbor_mean": 0.6649390113126706,
    "nearest_neighbor_min": 0.5907901002581565,
    "marginal_fidelity_score": 0.6645491082976928,
    "objective_metrics": {
      "data_generation_fidelity": 0.6645491082976928
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
        "test_mean": 0.08749999999999997,
        "synthetic_mean": 0.9,
        "absolute_mean_difference": 0.8125,
        "test_std": 0.5649944689994761,
        "synthetic_std": 0.9974968671630001,
        "absolute_median_difference": 1.125,
        "iqr_difference": 0.7125
      },
      "x2": {
        "test_mean": 0.22499999999999998,
        "synthetic_mean": 0.425,
        "absolute_mean_difference": 0.2,
        "test_std": 0.20155644370746373,
        "synthetic_std": 0.2940379794062892,
        "absolute_median_difference": 0.15000000000000002,
        "iqr_difference": 0.21250000000000008
      },
      "flag": {
        "test_positive_rate": 0.75,
        "synthetic_positive_rate": 0.6666666666666666,
        "absolute_rate_difference": 0.08333333333333337
      }
    },
    "correlation_l1": 0.16695175418254715,
    "duplicate_fraction": 0.16666666666666666,
    "unique_fraction": 0.8333333333333334,
    "nearest_neighbor_mean": 0.0,
    "nearest_neighbor_min": 0.0,
    "marginal_fidelity_score": 0.6798866855524079,
    "objective_metrics": {
      "data_generation_fidelity": 0.6798866855524079
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
    "max_models=2",
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
      "n_samples=24",
      "n_features=5",
      "max_models=2",
      "max_epochs=3",
      "max_wall_minutes=30",
      "seed=2026"
    ],
    "supported_objective": true,
    "uses_generative_model": true
  },
  "selection": "Selected lowest validation loss among successful candidates; test data was not used.",
  "planner": "deterministic rules plus validation-only candidate search; no LLM reasoning or autonomous optimization"
}
```

## Limitations
- Nearest-neighbor diagnostics are proxies, not privacy guarantees.
- The decision layer is deterministic and does not perform LLM-based reasoning.
- No downstream predictive utility is reported unless a separate valid protocol is implemented.
