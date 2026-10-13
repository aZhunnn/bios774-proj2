# Phase 2.5 Tabular VAE Report

## Scope
This report is generated from stored workflow results for unconditional tabular synthetic data generation.

## Modeling Assumptions
- Continuous features use a unit-variance Gaussian observation model after training-only standardization.
- Binary features use a Bernoulli likelihood parameterized by decoder logits.
- Within-family selection uses validation loss or validation likelihood only inside that family.
- max_models is total trained generative candidates across all families; bootstrap baseline does not consume a training slot.
- Eligibility is applied before performance ranking; hard eligibility constraints cannot be overridden by high fidelity.
- Cross-family selection uses the predeclared validation selection_score among eligible methods; held-out test data is used only for final evaluation.
- RealNVP flow is available only for continuous tabular features with no missing values.
- GMM and Gaussian copula are available only for continuous tabular features with no missing values.
- Conditional generation, class balancing, image generation, GANs, and diffusion are not implemented.

## Selected Configuration
```json
{
  "method": "training_row_bootstrap"
}
```

## Candidate Search
```json
{
  "planning": {
    "profile_stats": {
      "n_samples": 3429,
      "n_features": 11,
      "continuous_features": 11,
      "binary_features": 0,
      "continuous_only": true,
      "mean_abs_skewness": 1.2970918522974244,
      "mean_abs_correlation": 0.1776881752817795,
      "max_abs_correlation": 0.839905259699901,
      "sample_feature_ratio": 311.72727272727275
    },
    "candidates": [
      {
        "model": "copula",
        "feasible": true,
        "eligible": false,
        "priority_score": -Infinity,
        "reasons": [
          "n=3429, p=11, continuous_only=True.",
          "Model is feasible under deterministic Phase 6B rules."
        ],
        "skip_reason": "No currently implemented method has a formal privacy-preserving mechanism or guarantee."
      },
      {
        "model": "flow",
        "feasible": true,
        "eligible": false,
        "priority_score": -Infinity,
        "reasons": [
          "n=3429, p=11, continuous_only=True.",
          "Model is feasible under deterministic Phase 6B rules."
        ],
        "skip_reason": "No currently implemented method has a formal privacy-preserving mechanism or guarantee."
      },
      {
        "model": "gmm",
        "feasible": true,
        "eligible": false,
        "priority_score": -Infinity,
        "reasons": [
          "n=3429, p=11, continuous_only=True.",
          "Model is feasible under deterministic Phase 6B rules."
        ],
        "skip_reason": "No currently implemented method has a formal privacy-preserving mechanism or guarantee."
      },
      {
        "model": "vae",
        "feasible": true,
        "eligible": false,
        "priority_score": -Infinity,
        "reasons": [
          "n=3429, p=11, continuous_only=True.",
          "Model is feasible under deterministic Phase 6B rules."
        ],
        "skip_reason": "No currently implemented method has a formal privacy-preserving mechanism or guarantee."
      }
    ],
    "allocation": {
      "vae": 0,
      "flow": 0,
      "gmm": 0,
      "copula": 0
    },
    "allocated_models": [],
    "skipped_models": {
      "copula": "No currently implemented method has a formal privacy-preserving mechanism or guarantee.",
      "flow": "No currently implemented method has a formal privacy-preserving mechanism or guarantee.",
      "gmm": "No currently implemented method has a formal privacy-preserving mechanism or guarantee.",
      "vae": "No currently implemented method has a formal privacy-preserving mechanism or guarantee."
    },
    "planning_rule": "Rank feasible and eligible tabular candidates using deterministic dataset-profile scores from modality, feature types, sample size, dimensionality, skewness, correlation, objective, budget, eligibility, and optional prior validation evidence; allocate max_models total trained candidates by priority."
  },
  "budget": {
    "max_trained_candidates": 2,
    "trained_candidates": 0
  }
}
```

## Generation
Generated samples: 1000
Method: rejected_no_eligible_method

## Evaluation
```json
{
  "selected": {
    "method": "rejected",
    "reason": "No implemented model is eligible for the requested objective/use case."
  },
  "baseline": {
    "method": "training_row_bootstrap",
    "feature_differences": {
      "fixed acidity": {
        "test_mean": 6.775034013605442,
        "synthetic_mean": 6.8584,
        "absolute_mean_difference": 0.08336598639455772,
        "test_std": 0.8074604298551417,
        "synthetic_std": 0.8467286696457136,
        "absolute_median_difference": 0.09999999999999964,
        "iqr_difference": 0.09999999999999964
      },
      "volatile acidity": {
        "test_mean": 0.27482312925170066,
        "synthetic_mean": 0.282595,
        "absolute_mean_difference": 0.0077718707482993254,
        "test_std": 0.09522423951688104,
        "synthetic_std": 0.10781275886925443,
        "absolute_median_difference": 0.0,
        "iqr_difference": 2.7755575615628914e-17
      },
      "citric acid": {
        "test_mean": 0.3271156462585034,
        "synthetic_mean": 0.33703,
        "absolute_mean_difference": 0.009914353741496573,
        "test_std": 0.11298236073311288,
        "synthetic_std": 0.12878928177453278,
        "absolute_median_difference": 0.010000000000000009,
        "iqr_difference": 0.0
      },
      "residual sugar": {
        "test_mean": 6.242448979591837,
        "synthetic_mean": 6.64635,
        "absolute_mean_difference": 0.40390102040816256,
        "test_std": 5.063269169496232,
        "synthetic_std": 5.024531737137303,
        "absolute_median_difference": 1.0,
        "iqr_difference": 0.4999999999999991
      },
      "chlorides": {
        "test_mean": 0.04540952380952381,
        "synthetic_mean": 0.045854000000000006,
        "absolute_mean_difference": 0.0004444761904761965,
        "test_std": 0.021792020424802546,
        "synthetic_std": 0.022199609996574266,
        "absolute_median_difference": 0.000999999999999994,
        "iqr_difference": 0.0010000000000000009
      },
      "free sulfur dioxide": {
        "test_mean": 33.76190476190476,
        "synthetic_mean": 34.9745,
        "absolute_mean_difference": 1.2125952380952398,
        "test_std": 16.253302472723544,
        "synthetic_std": 16.208149177188616,
        "absolute_median_difference": 1.0,
        "iqr_difference": 0.0
      },
      "total sulfur dioxide": {
        "test_mean": 134.2687074829932,
        "synthetic_mean": 137.8385,
        "absolute_mean_difference": 3.5697925170068174,
        "test_std": 40.114287075568235,
        "synthetic_std": 42.54023586852804,
        "absolute_median_difference": 3.0,
        "iqr_difference": 6.5
      },
      "density": {
        "test_mean": 0.993851605442177,
        "synthetic_mean": 0.9940851,
        "absolute_mean_difference": 0.00023349455782306627,
        "test_std": 0.0029466864373343797,
        "synthetic_std": 0.002890006019024873,
        "absolute_median_difference": 0.00023999999999990695,
        "iqr_difference": 0.00012000000000000899
      },
      "pH": {
        "test_mean": 3.1953605442176873,
        "synthetic_mean": 3.1851700000000003,
        "absolute_mean_difference": 0.010190544217687059,
        "test_std": 0.15403718720452117,
        "synthetic_std": 0.1468889754202132,
        "absolute_median_difference": 0.0,
        "iqr_difference": 4.440892098500626e-16
      },
      "sulphates": {
        "test_mean": 0.48455782312925166,
        "synthetic_mean": 0.49087000000000003,
        "absolute_mean_difference": 0.006312176870748365,
        "test_std": 0.11038480185068494,
        "synthetic_std": 0.11657848472166724,
        "absolute_median_difference": 0.019999999999999962,
        "iqr_difference": 5.551115123125783e-17
      },
      "alcohol": {
        "test_mean": 10.551224489795917,
        "synthetic_mean": 10.4858,
        "absolute_mean_difference": 0.06542448979591775,
        "test_std": 1.2455696169595525,
        "synthetic_std": 1.217519803169998,
        "absolute_median_difference": 0.0,
        "iqr_difference": 0.0
      }
    },
    "correlation_l1": 0.032526009379635484,
    "duplicate_fraction": 0.178,
    "unique_fraction": 0.8220000000000001,
    "nearest_neighbor_mean": 0.0,
    "nearest_neighbor_min": 0.0,
    "marginal_fidelity_score": 0.652144190630799,
    "train_scale_normalized_marginal_fidelity": 0.943862011835702,
    "dependence_preservation_score": 0.968498605280483,
    "selection_score": 0.8941412553003947,
    "objective_metrics": {
      "data_generation_fidelity": 0.652144190630799,
      "train_scale_normalized_marginal_fidelity": 0.943862011835702,
      "dependence_preservation": 0.968498605280483,
      "selection_score": 0.8941412553003947
    },
    "notes": [
      "Nearest-neighbor distances are memorization proxies, not formal privacy guarantees.",
      "Bootstrap baseline is evaluated with the same metrics for fairness.",
      "Selection score is 0.45 train-scale-normalized marginal fidelity, 0.35 dependence preservation when valid, and 0.20 legacy marginal fidelity."
    ]
  }
}
```

## Agent Decisions
```json
{
  "feasible": true,
  "reasons": [],
  "warnings": [],
  "constraints": [
    "max_models=2",
    "max_epochs=5",
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
      "n_samples=4898",
      "n_features=12",
      "max_models=2",
      "max_epochs=5",
      "max_wall_minutes=30",
      "seed=2026"
    ],
    "supported_objective": true,
    "uses_generative_model": true
  },
  "selection": "Selected rejected using shared validation selection_score; held-out test was not used.",
  "validation_comparison": {
    "planning": {
      "profile_stats": {
        "n_samples": 3429,
        "n_features": 11,
        "continuous_features": 11,
        "binary_features": 0,
        "continuous_only": true,
        "mean_abs_skewness": 1.2970918522974244,
        "mean_abs_correlation": 0.1776881752817795,
        "max_abs_correlation": 0.839905259699901,
        "sample_feature_ratio": 311.72727272727275
      },
      "candidates": [
        {
          "model": "copula",
          "feasible": true,
          "eligible": false,
          "priority_score": -Infinity,
          "reasons": [
            "n=3429, p=11, continuous_only=True.",
            "Model is feasible under deterministic Phase 6B rules."
          ],
          "skip_reason": "No currently implemented method has a formal privacy-preserving mechanism or guarantee."
        },
        {
          "model": "flow",
          "feasible": true,
          "eligible": false,
          "priority_score": -Infinity,
          "reasons": [
            "n=3429, p=11, continuous_only=True.",
            "Model is feasible under deterministic Phase 6B rules."
          ],
          "skip_reason": "No currently implemented method has a formal privacy-preserving mechanism or guarantee."
        },
        {
          "model": "gmm",
          "feasible": true,
          "eligible": false,
          "priority_score": -Infinity,
          "reasons": [
            "n=3429, p=11, continuous_only=True.",
            "Model is feasible under deterministic Phase 6B rules."
          ],
          "skip_reason": "No currently implemented method has a formal privacy-preserving mechanism or guarantee."
        },
        {
          "model": "vae",
          "feasible": true,
          "eligible": false,
          "priority_score": -Infinity,
          "reasons": [
            "n=3429, p=11, continuous_only=True.",
            "Model is feasible under deterministic Phase 6B rules."
          ],
          "skip_reason": "No currently implemented method has a formal privacy-preserving mechanism or guarantee."
        }
      ],
      "allocation": {
        "vae": 0,
        "flow": 0,
        "gmm": 0,
        "copula": 0
      },
      "allocated_models": [],
      "skipped_models": {
        "copula": "No currently implemented method has a formal privacy-preserving mechanism or guarantee.",
        "flow": "No currently implemented method has a formal privacy-preserving mechanism or guarantee.",
        "gmm": "No currently implemented method has a formal privacy-preserving mechanism or guarantee.",
        "vae": "No currently implemented method has a formal privacy-preserving mechanism or guarantee."
      },
      "planning_rule": "Rank feasible and eligible tabular candidates using deterministic dataset-profile scores from modality, feature types, sample size, dimensionality, skewness, correlation, objective, budget, eligibility, and optional prior validation evidence; allocate max_models total trained candidates by priority."
    },
    "objective_recommendation": {
      "use_case": "formal_privacy",
      "selected_family": "rejected",
      "relative_ranking": [],
      "selected_validation_score": null,
      "adequacy_warning_threshold": 0.85,
      "adequacy_claim": false,
      "recommendation": "reject",
      "reasons": [
        "No implemented method provides a formal privacy guarantee."
      ],
      "eligibility_summary": {
        "selected_method": null,
        "selection_reason": "No implemented model is eligible for the requested objective/use case."
      },
      "language_guardrail": "Eligibility and high relative fidelity do not establish privacy, safety, clinical utility, or external-release suitability."
    }
  },
  "eligibility": {},
  "objective_recommendation": {
    "use_case": "formal_privacy",
    "selected_family": "rejected",
    "relative_ranking": [],
    "selected_validation_score": null,
    "adequacy_warning_threshold": 0.85,
    "adequacy_claim": false,
    "recommendation": "reject",
    "reasons": [
      "No implemented method provides a formal privacy guarantee."
    ],
    "eligibility_summary": {
      "selected_method": null,
      "selection_reason": "No implemented model is eligible for the requested objective/use case."
    },
    "language_guardrail": "Eligibility and high relative fidelity do not establish privacy, safety, clinical utility, or external-release suitability."
  },
  "model_limitations": {
    "vae": "Supports continuous and binary tabular features with Gaussian/Bernoulli observation assumptions.",
    "flow": "Phase 3A flow supports continuous-only tabular features with no missing values.",
    "gmm": "Gaussian mixtures support continuous-only tabular data and can be sensitive to component count and covariance regularization.",
    "copula": "Gaussian copula supports continuous-only tabular data with empirical marginals and linear latent dependence; it does not capture arbitrary nonlinear or tail dependence.",
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
