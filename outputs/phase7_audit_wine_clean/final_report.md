# Project 2 Tabular Generative Agent Report

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
- Conditional generation, imputation, formal privacy mechanisms, Bayesian decision-making, diffusion, and Project 4 integration are not implemented.

## Selected Configuration
```json
{
  "method": "training_row_bootstrap"
}
```

## Candidate Search
```json
{
  "vae": {
    "candidates": [
      {
        "config": {
          "input_dim": 11,
          "continuous_dim": 11,
          "binary_dim": 0,
          "latent_dim": 4,
          "hidden_dims": [
            32,
            16
          ],
          "learning_rate": 0.001,
          "batch_size": 32,
          "epochs": 5,
          "kl_weight": 1.0,
          "early_stopping_patience": 5,
          "seed": 2026,
          "device": "cpu"
        },
        "validation_loss": 15.23876953125,
        "epochs_ran": 5,
        "wall_seconds": 0.2566531249321997,
        "checkpoint_path": "outputs/phase7_audit_wine_clean/tabular_vae_candidate_0.pt",
        "failed": false,
        "failure_reason": null,
        "history": {
          "train_loss": [
            15.650201347139147,
            15.525706891660336,
            15.234358840518528,
            15.169126916814733,
            15.102156762723569
          ],
          "validation_loss": [
            15.773663520812988,
            15.548996925354004,
            15.310150146484375,
            15.23876953125,
            15.249807357788086
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
          "input_dim": 11,
          "num_layers": 4,
          "hidden_dim": 32,
          "learning_rate": 0.001,
          "batch_size": 32,
          "epochs": 5,
          "early_stopping_patience": 5,
          "seed": 2127,
          "device": "cpu"
        },
        "validation_nll": 11.842278480529785,
        "epochs_ran": 5,
        "wall_seconds": 0.40667158318683505,
        "checkpoint_path": "outputs/phase7_audit_wine_clean/tabular_flow_candidate_0.pt",
        "failed": false,
        "failure_reason": null,
        "history": {
          "train_nll": [
            15.124412713227448,
            13.796849065356785,
            12.754572100109524,
            12.102241613246777,
            11.79441112059134
          ],
          "validation_nll": [
            14.3749418258667,
            13.26424789428711,
            12.397820472717285,
            12.058106422424316,
            11.842278480529785
          ]
        }
      }
    ],
    "selected_index": 0,
    "selection_reason": "Selected lowest validation negative log likelihood within flow family; test data was not used."
  },
  "gmm": {
    "candidates": [
      {
        "config": {
          "n_components": 1,
          "covariance_type": "full",
          "reg_covar": 1e-05,
          "max_iter": 200,
          "seed": 2057
        },
        "diagnostics": {
          "converged": true,
          "n_iter": 2,
          "lower_bound": -13.141667366027832,
          "weights": [
            1.0
          ],
          "means": [
            [
              -3.0210809853770115e-08,
              -4.839292344627211e-08,
              -1.6673308778081264e-07,
              6.431530419348519e-09,
              1.1124809518037182e-08,
              9.289215796570716e-08,
              6.118645146102608e-09,
              -1.0290449381500366e-08,
              1.668721409942009e-07,
              -4.449923984850557e-09,
              -1.0874501299440453e-07
            ]
          ],
          "covariance_type": "full",
          "reg_covar": 1e-05
        },
        "checkpoint_path": "outputs/phase7_audit_wine_clean/tabular_gmm_candidate_0.pkl",
        "wall_seconds": 0.02664020797237754,
        "failed": false,
        "failure_reason": null,
        "validation_log_likelihood": -13.116392135620117
      }
    ],
    "selected_index": 0,
    "selected": {
      "config": {
        "n_components": 1,
        "covariance_type": "full",
        "reg_covar": 1e-05,
        "max_iter": 200,
        "seed": 2057
      },
      "diagnostics": {
        "converged": true,
        "n_iter": 2,
        "lower_bound": -13.141667366027832,
        "weights": [
          1.0
        ],
        "means": [
          [
            -3.0210809853770115e-08,
            -4.839292344627211e-08,
            -1.6673308778081264e-07,
            6.431530419348519e-09,
            1.1124809518037182e-08,
            9.289215796570716e-08,
            6.118645146102608e-09,
            -1.0290449381500366e-08,
            1.668721409942009e-07,
            -4.449923984850557e-09,
            -1.0874501299440453e-07
          ]
        ],
        "covariance_type": "full",
        "reg_covar": 1e-05
      },
      "checkpoint_path": "outputs/phase7_audit_wine_clean/tabular_gmm_candidate_0.pkl",
      "wall_seconds": 0.02664020797237754,
      "failed": false,
      "failure_reason": null,
      "validation_log_likelihood": -13.116392135620117
    },
    "selection_reason": "Selected highest validation average log likelihood within GMM family; test data was not used."
  },
  "copula": {
    "candidates": [
      {
        "config": {
          "correlation_regularization": 1e-06,
          "min_u": 1e-06,
          "seed": 2067
        },
        "diagnostics": {
          "n_features": 11,
          "min_eigenvalue": 0.04430045722200077,
          "correlation_regularization": 1e-06,
          "marginal_estimator": "empirical_quantile_linear_interpolation",
          "limitations": [
            "Empirical marginals reproduce observed support approximately and do not extrapolate tails beyond training extremes.",
            "Gaussian copula captures linear latent dependence, not arbitrary nonlinear or tail dependence."
          ]
        },
        "checkpoint_path": "outputs/phase7_audit_wine_clean/tabular_copula_candidate_0.pkl",
        "wall_seconds": 0.004943375010043383,
        "failed": false,
        "failure_reason": null,
        "validation_dependence_score": -0.03825878186115019
      }
    ],
    "selected_index": 0,
    "selected": {
      "config": {
        "correlation_regularization": 1e-06,
        "min_u": 1e-06,
        "seed": 2067
      },
      "diagnostics": {
        "n_features": 11,
        "min_eigenvalue": 0.04430045722200077,
        "correlation_regularization": 1e-06,
        "marginal_estimator": "empirical_quantile_linear_interpolation",
        "limitations": [
          "Empirical marginals reproduce observed support approximately and do not extrapolate tails beyond training extremes.",
          "Gaussian copula captures linear latent dependence, not arbitrary nonlinear or tail dependence."
        ]
      },
      "checkpoint_path": "outputs/phase7_audit_wine_clean/tabular_copula_candidate_0.pkl",
      "wall_seconds": 0.004943375010043383,
      "failed": false,
      "failure_reason": null,
      "validation_dependence_score": -0.03825878186115019
    },
    "selection_reason": "Selected highest validation dependence proxy within copula family; test data was not used."
  },
  "budget": {
    "semantics": "max_models is the total number of trained generative candidates across all model families. Bootstrap baseline does not consume a training slot.",
    "max_trained_candidates": 4,
    "trained_candidates": 4,
    "allocation": {
      "vae": 1,
      "flow": 1,
      "gmm": 1,
      "copula": 1
    }
  },
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
        "model": "flow",
        "feasible": true,
        "eligible": true,
        "priority_score": 0.7615096120531748,
        "reasons": [
          "n=3429, p=11, continuous_only=True.",
          "Model is feasible under deterministic Phase 6B rules.",
          "Objective is unconditional synthetic data generation."
        ],
        "skip_reason": null
      },
      {
        "model": "gmm",
        "feasible": true,
        "eligible": true,
        "priority_score": 0.7206612905169067,
        "reasons": [
          "n=3429, p=11, continuous_only=True.",
          "Model is feasible under deterministic Phase 6B rules.",
          "Objective is unconditional synthetic data generation."
        ],
        "skip_reason": null
      },
      {
        "model": "copula",
        "feasible": true,
        "eligible": true,
        "priority_score": 0.7094654606007138,
        "reasons": [
          "n=3429, p=11, continuous_only=True.",
          "Model is feasible under deterministic Phase 6B rules.",
          "Objective is unconditional synthetic data generation."
        ],
        "skip_reason": null
      },
      {
        "model": "vae",
        "feasible": true,
        "eligible": true,
        "priority_score": 0.6800000000000002,
        "reasons": [
          "n=3429, p=11, continuous_only=True.",
          "Model is feasible under deterministic Phase 6B rules.",
          "Objective is unconditional synthetic data generation."
        ],
        "skip_reason": null
      }
    ],
    "allocation": {
      "vae": 1,
      "flow": 1,
      "gmm": 1,
      "copula": 1
    },
    "allocated_models": [
      "flow",
      "gmm",
      "copula",
      "vae"
    ],
    "skipped_models": {},
    "planning_rule": "Rank feasible and eligible tabular candidates using deterministic dataset-profile scores from modality, feature types, sample size, dimensionality, skewness, correlation, objective, budget, eligibility, and optional prior validation evidence; allocate max_models total trained candidates by priority.",
    "prior_evidence_status": {
      "used": false,
      "reason": "Prior evidence reuse is disabled by default and no prior evidence was provided.",
      "dataset_fingerprint": "9b9c6067b5cde6bce547d49ad9bd507446d343055064db20cd505a4d94351222"
    }
  }
}
```

## Generation
Generated samples: 1000
Method: training_row_bootstrap_selected_by_validation

## Evaluation
```json
{
  "selected": {
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
  },
  "vae": {
    "method": "tabular_vae",
    "feature_differences": {
      "fixed acidity": {
        "test_mean": 6.775034013605442,
        "synthetic_mean": 6.8391670794487,
        "absolute_mean_difference": 0.06413306584325795,
        "test_std": 0.8074604298551417,
        "synthetic_std": 0.13961923326565298,
        "absolute_median_difference": 0.14209346771240217,
        "iqr_difference": 0.886322712898254
      },
      "volatile acidity": {
        "test_mean": 0.27482312925170066,
        "synthetic_mean": 0.2791874502301216,
        "absolute_mean_difference": 0.0043643209784209325,
        "test_std": 0.09522423951688104,
        "synthetic_std": 0.005136514660811657,
        "absolute_median_difference": 0.018438448905944815,
        "iqr_difference": 0.11423518031835556
      },
      "citric acid": {
        "test_mean": 0.3271156462585034,
        "synthetic_mean": 0.337535407692194,
        "absolute_mean_difference": 0.010419761433690555,
        "test_std": 0.11298236073311288,
        "synthetic_std": 0.02252380762685767,
        "absolute_median_difference": 0.022283914089202883,
        "iqr_difference": 0.08415001153945922
      },
      "residual sugar": {
        "test_mean": 6.242448979591837,
        "synthetic_mean": 6.372260841846466,
        "absolute_mean_difference": 0.1298118622546287,
        "test_std": 5.063269169496232,
        "synthetic_std": 2.720187598356821,
        "absolute_median_difference": 1.2274890422821043,
        "iqr_difference": 3.423896932601928
      },
      "chlorides": {
        "test_mean": 0.04540952380952381,
        "synthetic_mean": 0.04469380914233625,
        "absolute_mean_difference": 0.0007157146671875614,
        "test_std": 0.021792020424802546,
        "synthetic_std": 0.00637497851232636,
        "absolute_median_difference": 0.0021617909818887684,
        "iqr_difference": 0.004953905344009399
      },
      "free sulfur dioxide": {
        "test_mean": 33.76190476190476,
        "synthetic_mean": 34.38111012840271,
        "absolute_mean_difference": 0.6192053664979511,
        "test_std": 16.253302472723544,
        "synthetic_std": 6.439153560272506,
        "absolute_median_difference": 1.1564216613769531,
        "iqr_difference": 13.08882761001587
      },
      "total sulfur dioxide": {
        "test_mean": 134.2687074829932,
        "synthetic_mean": 137.0732196121216,
        "absolute_mean_difference": 2.8045121291283976,
        "test_std": 40.114287075568235,
        "synthetic_std": 21.514109057826126,
        "absolute_median_difference": 4.960731506347656,
        "iqr_difference": 17.925588607788086
      },
      "density": {
        "test_mean": 0.993851605442177,
        "synthetic_mean": 0.9939645262360572,
        "absolute_mean_difference": 0.00011292079388025567,
        "test_std": 0.0029466864373343797,
        "synthetic_std": 0.001930262774598565,
        "absolute_median_difference": 0.00011195669174191014,
        "iqr_difference": 0.0012261665916443176
      },
      "pH": {
        "test_mean": 3.1953605442176873,
        "synthetic_mean": 3.191366830825806,
        "absolute_mean_difference": 0.0039937133918814816,
        "test_std": 0.15403718720452117,
        "synthetic_std": 0.02254835533303863,
        "absolute_median_difference": 0.017374939918517907,
        "iqr_difference": 0.16658939123153704
      },
      "sulphates": {
        "test_mean": 0.48455782312925166,
        "synthetic_mean": 0.4900126293301582,
        "absolute_mean_difference": 0.005454806200906548,
        "test_std": 0.11038480185068494,
        "synthetic_std": 0.007903518978698374,
        "absolute_median_difference": 0.029489629864692668,
        "iqr_difference": 0.1280311191082001
      },
      "alcohol": {
        "test_mean": 10.551224489795917,
        "synthetic_mean": 10.578389053344727,
        "absolute_mean_difference": 0.02716456354881025,
        "test_std": 1.2455696169595525,
        "synthetic_std": 0.6858853704857802,
        "absolute_median_difference": 0.19584188461303675,
        "iqr_difference": 0.7758554458618168
      }
    },
    "correlation_l1": 0.6306522444352353,
    "duplicate_fraction": 0.0,
    "unique_fraction": 1.0,
    "nearest_neighbor_mean": 0.8535843462865071,
    "nearest_neighbor_min": 0.5602313901001218,
    "marginal_fidelity_score": 0.4071962507603276,
    "train_scale_normalized_marginal_fidelity": 0.7886970402928093,
    "dependence_preservation_score": 0.6132515399359983,
    "selection_score": 0.6509909572614292,
    "objective_metrics": {
      "data_generation_fidelity": 0.4071962507603276,
      "train_scale_normalized_marginal_fidelity": 0.7886970402928093,
      "dependence_preservation": 0.6132515399359983,
      "selection_score": 0.6509909572614292
    },
    "notes": [
      "Nearest-neighbor distances are memorization proxies, not formal privacy guarantees.",
      "Bootstrap baseline is evaluated with the same metrics for fairness.",
      "Selection score is 0.45 train-scale-normalized marginal fidelity, 0.35 dependence preservation when valid, and 0.20 legacy marginal fidelity."
    ]
  },
  "flow": {
    "method": "tabular_flow",
    "feature_differences": {
      "fixed acidity": {
        "test_mean": 6.775034013605442,
        "synthetic_mean": 6.904121752023697,
        "absolute_mean_difference": 0.12908773841825472,
        "test_std": 0.8074604298551417,
        "synthetic_std": 0.865726538354469,
        "absolute_median_difference": 0.2062469005584715,
        "iqr_difference": 0.07127239704132116
      },
      "volatile acidity": {
        "test_mean": 0.27482312925170066,
        "synthetic_mean": 0.2785039943009615,
        "absolute_mean_difference": 0.003680865049260851,
        "test_std": 0.09522423951688104,
        "synthetic_std": 0.11341570095444016,
        "absolute_median_difference": 0.00873527467250823,
        "iqr_difference": 0.010859702825546269
      },
      "citric acid": {
        "test_mean": 0.3271156462585034,
        "synthetic_mean": 0.3328371481001377,
        "absolute_mean_difference": 0.00572150184163428,
        "test_std": 0.11298236073311288,
        "synthetic_std": 0.13076350601262368,
        "absolute_median_difference": 0.016101914048194887,
        "iqr_difference": 0.027562246769666676
      },
      "residual sugar": {
        "test_mean": 6.242448979591837,
        "synthetic_mean": 6.322059390306473,
        "absolute_mean_difference": 0.07961041071463537,
        "test_std": 5.063269169496232,
        "synthetic_std": 5.032470042570976,
        "absolute_median_difference": 0.6077247142791746,
        "iqr_difference": 1.1035156488418574
      },
      "chlorides": {
        "test_mean": 0.04540952380952381,
        "synthetic_mean": 0.04599700831435621,
        "absolute_mean_difference": 0.0005874845048323973,
        "test_std": 0.021792020424802546,
        "synthetic_std": 0.017206669319638518,
        "absolute_median_difference": 0.0008705606609582875,
        "iqr_difference": 0.0008373415470123297
      },
      "free sulfur dioxide": {
        "test_mean": 33.76190476190476,
        "synthetic_mean": 36.57608735275269,
        "absolute_mean_difference": 2.8141825908479277,
        "test_std": 16.253302472723544,
        "synthetic_std": 16.306505043299982,
        "absolute_median_difference": 3.794435501098633,
        "iqr_difference": 0.30485057830810547
      },
      "total sulfur dioxide": {
        "test_mean": 134.2687074829932,
        "synthetic_mean": 142.07941766357422,
        "absolute_mean_difference": 7.810710180581026,
        "test_std": 40.114287075568235,
        "synthetic_std": 44.327232444912944,
        "absolute_median_difference": 10.100837707519531,
        "iqr_difference": 7.2111358642578125
      },
      "density": {
        "test_mean": 0.993851605442177,
        "synthetic_mean": 0.9942417777776719,
        "absolute_mean_difference": 0.00039017233549487607,
        "test_std": 0.0029466864373343797,
        "synthetic_std": 0.0029451923809432207,
        "absolute_median_difference": 0.0004325104713439609,
        "iqr_difference": 0.0001272760772704773
      },
      "pH": {
        "test_mean": 3.1953605442176873,
        "synthetic_mean": 3.1912814013957975,
        "absolute_mean_difference": 0.004079142821889814,
        "test_std": 0.15403718720452117,
        "synthetic_std": 0.14724563181595993,
        "absolute_median_difference": 0.004538125991821129,
        "iqr_difference": 0.009833109378814875
      },
      "sulphates": {
        "test_mean": 0.48455782312925166,
        "synthetic_mean": 0.4911717791855335,
        "absolute_mean_difference": 0.0066139560562818445,
        "test_std": 0.11038480185068494,
        "synthetic_std": 0.10937420129948179,
        "absolute_median_difference": 0.021784939765930156,
        "iqr_difference": 0.0010054117441177235
      },
      "alcohol": {
        "test_mean": 10.551224489795917,
        "synthetic_mean": 10.479966157913209,
        "absolute_mean_difference": 0.07125833188270825,
        "test_std": 1.2455696169595525,
        "synthetic_std": 1.1799509535539652,
        "absolute_median_difference": 0.084759616851807,
        "iqr_difference": 0.29206051826477086
      }
    },
    "correlation_l1": 0.0709900583756309,
    "duplicate_fraction": 0.0,
    "unique_fraction": 1.0,
    "nearest_neighbor_mean": 1.503583830513473,
    "nearest_neighbor_min": 0.5783849266343054,
    "marginal_fidelity_score": 0.486686540464344,
    "train_scale_normalized_marginal_fidelity": 0.9071353623848434,
    "dependence_preservation_score": 0.9337154833320289,
    "selection_score": 0.8323486403322584,
    "objective_metrics": {
      "data_generation_fidelity": 0.486686540464344,
      "train_scale_normalized_marginal_fidelity": 0.9071353623848434,
      "dependence_preservation": 0.9337154833320289,
      "selection_score": 0.8323486403322584
    },
    "notes": [
      "Nearest-neighbor distances are memorization proxies, not formal privacy guarantees.",
      "Bootstrap baseline is evaluated with the same metrics for fairness.",
      "Selection score is 0.45 train-scale-normalized marginal fidelity, 0.35 dependence preservation when valid, and 0.20 legacy marginal fidelity."
    ]
  },
  "gmm": {
    "method": "tabular_gmm",
    "feature_differences": {
      "fixed acidity": {
        "test_mean": 6.775034013605442,
        "synthetic_mean": 6.8791779787540435,
        "absolute_mean_difference": 0.10414396514860158,
        "test_std": 0.8074604298551417,
        "synthetic_std": 0.8486150221034294,
        "absolute_median_difference": 0.20014815330505353,
        "iqr_difference": 0.009281420707702992
      },
      "volatile acidity": {
        "test_mean": 0.27482312925170066,
        "synthetic_mean": 0.284090207785368,
        "absolute_mean_difference": 0.009267078533667317,
        "test_std": 0.09522423951688104,
        "synthetic_std": 0.1012438205819425,
        "absolute_median_difference": 0.02215324878692626,
        "iqr_difference": 0.010811620503664021
      },
      "citric acid": {
        "test_mean": 0.3271156462585034,
        "synthetic_mean": 0.326835524559021,
        "absolute_mean_difference": 0.0002801216994824385,
        "test_std": 0.11298236073311288,
        "synthetic_std": 0.12036393690214134,
        "absolute_median_difference": 0.019123303294181826,
        "iqr_difference": 0.038071339130401616
      },
      "residual sugar": {
        "test_mean": 6.242448979591837,
        "synthetic_mean": 6.208352150917054,
        "absolute_mean_difference": 0.03409682867478381,
        "test_std": 5.063269169496232,
        "synthetic_std": 4.930262180499987,
        "absolute_median_difference": 1.520515489578247,
        "iqr_difference": 1.1079560160636897
      },
      "chlorides": {
        "test_mean": 0.04540952380952381,
        "synthetic_mean": 0.04616498128697276,
        "absolute_mean_difference": 0.0007554574774489495,
        "test_std": 0.021792020424802546,
        "synthetic_std": 0.0217325340303859,
        "absolute_median_difference": 0.005003062441945073,
        "iqr_difference": 0.014196870978921652
      },
      "free sulfur dioxide": {
        "test_mean": 33.76190476190476,
        "synthetic_mean": 35.640093210220336,
        "absolute_mean_difference": 1.8781884483155764,
        "test_std": 16.253302472723544,
        "synthetic_std": 15.944159627687641,
        "absolute_median_difference": 3.3662986755371094,
        "iqr_difference": 0.33621788024902344
      },
      "total sulfur dioxide": {
        "test_mean": 134.2687074829932,
        "synthetic_mean": 139.7255446472168,
        "absolute_mean_difference": 5.4568371642236,
        "test_std": 40.114287075568235,
        "synthetic_std": 42.182369151263764,
        "absolute_median_difference": 10.110176086425781,
        "iqr_difference": 4.169914245605469
      },
      "density": {
        "test_mean": 0.993851605442177,
        "synthetic_mean": 0.9939151661992073,
        "absolute_mean_difference": 6.356075703028985e-05,
        "test_std": 0.0029466864373343797,
        "synthetic_std": 0.002876980250004715,
        "absolute_median_difference": 0.0002640081405639316,
        "iqr_difference": 0.0005367941713333435
      },
      "pH": {
        "test_mean": 3.1953605442176873,
        "synthetic_mean": 3.188261971950531,
        "absolute_mean_difference": 0.007098572267156378,
        "test_std": 0.15403718720452117,
        "synthetic_std": 0.15154863544557873,
        "absolute_median_difference": 0.004942960739135582,
        "iqr_difference": 0.0027110457420347345
      },
      "sulphates": {
        "test_mean": 0.48455782312925166,
        "synthetic_mean": 0.4965401469320059,
        "absolute_mean_difference": 0.011982323802754213,
        "test_std": 0.11038480185068494,
        "synthetic_std": 0.11503354201233888,
        "absolute_median_difference": 0.04133153796195982,
        "iqr_difference": 0.01838278084993361
      },
      "alcohol": {
        "test_mean": 10.551224489795917,
        "synthetic_mean": 10.570740224838257,
        "absolute_mean_difference": 0.019515735042340054,
        "test_std": 1.2455696169595525,
        "synthetic_std": 1.172420536133753,
        "absolute_median_difference": 0.1621709823608395,
        "iqr_difference": 0.27774372100830114
      }
    },
    "correlation_l1": 0.03963352665278926,
    "duplicate_fraction": 0.0,
    "unique_fraction": 1.0,
    "nearest_neighbor_mean": 1.6662710426749692,
    "nearest_neighbor_min": 0.6768717451833869,
    "marginal_fidelity_score": 0.5326001269517471,
    "train_scale_normalized_marginal_fidelity": 0.8696963484379435,
    "dependence_preservation_score": 0.9618774061852415,
    "selection_score": 0.8345404743522585,
    "objective_metrics": {
      "data_generation_fidelity": 0.5326001269517471,
      "train_scale_normalized_marginal_fidelity": 0.8696963484379435,
      "dependence_preservation": 0.9618774061852415,
      "selection_score": 0.8345404743522585
    },
    "notes": [
      "Nearest-neighbor distances are memorization proxies, not formal privacy guarantees.",
      "Bootstrap baseline is evaluated with the same metrics for fairness.",
      "Selection score is 0.45 train-scale-normalized marginal fidelity, 0.35 dependence preservation when valid, and 0.20 legacy marginal fidelity."
    ]
  },
  "copula": {
    "method": "tabular_copula",
    "feature_differences": {
      "fixed acidity": {
        "test_mean": 6.775034013605442,
        "synthetic_mean": 6.8355183095932,
        "absolute_mean_difference": 0.06048429598775851,
        "test_std": 0.8074604298551417,
        "synthetic_std": 0.8243864822802164,
        "absolute_median_difference": 0.0999997138977049,
        "iqr_difference": 0.09999999999999964
      },
      "volatile acidity": {
        "test_mean": 0.27482312925170066,
        "synthetic_mean": 0.2812060264199972,
        "absolute_mean_difference": 0.006382897168296542,
        "test_std": 0.09522423951688104,
        "synthetic_std": 0.10376909379513691,
        "absolute_median_difference": 0.01000001072883605,
        "iqr_difference": 1.0132789607375514e-08
      },
      "citric acid": {
        "test_mean": 0.3271156462585034,
        "synthetic_mean": 0.3409102866500616,
        "absolute_mean_difference": 0.0137946403915582,
        "test_std": 0.11298236073311288,
        "synthetic_std": 0.12361041029210439,
        "absolute_median_difference": 0.00999999284744263,
        "iqr_difference": 0.009999995231628422
      },
      "residual sugar": {
        "test_mean": 6.242448979591837,
        "synthetic_mean": 6.301701301813126,
        "absolute_mean_difference": 0.05925232222128862,
        "test_std": 5.063269169496232,
        "synthetic_std": 4.9553765026388605,
        "absolute_median_difference": 0.349999952316284,
        "iqr_difference": 0.5000005722045904
      },
      "chlorides": {
        "test_mean": 0.04540952380952381,
        "synthetic_mean": 0.04589368472620845,
        "absolute_mean_difference": 0.00048416091668464223,
        "test_std": 0.021792020424802546,
        "synthetic_std": 0.021547513911262032,
        "absolute_median_difference": 0.0019999997615814183,
        "iqr_difference": 0.0009999977052211756
      },
      "free sulfur dioxide": {
        "test_mean": 33.76190476190476,
        "synthetic_mean": 35.36664580726624,
        "absolute_mean_difference": 1.6047410453614788,
        "test_std": 16.253302472723544,
        "synthetic_std": 17.263298256701802,
        "absolute_median_difference": 1.0,
        "iqr_difference": 0.8782939910888672
      },
      "total sulfur dioxide": {
        "test_mean": 134.2687074829932,
        "synthetic_mean": 139.04921788787843,
        "absolute_mean_difference": 4.780510404885234,
        "test_std": 40.114287075568235,
        "synthetic_std": 43.91675985282025,
        "absolute_median_difference": 5.0,
        "iqr_difference": 12.5428466796875
      },
      "density": {
        "test_mean": 0.993851605442177,
        "synthetic_mean": 0.9939621842503548,
        "absolute_mean_difference": 0.00011057880817777477,
        "test_std": 0.0029466864373343797,
        "synthetic_std": 0.0029337623120155813,
        "absolute_median_difference": 1.3787841796841782e-05,
        "iqr_difference": 0.00023506055831912231
      },
      "pH": {
        "test_mean": 3.1953605442176873,
        "synthetic_mean": 3.1925662393569945,
        "absolute_mean_difference": 0.0027943048606928755,
        "test_std": 0.15403718720452117,
        "synthetic_std": 0.14644891680612643,
        "absolute_median_difference": 1.7166137711299712e-07,
        "iqr_difference": 0.009999942779541193
      },
      "sulphates": {
        "test_mean": 0.48455782312925166,
        "synthetic_mean": 0.4882811504006386,
        "absolute_mean_difference": 0.0037233272713869114,
        "test_std": 0.11038480185068494,
        "synthetic_std": 0.11724299842734326,
        "absolute_median_difference": 0.009999998807907085,
        "iqr_difference": 0.010000005960464464
      },
      "alcohol": {
        "test_mean": 10.551224489795917,
        "synthetic_mean": 10.521145946502685,
        "absolute_mean_difference": 0.03007854329323223,
        "test_std": 1.2455696169595525,
        "synthetic_std": 1.2409544803145203,
        "absolute_median_difference": 0.0500005722045902,
        "iqr_difference": 0.09999999999999964
      }
    },
    "correlation_l1": 0.044323927611692196,
    "duplicate_fraction": 0.0,
    "unique_fraction": 1.0,
    "nearest_neighbor_mean": 1.4729031586356987,
    "nearest_neighbor_min": 0.5935095811355048,
    "marginal_fidelity_score": 0.5477474163514093,
    "train_scale_normalized_marginal_fidelity": 0.9384342376044634,
    "dependence_preservation_score": 0.9575572995698198,
    "selection_score": 0.8669899450417273,
    "objective_metrics": {
      "data_generation_fidelity": 0.5477474163514093,
      "train_scale_normalized_marginal_fidelity": 0.9384342376044634,
      "dependence_preservation": 0.9575572995698198,
      "selection_score": 0.8669899450417273
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
    "max_models=4",
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
      "max_models=4",
      "max_epochs=5",
      "max_wall_minutes=30",
      "seed=2026"
    ],
    "supported_objective": true,
    "uses_generative_model": true
  },
  "selection": "Selected baseline using shared validation selection_score; held-out test was not used.",
  "validation_comparison": {
    "decision_rule": "Select the method with the highest validation selection_score from synthetic-vs-validation evaluation. VAE ELBO and flow NLL are used only within their own family searches.",
    "metric_direction": "higher selection_score is better",
    "metric_definition": "selection_score = 0.45 train-scale-normalized marginal fidelity + 0.35 dependence preservation + 0.20 legacy marginal fidelity when correlations are valid; without correlations, 0.75 normalized marginal + 0.25 legacy marginal.",
    "sample_count": 734,
    "seeds": {
      "baseline": 2033,
      "vae": 2037,
      "flow": 2039,
      "gmm": 2043,
      "copula": 2045
    },
    "scores": {
      "baseline": {
        "data_generation_fidelity": 0.7120288033647164,
        "train_scale_normalized_marginal_fidelity": 0.9600359232470271,
        "dependence_preservation": 0.966044913283156,
        "selection_score": 0.9125376457832101,
        "duplicate_fraction": 0.1539509536784741,
        "nearest_neighbor_mean": 0.0
      },
      "vae": {
        "data_generation_fidelity": 0.3521210266826008,
        "train_scale_normalized_marginal_fidelity": 0.7874512703865743,
        "dependence_preservation": 0.6105978812294912,
        "selection_score": 0.6384865354408005,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 0.8453001530245791
      },
      "flow": {
        "data_generation_fidelity": 0.6843117865424767,
        "train_scale_normalized_marginal_fidelity": 0.9319773145841236,
        "dependence_preservation": 0.9335511167232102,
        "selection_score": 0.8829950397244746,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 1.5071339571278755
      },
      "gmm": {
        "data_generation_fidelity": 0.6375065034413024,
        "train_scale_normalized_marginal_fidelity": 0.8697914914775958,
        "dependence_preservation": 0.9732418746064648,
        "selection_score": 0.8595421279654413,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 1.711595127072366
      },
      "copula": {
        "data_generation_fidelity": 0.6212939850587208,
        "train_scale_normalized_marginal_fidelity": 0.9241721119278072,
        "dependence_preservation": 0.9532556444491405,
        "selection_score": 0.8737757229364566,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 1.4699613483794136
      }
    },
    "eligibility": {
      "use_case": "internal_testing",
      "selected_method": "baseline",
      "eligible_methods": [
        "baseline",
        "vae",
        "flow",
        "gmm",
        "copula"
      ],
      "rejected_methods": [],
      "method_eligibility": [
        {
          "method": "baseline",
          "eligible": true,
          "reasons": [
            "Method is eligible for internal testing or stress-testing under current deterministic rules."
          ],
          "limitations": [
            "Eligibility does not imply high fidelity or privacy protection."
          ]
        },
        {
          "method": "vae",
          "eligible": true,
          "reasons": [
            "Method is eligible for internal testing or stress-testing under current deterministic rules."
          ],
          "limitations": [
            "Eligibility does not imply high fidelity or privacy protection."
          ]
        },
        {
          "method": "flow",
          "eligible": true,
          "reasons": [
            "Method is eligible for internal testing or stress-testing under current deterministic rules."
          ],
          "limitations": [
            "Eligibility does not imply high fidelity or privacy protection."
          ]
        },
        {
          "method": "gmm",
          "eligible": true,
          "reasons": [
            "Method is eligible for internal testing or stress-testing under current deterministic rules."
          ],
          "limitations": [
            "Eligibility does not imply high fidelity or privacy protection."
          ]
        },
        {
          "method": "copula",
          "eligible": true,
          "reasons": [
            "Method is eligible for internal testing or stress-testing under current deterministic rules."
          ],
          "limitations": [
            "Eligibility does not imply high fidelity or privacy protection."
          ]
        }
      ],
      "ranking_metric": "selection_score",
      "selection_reason": "Selected highest selection_score among eligible methods; hard eligibility rules were applied before ranking."
    },
    "objective_recommendation": {
      "use_case": "internal_testing",
      "selected_family": "baseline",
      "relative_ranking": [
        "baseline",
        "flow",
        "copula",
        "gmm",
        "vae"
      ],
      "selected_validation_score": 0.9125376457832101,
      "adequacy_warning_threshold": 0.85,
      "adequacy_claim": false,
      "recommendation": "consider_selected_with_limitations",
      "reasons": [
        "Selected method ranked highest among eligible validation candidates."
      ],
      "eligibility_summary": {
        "use_case": "internal_testing",
        "selected_method": "baseline",
        "eligible_methods": [
          "baseline",
          "vae",
          "flow",
          "gmm",
          "copula"
        ],
        "rejected_methods": [],
        "method_eligibility": [
          {
            "method": "baseline",
            "eligible": true,
            "reasons": [
              "Method is eligible for internal testing or stress-testing under current deterministic rules."
            ],
            "limitations": [
              "Eligibility does not imply high fidelity or privacy protection."
            ]
          },
          {
            "method": "vae",
            "eligible": true,
            "reasons": [
              "Method is eligible for internal testing or stress-testing under current deterministic rules."
            ],
            "limitations": [
              "Eligibility does not imply high fidelity or privacy protection."
            ]
          },
          {
            "method": "flow",
            "eligible": true,
            "reasons": [
              "Method is eligible for internal testing or stress-testing under current deterministic rules."
            ],
            "limitations": [
              "Eligibility does not imply high fidelity or privacy protection."
            ]
          },
          {
            "method": "gmm",
            "eligible": true,
            "reasons": [
              "Method is eligible for internal testing or stress-testing under current deterministic rules."
            ],
            "limitations": [
              "Eligibility does not imply high fidelity or privacy protection."
            ]
          },
          {
            "method": "copula",
            "eligible": true,
            "reasons": [
              "Method is eligible for internal testing or stress-testing under current deterministic rules."
            ],
            "limitations": [
              "Eligibility does not imply high fidelity or privacy protection."
            ]
          }
        ],
        "ranking_metric": "selection_score",
        "selection_reason": "Selected highest selection_score among eligible methods; hard eligibility rules were applied before ranking."
      },
      "language_guardrail": "Eligibility and high relative fidelity do not establish privacy, safety, clinical utility, or external-release suitability."
    }
  },
  "eligibility": {
    "use_case": "internal_testing",
    "selected_method": "baseline",
    "eligible_methods": [
      "baseline",
      "vae",
      "flow",
      "gmm",
      "copula"
    ],
    "rejected_methods": [],
    "method_eligibility": [
      {
        "method": "baseline",
        "eligible": true,
        "reasons": [
          "Method is eligible for internal testing or stress-testing under current deterministic rules."
        ],
        "limitations": [
          "Eligibility does not imply high fidelity or privacy protection."
        ]
      },
      {
        "method": "vae",
        "eligible": true,
        "reasons": [
          "Method is eligible for internal testing or stress-testing under current deterministic rules."
        ],
        "limitations": [
          "Eligibility does not imply high fidelity or privacy protection."
        ]
      },
      {
        "method": "flow",
        "eligible": true,
        "reasons": [
          "Method is eligible for internal testing or stress-testing under current deterministic rules."
        ],
        "limitations": [
          "Eligibility does not imply high fidelity or privacy protection."
        ]
      },
      {
        "method": "gmm",
        "eligible": true,
        "reasons": [
          "Method is eligible for internal testing or stress-testing under current deterministic rules."
        ],
        "limitations": [
          "Eligibility does not imply high fidelity or privacy protection."
        ]
      },
      {
        "method": "copula",
        "eligible": true,
        "reasons": [
          "Method is eligible for internal testing or stress-testing under current deterministic rules."
        ],
        "limitations": [
          "Eligibility does not imply high fidelity or privacy protection."
        ]
      }
    ],
    "ranking_metric": "selection_score",
    "selection_reason": "Selected highest selection_score among eligible methods; hard eligibility rules were applied before ranking."
  },
  "objective_recommendation": {
    "use_case": "internal_testing",
    "selected_family": "baseline",
    "relative_ranking": [
      "baseline",
      "flow",
      "copula",
      "gmm",
      "vae"
    ],
    "selected_validation_score": 0.9125376457832101,
    "adequacy_warning_threshold": 0.85,
    "adequacy_claim": false,
    "recommendation": "consider_selected_with_limitations",
    "reasons": [
      "Selected method ranked highest among eligible validation candidates."
    ],
    "eligibility_summary": {
      "use_case": "internal_testing",
      "selected_method": "baseline",
      "eligible_methods": [
        "baseline",
        "vae",
        "flow",
        "gmm",
        "copula"
      ],
      "rejected_methods": [],
      "method_eligibility": [
        {
          "method": "baseline",
          "eligible": true,
          "reasons": [
            "Method is eligible for internal testing or stress-testing under current deterministic rules."
          ],
          "limitations": [
            "Eligibility does not imply high fidelity or privacy protection."
          ]
        },
        {
          "method": "vae",
          "eligible": true,
          "reasons": [
            "Method is eligible for internal testing or stress-testing under current deterministic rules."
          ],
          "limitations": [
            "Eligibility does not imply high fidelity or privacy protection."
          ]
        },
        {
          "method": "flow",
          "eligible": true,
          "reasons": [
            "Method is eligible for internal testing or stress-testing under current deterministic rules."
          ],
          "limitations": [
            "Eligibility does not imply high fidelity or privacy protection."
          ]
        },
        {
          "method": "gmm",
          "eligible": true,
          "reasons": [
            "Method is eligible for internal testing or stress-testing under current deterministic rules."
          ],
          "limitations": [
            "Eligibility does not imply high fidelity or privacy protection."
          ]
        },
        {
          "method": "copula",
          "eligible": true,
          "reasons": [
            "Method is eligible for internal testing or stress-testing under current deterministic rules."
          ],
          "limitations": [
            "Eligibility does not imply high fidelity or privacy protection."
          ]
        }
      ],
      "ranking_metric": "selection_score",
      "selection_reason": "Selected highest selection_score among eligible methods; hard eligibility rules were applied before ranking."
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
