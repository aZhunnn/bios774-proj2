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
        "wall_seconds": 0.2347261249087751,
        "checkpoint_path": "outputs/phase6a_wine_white_classical/tabular_vae_candidate_0.pt",
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
        "wall_seconds": 0.3975411248393357,
        "checkpoint_path": "outputs/phase6a_wine_white_classical/tabular_flow_candidate_0.pt",
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
        "checkpoint_path": "outputs/phase6a_wine_white_classical/tabular_gmm_candidate_0.pkl",
        "wall_seconds": 0.02233445830643177,
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
      "checkpoint_path": "outputs/phase6a_wine_white_classical/tabular_gmm_candidate_0.pkl",
      "wall_seconds": 0.02233445830643177,
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
        "checkpoint_path": "outputs/phase6a_wine_white_classical/tabular_copula_candidate_0.pkl",
        "wall_seconds": 0.005098042078316212,
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
      "checkpoint_path": "outputs/phase6a_wine_white_classical/tabular_copula_candidate_0.pkl",
      "wall_seconds": 0.005098042078316212,
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
  }
}
```

## Generation
Generated samples: 4898
Method: training_row_bootstrap_selected_by_validation

## Evaluation
```json
{
  "selected": {
    "method": "training_row_bootstrap",
    "feature_differences": {
      "fixed acidity": {
        "test_mean": 6.775034013605442,
        "synthetic_mean": 6.863403429971417,
        "absolute_mean_difference": 0.08836941636597473,
        "test_std": 0.8074604298551417,
        "synthetic_std": 0.8518549841435954,
        "absolute_median_difference": 0.09999999999999964,
        "iqr_difference": 0.09999999999999964
      },
      "volatile acidity": {
        "test_mean": 0.27482312925170066,
        "synthetic_mean": 0.28158840342997143,
        "absolute_mean_difference": 0.006765274178270775,
        "test_std": 0.09522423951688104,
        "synthetic_std": 0.10526424800077763,
        "absolute_median_difference": 0.0,
        "iqr_difference": 2.7755575615628914e-17
      },
      "citric acid": {
        "test_mean": 0.3271156462585034,
        "synthetic_mean": 0.33542057982850143,
        "absolute_mean_difference": 0.008304933569998008,
        "test_std": 0.11298236073311288,
        "synthetic_std": 0.1268968641430882,
        "absolute_median_difference": 0.010000000000000009,
        "iqr_difference": 0.0
      },
      "residual sugar": {
        "test_mean": 6.242448979591837,
        "synthetic_mean": 6.365516537362189,
        "absolute_mean_difference": 0.1230675577703515,
        "test_std": 5.063269169496232,
        "synthetic_std": 4.950591955216673,
        "absolute_median_difference": 0.5,
        "iqr_difference": 0.20000000000000195
      },
      "chlorides": {
        "test_mean": 0.04540952380952381,
        "synthetic_mean": 0.04552735810534912,
        "absolute_mean_difference": 0.00011783429582531285,
        "test_std": 0.021792020424802546,
        "synthetic_std": 0.022358053834276526,
        "absolute_median_difference": 0.000999999999999994,
        "iqr_difference": 0.0
      },
      "free sulfur dioxide": {
        "test_mean": 33.76190476190476,
        "synthetic_mean": 35.06921192323397,
        "absolute_mean_difference": 1.3073071613292129,
        "test_std": 16.253302472723544,
        "synthetic_std": 16.716742408658956,
        "absolute_median_difference": 2.0,
        "iqr_difference": 0.0
      },
      "total sulfur dioxide": {
        "test_mean": 134.2687074829932,
        "synthetic_mean": 137.3523887300939,
        "absolute_mean_difference": 3.0836812471007136,
        "test_std": 40.114287075568235,
        "synthetic_std": 42.89096176674534,
        "absolute_median_difference": 2.0,
        "iqr_difference": 8.5
      },
      "density": {
        "test_mean": 0.993851605442177,
        "synthetic_mean": 0.9940069732543895,
        "absolute_mean_difference": 0.0001553678122124813,
        "test_std": 0.0029466864373343797,
        "synthetic_std": 0.0028989655317450537,
        "absolute_median_difference": 4.999999999999449e-05,
        "iqr_difference": 0.000160000000000049
      },
      "pH": {
        "test_mean": 3.1953605442176873,
        "synthetic_mean": 3.1847121273989387,
        "absolute_mean_difference": 0.010648416818748618,
        "test_std": 0.15403718720452117,
        "synthetic_std": 0.14801479577115562,
        "absolute_median_difference": 0.0,
        "iqr_difference": 0.010000000000000231
      },
      "sulphates": {
        "test_mean": 0.48455782312925166,
        "synthetic_mean": 0.4879889750918742,
        "absolute_mean_difference": 0.0034311519626225584,
        "test_std": 0.11038480185068494,
        "synthetic_std": 0.11529241978089118,
        "absolute_median_difference": 0.019999999999999962,
        "iqr_difference": 0.009999999999999953
      },
      "alcohol": {
        "test_mean": 10.551224489795917,
        "synthetic_mean": 10.50432829726419,
        "absolute_mean_difference": 0.04689619253172772,
        "test_std": 1.2455696169595525,
        "synthetic_std": 1.228759689042184,
        "absolute_median_difference": 0.09999999999999964,
        "iqr_difference": 0.09999999999999964
      }
    },
    "correlation_l1": 0.030337311996453203,
    "duplicate_fraction": 0.5261331155573703,
    "unique_fraction": 0.4738668844426297,
    "nearest_neighbor_mean": 0.0,
    "nearest_neighbor_min": 0.0,
    "marginal_fidelity_score": 0.642899458744959,
    "train_scale_normalized_marginal_fidelity": 0.9463459694461286,
    "dependence_preservation_score": 0.9705559415899736,
    "selection_score": 0.8941301575562404,
    "objective_metrics": {
      "data_generation_fidelity": 0.642899458744959,
      "train_scale_normalized_marginal_fidelity": 0.9463459694461286,
      "dependence_preservation": 0.9705559415899736,
      "selection_score": 0.8941301575562404
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
        "synthetic_mean": 6.863403429971417,
        "absolute_mean_difference": 0.08836941636597473,
        "test_std": 0.8074604298551417,
        "synthetic_std": 0.8518549841435954,
        "absolute_median_difference": 0.09999999999999964,
        "iqr_difference": 0.09999999999999964
      },
      "volatile acidity": {
        "test_mean": 0.27482312925170066,
        "synthetic_mean": 0.28158840342997143,
        "absolute_mean_difference": 0.006765274178270775,
        "test_std": 0.09522423951688104,
        "synthetic_std": 0.10526424800077763,
        "absolute_median_difference": 0.0,
        "iqr_difference": 2.7755575615628914e-17
      },
      "citric acid": {
        "test_mean": 0.3271156462585034,
        "synthetic_mean": 0.33542057982850143,
        "absolute_mean_difference": 0.008304933569998008,
        "test_std": 0.11298236073311288,
        "synthetic_std": 0.1268968641430882,
        "absolute_median_difference": 0.010000000000000009,
        "iqr_difference": 0.0
      },
      "residual sugar": {
        "test_mean": 6.242448979591837,
        "synthetic_mean": 6.365516537362189,
        "absolute_mean_difference": 0.1230675577703515,
        "test_std": 5.063269169496232,
        "synthetic_std": 4.950591955216673,
        "absolute_median_difference": 0.5,
        "iqr_difference": 0.20000000000000195
      },
      "chlorides": {
        "test_mean": 0.04540952380952381,
        "synthetic_mean": 0.04552735810534912,
        "absolute_mean_difference": 0.00011783429582531285,
        "test_std": 0.021792020424802546,
        "synthetic_std": 0.022358053834276526,
        "absolute_median_difference": 0.000999999999999994,
        "iqr_difference": 0.0
      },
      "free sulfur dioxide": {
        "test_mean": 33.76190476190476,
        "synthetic_mean": 35.06921192323397,
        "absolute_mean_difference": 1.3073071613292129,
        "test_std": 16.253302472723544,
        "synthetic_std": 16.716742408658956,
        "absolute_median_difference": 2.0,
        "iqr_difference": 0.0
      },
      "total sulfur dioxide": {
        "test_mean": 134.2687074829932,
        "synthetic_mean": 137.3523887300939,
        "absolute_mean_difference": 3.0836812471007136,
        "test_std": 40.114287075568235,
        "synthetic_std": 42.89096176674534,
        "absolute_median_difference": 2.0,
        "iqr_difference": 8.5
      },
      "density": {
        "test_mean": 0.993851605442177,
        "synthetic_mean": 0.9940069732543895,
        "absolute_mean_difference": 0.0001553678122124813,
        "test_std": 0.0029466864373343797,
        "synthetic_std": 0.0028989655317450537,
        "absolute_median_difference": 4.999999999999449e-05,
        "iqr_difference": 0.000160000000000049
      },
      "pH": {
        "test_mean": 3.1953605442176873,
        "synthetic_mean": 3.1847121273989387,
        "absolute_mean_difference": 0.010648416818748618,
        "test_std": 0.15403718720452117,
        "synthetic_std": 0.14801479577115562,
        "absolute_median_difference": 0.0,
        "iqr_difference": 0.010000000000000231
      },
      "sulphates": {
        "test_mean": 0.48455782312925166,
        "synthetic_mean": 0.4879889750918742,
        "absolute_mean_difference": 0.0034311519626225584,
        "test_std": 0.11038480185068494,
        "synthetic_std": 0.11529241978089118,
        "absolute_median_difference": 0.019999999999999962,
        "iqr_difference": 0.009999999999999953
      },
      "alcohol": {
        "test_mean": 10.551224489795917,
        "synthetic_mean": 10.50432829726419,
        "absolute_mean_difference": 0.04689619253172772,
        "test_std": 1.2455696169595525,
        "synthetic_std": 1.228759689042184,
        "absolute_median_difference": 0.09999999999999964,
        "iqr_difference": 0.09999999999999964
      }
    },
    "correlation_l1": 0.030337311996453203,
    "duplicate_fraction": 0.5261331155573703,
    "unique_fraction": 0.4738668844426297,
    "nearest_neighbor_mean": 0.0,
    "nearest_neighbor_min": 0.0,
    "marginal_fidelity_score": 0.642899458744959,
    "train_scale_normalized_marginal_fidelity": 0.9463459694461286,
    "dependence_preservation_score": 0.9705559415899736,
    "selection_score": 0.8941301575562404,
    "objective_metrics": {
      "data_generation_fidelity": 0.642899458744959,
      "train_scale_normalized_marginal_fidelity": 0.9463459694461286,
      "dependence_preservation": 0.9705559415899736,
      "selection_score": 0.8941301575562404
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
        "synthetic_mean": 6.837841855014475,
        "absolute_mean_difference": 0.0628078414090334,
        "test_std": 0.8074604298551417,
        "synthetic_std": 0.14213720634443971,
        "absolute_median_difference": 0.14310269355773908,
        "iqr_difference": 0.8857965230941769
      },
      "volatile acidity": {
        "test_mean": 0.27482312925170066,
        "synthetic_mean": 0.2790712210999649,
        "absolute_mean_difference": 0.004248091848264246,
        "test_std": 0.09522423951688104,
        "synthetic_std": 0.00525675883121977,
        "absolute_median_difference": 0.01825184166431426,
        "iqr_difference": 0.11407176673412323
      },
      "citric acid": {
        "test_mean": 0.3271156462585034,
        "synthetic_mean": 0.3373536885015913,
        "absolute_mean_difference": 0.010238042243087886,
        "test_std": 0.11298236073311288,
        "synthetic_std": 0.022750936782357265,
        "absolute_median_difference": 0.022157000899314883,
        "iqr_difference": 0.08393763273954391
      },
      "residual sugar": {
        "test_mean": 6.242448979591837,
        "synthetic_mean": 6.359479707715559,
        "absolute_mean_difference": 0.11703072812372195,
        "test_std": 5.063269169496232,
        "synthetic_std": 2.742568903189855,
        "absolute_median_difference": 1.2238786697387694,
        "iqr_difference": 3.462938213348388
      },
      "chlorides": {
        "test_mean": 0.04540952380952381,
        "synthetic_mean": 0.04467797432081122,
        "absolute_mean_difference": 0.0007315494887125892,
        "test_std": 0.021792020424802546,
        "synthetic_std": 0.0064263282224528324,
        "absolute_median_difference": 0.0021262256354093526,
        "iqr_difference": 0.005025999955832958
      },
      "free sulfur dioxide": {
        "test_mean": 33.76190476190476,
        "synthetic_mean": 34.34667022240897,
        "absolute_mean_difference": 0.5847654605042081,
        "test_std": 16.253302472723544,
        "synthetic_std": 6.51724623736748,
        "absolute_median_difference": 1.0829715728759766,
        "iqr_difference": 13.254451274871826
      },
      "total sulfur dioxide": {
        "test_mean": 134.2687074829932,
        "synthetic_mean": 137.01871192966397,
        "absolute_mean_difference": 2.750004446670772,
        "test_std": 40.114287075568235,
        "synthetic_std": 21.673801387113546,
        "absolute_median_difference": 4.76220703125,
        "iqr_difference": 18.44341468811035
      },
      "density": {
        "test_mean": 0.993851605442177,
        "synthetic_mean": 0.9939591215794892,
        "absolute_mean_difference": 0.00010751613731219134,
        "test_std": 0.0029466864373343797,
        "synthetic_std": 0.0019445763121480665,
        "absolute_median_difference": 0.00010498294830318944,
        "iqr_difference": 0.0012628383493423767
      },
      "pH": {
        "test_mean": 3.1953605442176873,
        "synthetic_mean": 3.1916242997078763,
        "absolute_mean_difference": 0.0037362445098110264,
        "test_std": 0.15403718720452117,
        "synthetic_std": 0.022707742013197287,
        "absolute_median_difference": 0.017779178619384606,
        "iqr_difference": 0.16775847673416155
      },
      "sulphates": {
        "test_mean": 0.48455782312925166,
        "synthetic_mean": 0.4900989883064786,
        "absolute_mean_difference": 0.005541165177226914,
        "test_std": 0.11038480185068494,
        "synthetic_std": 0.00790652007237125,
        "absolute_median_difference": 0.029641070365905742,
        "iqr_difference": 0.12867502808570863
      },
      "alcohol": {
        "test_mean": 10.551224489795917,
        "synthetic_mean": 10.579934480580665,
        "absolute_mean_difference": 0.028709990784747674,
        "test_std": 1.2455696169595525,
        "synthetic_std": 0.6928781745876788,
        "absolute_median_difference": 0.21057758331298793,
        "iqr_difference": 0.7962261676788334
      }
    },
    "correlation_l1": 0.6308494388847256,
    "duplicate_fraction": 0.0,
    "unique_fraction": 1.0,
    "nearest_neighbor_mean": 0.8549224022509273,
    "nearest_neighbor_min": 0.5364035971367365,
    "marginal_fidelity_score": 0.40528452935692416,
    "train_scale_normalized_marginal_fidelity": 0.7879112889916646,
    "dependence_preservation_score": 0.6131773885171529,
    "selection_score": 0.6502290718986374,
    "objective_metrics": {
      "data_generation_fidelity": 0.40528452935692416,
      "train_scale_normalized_marginal_fidelity": 0.7879112889916646,
      "dependence_preservation": 0.6131773885171529,
      "selection_score": 0.6502290718986374
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
        "synthetic_mean": 6.891864492631338,
        "absolute_mean_difference": 0.1168304790258965,
        "test_std": 0.8074604298551417,
        "synthetic_std": 0.8440275604871224,
        "absolute_median_difference": 0.19834880828857404,
        "iqr_difference": 0.027043366432190297
      },
      "volatile acidity": {
        "test_mean": 0.27482312925170066,
        "synthetic_mean": 0.27751561598954955,
        "absolute_mean_difference": 0.002692486737848887,
        "test_std": 0.09522423951688104,
        "synthetic_std": 0.10896144833000564,
        "absolute_median_difference": 0.0069333070516586215,
        "iqr_difference": 0.006025423407554631
      },
      "citric acid": {
        "test_mean": 0.3271156462585034,
        "synthetic_mean": 0.33004879134012366,
        "absolute_mean_difference": 0.0029331450816202365,
        "test_std": 0.11298236073311288,
        "synthetic_std": 0.12730291064938262,
        "absolute_median_difference": 0.014058711528778078,
        "iqr_difference": 0.0205944973230362
      },
      "residual sugar": {
        "test_mean": 6.242448979591837,
        "synthetic_mean": 6.221969321680439,
        "absolute_mean_difference": 0.020479657911398164,
        "test_std": 5.063269169496232,
        "synthetic_std": 5.020806773130604,
        "absolute_median_difference": 0.5092072963714598,
        "iqr_difference": 1.1320286035537714
      },
      "chlorides": {
        "test_mean": 0.04540952380952381,
        "synthetic_mean": 0.04526514468411126,
        "absolute_mean_difference": 0.00014437912541254844,
        "test_std": 0.021792020424802546,
        "synthetic_std": 0.016205646096082375,
        "absolute_median_difference": 0.0008380891680717442,
        "iqr_difference": 0.0013481282070279127
      },
      "free sulfur dioxide": {
        "test_mean": 33.76190476190476,
        "synthetic_mean": 35.79340451218343,
        "absolute_mean_difference": 2.0314997502786696,
        "test_std": 16.253302472723544,
        "synthetic_std": 16.415129734452382,
        "absolute_median_difference": 2.650308609008789,
        "iqr_difference": 1.1890339851379395
      },
      "total sulfur dioxide": {
        "test_mean": 134.2687074829932,
        "synthetic_mean": 140.20895131688647,
        "absolute_mean_difference": 5.940243833893277,
        "test_std": 40.114287075568235,
        "synthetic_std": 45.908719198820506,
        "absolute_median_difference": 6.9843597412109375,
        "iqr_difference": 10.204418182373047
      },
      "density": {
        "test_mean": 0.993851605442177,
        "synthetic_mean": 0.9941642423000273,
        "absolute_mean_difference": 0.0003126368578503236,
        "test_std": 0.0029466864373343797,
        "synthetic_std": 0.003016593859489327,
        "absolute_median_difference": 0.00037713775634762303,
        "iqr_difference": 0.0001632325792312317
      },
      "pH": {
        "test_mean": 3.1953605442176873,
        "synthetic_mean": 3.190860709495669,
        "absolute_mean_difference": 0.004499834722018203,
        "test_std": 0.15403718720452117,
        "synthetic_std": 0.14648527786627738,
        "absolute_median_difference": 0.011739201545715172,
        "iqr_difference": 0.00999016761779803
      },
      "sulphates": {
        "test_mean": 0.48455782312925166,
        "synthetic_mean": 0.49064245432152853,
        "absolute_mean_difference": 0.006084631192276868,
        "test_std": 0.11038480185068494,
        "synthetic_std": 0.1130081503880902,
        "absolute_median_difference": 0.02276449739933012,
        "iqr_difference": 0.0012279009819030628
      },
      "alcohol": {
        "test_mean": 10.551224489795917,
        "synthetic_mean": 10.50208594819486,
        "absolute_mean_difference": 0.04913854160105657,
        "test_std": 1.2455696169595525,
        "synthetic_std": 1.2390993585161156,
        "absolute_median_difference": 0.0756510734558109,
        "iqr_difference": 0.14325728416442907
      }
    },
    "correlation_l1": 0.0652919300086447,
    "duplicate_fraction": 0.0,
    "unique_fraction": 1.0,
    "nearest_neighbor_mean": 1.4960386638486673,
    "nearest_neighbor_min": 0.5673650481739669,
    "marginal_fidelity_score": 0.5125451114520695,
    "train_scale_normalized_marginal_fidelity": 0.9189371602416226,
    "dependence_preservation_score": 0.9387098238807507,
    "selection_score": 0.8445791827574068,
    "objective_metrics": {
      "data_generation_fidelity": 0.5125451114520695,
      "train_scale_normalized_marginal_fidelity": 0.9189371602416226,
      "dependence_preservation": 0.9387098238807507,
      "selection_score": 0.8445791827574068
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
        "synthetic_mean": 6.845315684781555,
        "absolute_mean_difference": 0.07028167117611339,
        "test_std": 0.8074604298551417,
        "synthetic_std": 0.8427462329948442,
        "absolute_median_difference": 0.1567631244659422,
        "iqr_difference": 0.045298838615417836
      },
      "volatile acidity": {
        "test_mean": 0.27482312925170066,
        "synthetic_mean": 0.2799308824327441,
        "absolute_mean_difference": 0.005107753181043451,
        "test_std": 0.09522423951688104,
        "synthetic_std": 0.10208131678615986,
        "absolute_median_difference": 0.01880862355232238,
        "iqr_difference": 0.016168703436851506
      },
      "citric acid": {
        "test_mean": 0.3271156462585034,
        "synthetic_mean": 0.3335094526957716,
        "absolute_mean_difference": 0.006393806437268168,
        "test_std": 0.11298236073311288,
        "synthetic_std": 0.12341756113659648,
        "absolute_median_difference": 0.022970961928367617,
        "iqr_difference": 0.047504757642745976
      },
      "residual sugar": {
        "test_mean": 6.242448979591837,
        "synthetic_mean": 6.453654754595058,
        "absolute_mean_difference": 0.21120577500322035,
        "test_std": 5.063269169496232,
        "synthetic_std": 5.087095887860263,
        "absolute_median_difference": 1.7568681716918944,
        "iqr_difference": 1.0293433308601374
      },
      "chlorides": {
        "test_mean": 0.04540952380952381,
        "synthetic_mean": 0.04628965700111398,
        "absolute_mean_difference": 0.0008801331915901725,
        "test_std": 0.021792020424802546,
        "synthetic_std": 0.02158160488464629,
        "absolute_median_difference": 0.004117283403873441,
        "iqr_difference": 0.014583480209112168
      },
      "free sulfur dioxide": {
        "test_mean": 33.76190476190476,
        "synthetic_mean": 35.55301299393055,
        "absolute_mean_difference": 1.7911082320257918,
        "test_std": 16.253302472723544,
        "synthetic_std": 16.448473772200654,
        "absolute_median_difference": 3.4875240325927734,
        "iqr_difference": 0.5453386306762695
      },
      "total sulfur dioxide": {
        "test_mean": 134.2687074829932,
        "synthetic_mean": 138.91533095304115,
        "absolute_mean_difference": 4.646623470047956,
        "test_std": 40.114287075568235,
        "synthetic_std": 42.39459856055504,
        "absolute_median_difference": 9.005020141601562,
        "iqr_difference": 4.265651702880859
      },
      "density": {
        "test_mean": 0.993851605442177,
        "synthetic_mean": 0.9940428705958261,
        "absolute_mean_difference": 0.00019126515364908947,
        "test_std": 0.0029466864373343797,
        "synthetic_std": 0.002961807231476841,
        "absolute_median_difference": 0.0004001153469085361,
        "iqr_difference": 0.00035555134773257446
      },
      "pH": {
        "test_mean": 3.1953605442176873,
        "synthetic_mean": 3.1882744065690107,
        "absolute_mean_difference": 0.007086137648676605,
        "test_std": 0.15403718720452117,
        "synthetic_std": 0.14978975408410652,
        "absolute_median_difference": 0.009121246337890465,
        "iqr_difference": 0.0017757534980775702
      },
      "sulphates": {
        "test_mean": 0.48455782312925166,
        "synthetic_mean": 0.48859423591548834,
        "absolute_mean_difference": 0.004036412786236676,
        "test_std": 0.11038480185068494,
        "synthetic_std": 0.1145754702420216,
        "absolute_median_difference": 0.030242034196853618,
        "iqr_difference": 0.013030961751937853
      },
      "alcohol": {
        "test_mean": 10.551224489795917,
        "synthetic_mean": 10.511326918849267,
        "absolute_mean_difference": 0.03989757094664981,
        "test_std": 1.2455696169595525,
        "synthetic_std": 1.2032213276747494,
        "absolute_median_difference": 0.12256536483764613,
        "iqr_difference": 0.28014435768127477
      }
    },
    "correlation_l1": 0.03551078310758879,
    "duplicate_fraction": 0.0,
    "unique_fraction": 1.0,
    "nearest_neighbor_mean": 1.6679023481597903,
    "nearest_neighbor_min": 0.5475906539553922,
    "marginal_fidelity_score": 0.5440480293594836,
    "train_scale_normalized_marginal_fidelity": 0.8706444069167723,
    "dependence_preservation_score": 0.9657069885829482,
    "selection_score": 0.838597034988476,
    "objective_metrics": {
      "data_generation_fidelity": 0.5440480293594836,
      "train_scale_normalized_marginal_fidelity": 0.8706444069167723,
      "dependence_preservation": 0.9657069885829482,
      "selection_score": 0.838597034988476
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
        "synthetic_mean": 6.857531083073115,
        "absolute_mean_difference": 0.08249706946767343,
        "test_std": 0.8074604298551417,
        "synthetic_std": 0.824720089958942,
        "absolute_median_difference": 0.0999997138977049,
        "iqr_difference": 0.09999999999999964
      },
      "volatile acidity": {
        "test_mean": 0.27482312925170066,
        "synthetic_mean": 0.28016797395665677,
        "absolute_mean_difference": 0.005344844704956109,
        "test_std": 0.09522423951688104,
        "synthetic_std": 0.10316396954507558,
        "absolute_median_difference": 9.536743172944284e-09,
        "iqr_difference": 1.0132789607375514e-08
      },
      "citric acid": {
        "test_mean": 0.3271156462585034,
        "synthetic_mean": 0.3381319893512642,
        "absolute_mean_difference": 0.011016343092760783,
        "test_std": 0.11298236073311288,
        "synthetic_std": 0.12388237800594912,
        "absolute_median_difference": 0.00999999284744263,
        "iqr_difference": 4.768371586472142e-09
      },
      "residual sugar": {
        "test_mean": 6.242448979591837,
        "synthetic_mean": 6.307777755890345,
        "absolute_mean_difference": 0.06532877629850731,
        "test_std": 5.063269169496232,
        "synthetic_std": 4.9867180940970295,
        "absolute_median_difference": 0.3999999046325682,
        "iqr_difference": 0.10000000000000053
      },
      "chlorides": {
        "test_mean": 0.04540952380952381,
        "synthetic_mean": 0.04575901340256666,
        "absolute_mean_difference": 0.0003494895930428524,
        "test_std": 0.021792020424802546,
        "synthetic_std": 0.02173121804292642,
        "absolute_median_difference": 0.0010000014603137944,
        "iqr_difference": 0.0009999977052211756
      },
      "free sulfur dioxide": {
        "test_mean": 33.76190476190476,
        "synthetic_mean": 35.39282435794517,
        "absolute_mean_difference": 1.630919596040414,
        "test_std": 16.253302472723544,
        "synthetic_std": 16.6323214214694,
        "absolute_median_difference": 2.0,
        "iqr_difference": 0.9999980926513672
      },
      "total sulfur dioxide": {
        "test_mean": 134.2687074829932,
        "synthetic_mean": 138.70664905284852,
        "absolute_mean_difference": 4.437941569855326,
        "test_std": 40.114287075568235,
        "synthetic_std": 42.36542315723191,
        "absolute_median_difference": 4.0,
        "iqr_difference": 7.498775482177734
      },
      "density": {
        "test_mean": 0.993851605442177,
        "synthetic_mean": 0.9940124899818246,
        "absolute_mean_difference": 0.00016088453964757665,
        "test_std": 0.0029466864373343797,
        "synthetic_std": 0.0028932290528005922,
        "absolute_median_difference": 5.002746582027928e-05,
        "iqr_difference": 0.00016001831054690552
      },
      "pH": {
        "test_mean": 3.1953605442176873,
        "synthetic_mean": 3.1915171807422693,
        "absolute_mean_difference": 0.003843363475418027,
        "test_std": 0.15403718720452117,
        "synthetic_std": 0.1507884721042769,
        "absolute_median_difference": 1.7166137711299712e-07,
        "iqr_difference": 0.009999942779541193
      },
      "sulphates": {
        "test_mean": 0.48455782312925166,
        "synthetic_mean": 0.4919271468751136,
        "absolute_mean_difference": 0.007369323745861911,
        "test_std": 0.11038480185068494,
        "synthetic_std": 0.11505536612813631,
        "absolute_median_difference": 0.020000019073486308,
        "iqr_difference": 1.4305114759416426e-08
      },
      "alcohol": {
        "test_mean": 10.551224489795917,
        "synthetic_mean": 10.511371382015877,
        "absolute_mean_difference": 0.03985310778003992,
        "test_std": 1.2455696169595525,
        "synthetic_std": 1.2247065210825345,
        "absolute_median_difference": 3.8146972691777137e-07,
        "iqr_difference": 0.09999999999999964
      }
    },
    "correlation_l1": 0.044042743440390525,
    "duplicate_fraction": 0.0,
    "unique_fraction": 1.0,
    "nearest_neighbor_mean": 1.4769551331861082,
    "nearest_neighbor_min": 0.4904475860206123,
    "marginal_fidelity_score": 0.6041122674055578,
    "train_scale_normalized_marginal_fidelity": 0.9463727955081825,
    "dependence_preservation_score": 0.9578151912676887,
    "selection_score": 0.8819255284034847,
    "objective_metrics": {
      "data_generation_fidelity": 0.6041122674055578,
      "train_scale_normalized_marginal_fidelity": 0.9463727955081825,
      "dependence_preservation": 0.9578151912676887,
      "selection_score": 0.8819255284034847
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
