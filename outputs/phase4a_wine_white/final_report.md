# Phase 2.5 Tabular VAE Report

## Scope
This report is generated from stored workflow results for unconditional tabular synthetic data generation.

## Modeling Assumptions
- Continuous features use a unit-variance Gaussian observation model after training-only standardization.
- Binary features use a Bernoulli likelihood parameterized by decoder logits.
- Within-family selection uses validation loss or validation likelihood only inside that family.
- max_models is total trained generative candidates across all families; bootstrap baseline does not consume a training slot.
- Cross-family selection uses the predeclared validation selection_score; held-out test data is used only for final evaluation.
- RealNVP flow is available only for continuous tabular features with no missing values.
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
          "epochs": 20,
          "kl_weight": 1.0,
          "early_stopping_patience": 5,
          "seed": 2026,
          "device": "cpu"
        },
        "validation_loss": 15.17319107055664,
        "epochs_ran": 19,
        "wall_seconds": 0.913355415686965,
        "checkpoint_path": "outputs/phase4a_wine_white/tabular_vae_candidate_0.pt",
        "failed": false,
        "failure_reason": null,
        "history": {
          "train_loss": [
            15.650201347139147,
            15.525706891660336,
            15.234358840518528,
            15.169126916814733,
            15.102156762723569,
            15.07972005561546,
            15.090545936867043,
            15.076345779277661,
            15.063977894959626,
            15.071920933546844,
            15.084191507763332,
            15.080587404745597,
            15.025966820893464,
            15.040613615954364,
            15.033105823728773,
            15.053899164553043,
            15.05758183090775,
            15.043216855437667,
            15.073881776244551
          ],
          "validation_loss": [
            15.773663520812988,
            15.548996925354004,
            15.310150146484375,
            15.23876953125,
            15.249807357788086,
            15.227417945861816,
            15.21489143371582,
            15.239546775817871,
            15.250815391540527,
            15.186378479003906,
            15.226024627685547,
            15.204649925231934,
            15.226320266723633,
            15.17319107055664,
            15.193154335021973,
            15.23129940032959,
            15.238253593444824,
            15.188590049743652,
            15.192652702331543
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
          "epochs": 20,
          "early_stopping_patience": 5,
          "seed": 2127,
          "device": "cpu"
        },
        "validation_nll": 10.96346664428711,
        "epochs_ran": 20,
        "wall_seconds": 1.7009117919951677,
        "checkpoint_path": "outputs/phase4a_wine_white/tabular_flow_candidate_0.pt",
        "failed": false,
        "failure_reason": null,
        "history": {
          "train_nll": [
            15.124412713227448,
            13.796849065356785,
            12.754572100109524,
            12.102241613246777,
            11.79441112059134,
            11.566847757056907,
            11.422965014422381,
            11.306659998717132,
            11.241714239120483,
            11.131483722616125,
            11.076724149562695,
            10.976132286919487,
            10.910592008520055,
            10.893827579639575,
            10.831578969955444,
            10.814164152851811,
            10.770016679057369,
            10.70028089593958,
            10.690314407701846,
            10.678472810321384
          ],
          "validation_nll": [
            14.3749418258667,
            13.26424789428711,
            12.397820472717285,
            12.058106422424316,
            11.842278480529785,
            11.646148681640625,
            11.54341983795166,
            11.567819595336914,
            11.443259239196777,
            11.322436332702637,
            11.234894752502441,
            11.212071418762207,
            11.146615982055664,
            11.07730770111084,
            11.0531005859375,
            11.088271141052246,
            10.986820220947266,
            10.96346664428711,
            10.976367950439453,
            10.984213829040527
          ]
        }
      }
    ],
    "selected_index": 0,
    "selection_reason": "Selected lowest validation negative log likelihood within flow family; test data was not used."
  },
  "budget": {
    "semantics": "max_models is the total number of trained generative candidates across all model families. Bootstrap baseline does not consume a training slot.",
    "max_trained_candidates": 2,
    "trained_candidates": 2,
    "allocation": {
      "vae": 1,
      "flow": 1
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
        "synthetic_mean": 6.875736463113725,
        "absolute_mean_difference": 0.10070244950828311,
        "test_std": 0.8074604298551417,
        "synthetic_std": 0.19055769670817188,
        "absolute_median_difference": 0.24777393341064435,
        "iqr_difference": 0.820016241073608
      },
      "volatile acidity": {
        "test_mean": 0.27482312925170066,
        "synthetic_mean": 0.2778334853453459,
        "absolute_mean_difference": 0.00301035609364525,
        "test_std": 0.09522423951688104,
        "synthetic_std": 0.012337109140592351,
        "absolute_median_difference": 0.015510966777801505,
        "iqr_difference": 0.1022339192032814
      },
      "citric acid": {
        "test_mean": 0.3271156462585034,
        "synthetic_mean": 0.33901810347912215,
        "absolute_mean_difference": 0.011902457220618723,
        "test_std": 0.11298236073311288,
        "synthetic_std": 0.026351113560518124,
        "absolute_median_difference": 0.01958638668060303,
        "iqr_difference": 0.0822918963432312
      },
      "residual sugar": {
        "test_mean": 6.242448979591837,
        "synthetic_mean": 6.305824176979922,
        "absolute_mean_difference": 0.06337519738808428,
        "test_std": 5.063269169496232,
        "synthetic_std": 3.313987210397491,
        "absolute_median_difference": 0.42172293663024885,
        "iqr_difference": 2.9959385037422175
      },
      "chlorides": {
        "test_mean": 0.04540952380952381,
        "synthetic_mean": 0.045952292799222534,
        "absolute_mean_difference": 0.0005427689896987248,
        "test_std": 0.021792020424802546,
        "synthetic_std": 0.007044906191790585,
        "absolute_median_difference": 0.004228393912315366,
        "iqr_difference": 0.0036876735836267466
      },
      "free sulfur dioxide": {
        "test_mean": 33.76190476190476,
        "synthetic_mean": 35.69514582039337,
        "absolute_mean_difference": 1.933241058488612,
        "test_std": 16.253302472723544,
        "synthetic_std": 7.299534703132283,
        "absolute_median_difference": 1.7805805206298828,
        "iqr_difference": 11.956844806671143
      },
      "total sulfur dioxide": {
        "test_mean": 134.2687074829932,
        "synthetic_mean": 139.43988396304633,
        "absolute_mean_difference": 5.171176480053134,
        "test_std": 40.114287075568235,
        "synthetic_std": 24.618610215956032,
        "absolute_median_difference": 6.670234680175781,
        "iqr_difference": 13.692033767700195
      },
      "density": {
        "test_mean": 0.993851605442177,
        "synthetic_mean": 0.9939953696002469,
        "absolute_mean_difference": 0.0001437641580699367,
        "test_std": 0.0029466864373343797,
        "synthetic_std": 0.002344753348751023,
        "absolute_median_difference": 8.67737293243076e-05,
        "iqr_difference": 0.0007028527116775818
      },
      "pH": {
        "test_mean": 3.1953605442176873,
        "synthetic_mean": 3.1872503773540126,
        "absolute_mean_difference": 0.008110166863674717,
        "test_std": 0.15403718720452117,
        "synthetic_std": 0.027156679287029636,
        "absolute_median_difference": 0.014850564002990563,
        "iqr_difference": 0.16229422092437762
      },
      "sulphates": {
        "test_mean": 0.48455782312925166,
        "synthetic_mean": 0.48775828178559777,
        "absolute_mean_difference": 0.0032004586563461057,
        "test_std": 0.11038480185068494,
        "synthetic_std": 0.007194617890127585,
        "absolute_median_difference": 0.02515889048576353,
        "iqr_difference": 0.13207741022109987
      },
      "alcohol": {
        "test_mean": 10.551224489795917,
        "synthetic_mean": 10.561842001715988,
        "absolute_mean_difference": 0.010617511920070655,
        "test_std": 1.2455696169595525,
        "synthetic_std": 0.8581224184785251,
        "absolute_median_difference": 0.051707363128661754,
        "iqr_difference": 0.4967934131622318
      }
    },
    "correlation_l1": 0.5647017062814379,
    "duplicate_fraction": 0.0,
    "unique_fraction": 1.0,
    "nearest_neighbor_mean": 0.8604684988400485,
    "nearest_neighbor_min": 0.5167123647198852,
    "marginal_fidelity_score": 0.4124877347584622,
    "train_scale_normalized_marginal_fidelity": 0.8009132419126543,
    "dependence_preservation_score": 0.6390994500648504,
    "selection_score": 0.6665933133350845,
    "objective_metrics": {
      "data_generation_fidelity": 0.4124877347584622,
      "train_scale_normalized_marginal_fidelity": 0.8009132419126543,
      "dependence_preservation": 0.6390994500648504,
      "selection_score": 0.6665933133350845
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
        "synthetic_mean": 6.827544393954154,
        "absolute_mean_difference": 0.05251038034871236,
        "test_std": 0.8074604298551417,
        "synthetic_std": 0.8248238010463025,
        "absolute_median_difference": 0.07047061920165998,
        "iqr_difference": 0.03482089042663539
      },
      "volatile acidity": {
        "test_mean": 0.27482312925170066,
        "synthetic_mean": 0.2794723718317979,
        "absolute_mean_difference": 0.004649242580097246,
        "test_std": 0.09522423951688104,
        "synthetic_std": 0.10386819025123337,
        "absolute_median_difference": 0.003427168130874625,
        "iqr_difference": 0.0010814246535301164
      },
      "citric acid": {
        "test_mean": 0.3271156462585034,
        "synthetic_mean": 0.33329465430687766,
        "absolute_mean_difference": 0.006179008048374235,
        "test_std": 0.11298236073311288,
        "synthetic_std": 0.12109702526742887,
        "absolute_median_difference": 0.009910064339637759,
        "iqr_difference": 0.0045627701282501265
      },
      "residual sugar": {
        "test_mean": 6.242448979591837,
        "synthetic_mean": 6.249043887563996,
        "absolute_mean_difference": 0.006594907972158559,
        "test_std": 5.063269169496232,
        "synthetic_std": 4.868808183820832,
        "absolute_median_difference": 0.07798557281494123,
        "iqr_difference": 0.7485275506973261
      },
      "chlorides": {
        "test_mean": 0.04540952380952381,
        "synthetic_mean": 0.04517667680802322,
        "absolute_mean_difference": 0.00023284700150059262,
        "test_std": 0.021792020424802546,
        "synthetic_std": 0.017012022375133724,
        "absolute_median_difference": 0.0011725084781646702,
        "iqr_difference": 0.0010590640455484385
      },
      "free sulfur dioxide": {
        "test_mean": 33.76190476190476,
        "synthetic_mean": 36.394824427553665,
        "absolute_mean_difference": 2.632919665648906,
        "test_std": 16.253302472723544,
        "synthetic_std": 16.91438965843107,
        "absolute_median_difference": 2.882617950439453,
        "iqr_difference": 0.5007944107055664
      },
      "total sulfur dioxide": {
        "test_mean": 134.2687074829932,
        "synthetic_mean": 138.31315793792487,
        "absolute_mean_difference": 4.0444504549316775,
        "test_std": 40.114287075568235,
        "synthetic_std": 42.45960320503492,
        "absolute_median_difference": 6.4967193603515625,
        "iqr_difference": 6.242549896240234
      },
      "density": {
        "test_mean": 0.993851605442177,
        "synthetic_mean": 0.9938849827716574,
        "absolute_mean_difference": 3.337732948038852e-05,
        "test_std": 0.0029466864373343797,
        "synthetic_std": 0.002912150162571602,
        "absolute_median_difference": 2.7228689193692368e-05,
        "iqr_difference": 2.189506530758667e-05
      },
      "pH": {
        "test_mean": 3.1953605442176873,
        "synthetic_mean": 3.195297392546863,
        "absolute_mean_difference": 6.315167082426854e-05,
        "test_std": 0.15403718720452117,
        "synthetic_std": 0.15277994150194396,
        "absolute_median_difference": 0.011898345947265465,
        "iqr_difference": 0.006186711788177313
      },
      "sulphates": {
        "test_mean": 0.48455782312925166,
        "synthetic_mean": 0.4845287946382558,
        "absolute_mean_difference": 2.9028490995841594e-05,
        "test_std": 0.11038480185068494,
        "synthetic_std": 0.10886078987845074,
        "absolute_median_difference": 0.015871920585632304,
        "iqr_difference": 0.006788387298583998
      },
      "alcohol": {
        "test_mean": 10.551224489795917,
        "synthetic_mean": 10.547363233936228,
        "absolute_mean_difference": 0.0038612558596895497,
        "test_std": 1.2455696169595525,
        "synthetic_std": 1.2166998452009397,
        "absolute_median_difference": 0.02724542617797887,
        "iqr_difference": 0.11595287322998082
      }
    },
    "correlation_l1": 0.035657732744951355,
    "duplicate_fraction": 0.0,
    "unique_fraction": 1.0,
    "nearest_neighbor_mean": 1.3784154777656976,
    "nearest_neighbor_min": 0.4206313973769945,
    "marginal_fidelity_score": 0.5788334767468954,
    "train_scale_normalized_marginal_fidelity": 0.9482815662334236,
    "dependence_preservation_score": 0.9655699642675938,
    "selection_score": 0.8804428876480775,
    "objective_metrics": {
      "data_generation_fidelity": 0.5788334767468954,
      "train_scale_normalized_marginal_fidelity": 0.9482815662334236,
      "dependence_preservation": 0.9655699642675938,
      "selection_score": 0.8804428876480775
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
    "max_epochs=20",
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
      "max_epochs=20",
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
      "flow": 2039
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
        "data_generation_fidelity": 0.406744168848898,
        "train_scale_normalized_marginal_fidelity": 0.8079649994575864,
        "dependence_preservation": 0.6321472709457345,
        "selection_score": 0.6661846283567006,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 0.8544359297273372
      },
      "flow": {
        "data_generation_fidelity": 0.7625078131558546,
        "train_scale_normalized_marginal_fidelity": 0.941405266720011,
        "dependence_preservation": 0.960810887402346,
        "selection_score": 0.912417743245997,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 1.3894101358012565
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
