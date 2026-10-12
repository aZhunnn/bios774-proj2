# Phase 5C Image Evaluation and Selection Report

## Dataset Profile
```json
{
  "name": "pathmnist",
  "n_samples": 1536,
  "split_counts": {
    "train": 1024,
    "validation": 256,
    "test": 256
  },
  "image_shape": [
    28,
    28,
    3
  ],
  "channels": 3,
  "value_range": [
    0.0,
    255.0
  ],
  "label_counts": {
    "0": 170,
    "1": 170,
    "2": 173,
    "3": 171,
    "4": 138,
    "5": 200,
    "6": 141,
    "7": 151,
    "8": 222
  },
  "labels_are_metadata": true,
  "normalization": "unit",
  "reconstruction_assumption": "gaussian_unit_variance",
  "notes": [
    "Labels are retained as metadata and are not used for conditioning in Phase 5A.",
    "Gaussian unit-variance reconstruction loss is used on normalized pixel intensities."
  ]
}
```

## Selected Model Configuration
```json
{
  "image_shape": [
    28,
    28,
    3
  ],
  "latent_dim": 24,
  "hidden_channels": [
    16,
    32
  ],
  "learning_rate": 0.001,
  "batch_size": 64,
  "epochs": 5,
  "kl_weight": 1.0,
  "early_stopping_patience": 3,
  "seed": 2026,
  "device": "cpu",
  "reconstruction_assumption": "gaussian_unit_variance"
}
```

## Selected Model Training
```json
{
  "history": {
    "train_loss": [
      2228.457977294922,
      2211.7002563476562,
      2193.2323455810547,
      2188.9945526123047,
      2186.745895385742
    ],
    "validation_loss": [
      2221.3397216796875,
      2198.5670776367188,
      2189.0623168945312,
      2186.6253051757812,
      2184.8577270507812
    ],
    "validation_reconstruction": [
      2220.3545532226562,
      2195.6784057617188,
      2186.6333618164062,
      2182.897705078125,
      2183.3090209960938
    ]
  },
  "best_validation_loss": 2184.8577270507812,
  "epochs_ran": 5,
  "checkpoint_path": "outputs/phase5c_pathmnist_reliable/image_vae_checkpoint.pt",
  "wall_seconds": 1.19178850017488,
  "device": "cpu",
  "failed": false,
  "failure_reason": null
}
```

## Validation Evaluation
```json
{
  "model_name": "image_vae",
  "selection_score": 0.7002887525616692,
  "pixel_mean_abs_error": 0.0682988166809082,
  "pixel_std_abs_error": 0.08950981497764587,
  "gradient_mean_abs_error": 0.013687863945960999,
  "gradient_std_abs_error": 0.02747713401913643,
  "local_patch_std_abs_error": 0.022813264280557632,
  "pairwise_distance_mean_abs_error": 0.11744822561740875,
  "feature_mmd": 0.4373190402984619,
  "reconstruction_mse": 0.018786534667015076,
  "pixel_mean_real": 0.6618775725364685,
  "pixel_mean_generated": 0.5935787558555603,
  "pixel_std_real": 0.16807186603546143,
  "pixel_std_generated": 0.07856205105781555,
  "duplicate_fraction": 0.0,
  "nearest_neighbor_mean": 3.118837833404541,
  "nearest_neighbor_min": 2.2839362621307373,
  "generation_seconds": 0.0019092499278485775,
  "notes": [
    "Reconstruction metrics evaluate autoencoding behavior, not unconditional generation quality.",
    "Nearest-neighbor diagnostics are memorization proxies, not privacy guarantees.",
    "Selection score is validation-only and combines pixel and simple gradient summary errors with a duplicate penalty; it does not use VAE ELBO or GAN discriminator loss.",
    "The handcrafted feature MMD is not a validated pathology quality metric and is not clinically meaningful by itself.",
    "No numeric threshold in this report establishes adequate image quality for clinical use or external release."
  ]
}
```

## Held-Out Test Evaluation
```json
{
  "model_name": "image_vae",
  "selection_score": 0.6692447373180194,
  "pixel_mean_abs_error": 0.07419431209564209,
  "pixel_std_abs_error": 0.12042797356843948,
  "gradient_mean_abs_error": 0.011801205575466156,
  "gradient_std_abs_error": 0.030975600704550743,
  "local_patch_std_abs_error": 0.02271663397550583,
  "pairwise_distance_mean_abs_error": 0.17321030050516129,
  "feature_mmd": 0.3291362524032593,
  "reconstruction_mse": 0.025586361065506935,
  "pixel_mean_real": 0.6644418835639954,
  "pixel_mean_generated": 0.5902475714683533,
  "pixel_std_real": 0.19895532727241516,
  "pixel_std_generated": 0.07852735370397568,
  "duplicate_fraction": 0.0,
  "nearest_neighbor_mean": 3.157261371612549,
  "nearest_neighbor_min": 2.377669095993042,
  "generation_seconds": 0.0019863336347043514,
  "notes": [
    "Reconstruction metrics evaluate autoencoding behavior, not unconditional generation quality.",
    "Nearest-neighbor diagnostics are memorization proxies, not privacy guarantees.",
    "Selection score is validation-only and combines pixel and simple gradient summary errors with a duplicate penalty; it does not use VAE ELBO or GAN discriminator loss.",
    "The handcrafted feature MMD is not a validated pathology quality metric and is not clinically meaningful by itself.",
    "No numeric threshold in this report establishes adequate image quality for clinical use or external release."
  ]
}
```

## Candidate Results
```json
{
  "image_gan": {
    "status": "trained",
    "config": {
      "image_shape": [
        28,
        28,
        3
      ],
      "latent_dim": 48,
      "generator_channels": [
        32,
        16
      ],
      "discriminator_channels": [
        16,
        32
      ],
      "generator_learning_rate": 0.0002,
      "discriminator_learning_rate": 0.0002,
      "batch_size": 64,
      "epochs": 5,
      "seed": 2043,
      "device": "cpu",
      "stability_patience": 2
    },
    "training": {
      "history": {
        "generator_loss": [
          0.9126000963151455,
          1.1514684781432152,
          1.4692856520414352,
          1.693654514849186,
          2.0377133041620255
        ],
        "discriminator_loss": [
          0.6181855574250221,
          0.48666675947606564,
          0.38040031492710114,
          0.32830602303147316,
          0.2787376195192337
        ],
        "sample_std": [
          0.08183473348617554,
          0.10121125727891922,
          0.10947027057409286,
          0.1185067668557167,
          0.12313059717416763
        ]
      },
      "checkpoint_path": "outputs/phase5c_pathmnist_reliable/image_gan_checkpoint.pt",
      "wall_seconds": 2.619235457852483,
      "device": "cpu",
      "epochs_ran": 5,
      "failed": false,
      "failure_reason": null
    },
    "validation_evaluation": {
      "model_name": "image_gan",
      "selection_score": 0.6800207180821909,
      "pixel_mean_abs_error": 0.11527329683303833,
      "pixel_std_abs_error": 0.04280710220336914,
      "gradient_mean_abs_error": 0.06018160283565521,
      "gradient_std_abs_error": 0.028724651783704758,
      "local_patch_std_abs_error": 0.035526592284440994,
      "pairwise_distance_mean_abs_error": 0.061475545167922974,
      "feature_mmd": 0.684031069278717,
      "reconstruction_mse": null,
      "pixel_mean_real": 0.6618775725364685,
      "pixel_mean_generated": 0.5466042757034302,
      "pixel_std_real": 0.16807186603546143,
      "pixel_std_generated": 0.12526476383209229,
      "duplicate_fraction": 0.0,
      "nearest_neighbor_mean": 5.099831581115723,
      "nearest_neighbor_min": 4.5270490646362305,
      "generation_seconds": 0.0027320832014083862,
      "notes": [
        "Reconstruction metrics evaluate autoencoding behavior, not unconditional generation quality.",
        "Nearest-neighbor diagnostics are memorization proxies, not privacy guarantees.",
        "Selection score is validation-only and combines pixel and simple gradient summary errors with a duplicate penalty; it does not use VAE ELBO or GAN discriminator loss.",
        "The handcrafted feature MMD is not a validated pathology quality metric and is not clinically meaningful by itself.",
        "No numeric threshold in this report establishes adequate image quality for clinical use or external release."
      ]
    },
    "validation_repeated_evaluations": [
      {
        "model_name": "image_gan",
        "selection_score": 0.6800207180821909,
        "pixel_mean_abs_error": 0.11527329683303833,
        "pixel_std_abs_error": 0.04280710220336914,
        "gradient_mean_abs_error": 0.06018160283565521,
        "gradient_std_abs_error": 0.028724651783704758,
        "local_patch_std_abs_error": 0.035526592284440994,
        "pairwise_distance_mean_abs_error": 0.061475545167922974,
        "feature_mmd": 0.684031069278717,
        "reconstruction_mse": null,
        "pixel_mean_real": 0.6618775725364685,
        "pixel_mean_generated": 0.5466042757034302,
        "pixel_std_real": 0.16807186603546143,
        "pixel_std_generated": 0.12526476383209229,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 5.099831581115723,
        "nearest_neighbor_min": 4.5270490646362305,
        "generation_seconds": 0.0027320832014083862,
        "notes": [
          "Reconstruction metrics evaluate autoencoding behavior, not unconditional generation quality.",
          "Nearest-neighbor diagnostics are memorization proxies, not privacy guarantees.",
          "Selection score is validation-only and combines pixel and simple gradient summary errors with a duplicate penalty; it does not use VAE ELBO or GAN discriminator loss.",
          "The handcrafted feature MMD is not a validated pathology quality metric and is not clinically meaningful by itself.",
          "No numeric threshold in this report establishes adequate image quality for clinical use or external release."
        ]
      },
      {
        "model_name": "image_gan",
        "selection_score": 0.6797123588587509,
        "pixel_mean_abs_error": 0.1151728630065918,
        "pixel_std_abs_error": 0.0420428067445755,
        "gradient_mean_abs_error": 0.06099116802215576,
        "gradient_std_abs_error": 0.02931879833340645,
        "local_patch_std_abs_error": 0.03622130677103996,
        "pairwise_distance_mean_abs_error": 0.060424238443374634,
        "feature_mmd": 0.6887774467468262,
        "reconstruction_mse": null,
        "pixel_mean_real": 0.6618775725364685,
        "pixel_mean_generated": 0.5467047095298767,
        "pixel_std_real": 0.16807186603546143,
        "pixel_std_generated": 0.12602905929088593,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 5.128130912780762,
        "nearest_neighbor_min": 4.2824554443359375,
        "generation_seconds": 0.003179582767188549,
        "notes": [
          "Reconstruction metrics evaluate autoencoding behavior, not unconditional generation quality.",
          "Nearest-neighbor diagnostics are memorization proxies, not privacy guarantees.",
          "Selection score is validation-only and combines pixel and simple gradient summary errors with a duplicate penalty; it does not use VAE ELBO or GAN discriminator loss.",
          "The handcrafted feature MMD is not a validated pathology quality metric and is not clinically meaningful by itself.",
          "No numeric threshold in this report establishes adequate image quality for clinical use or external release."
        ]
      },
      {
        "model_name": "image_gan",
        "selection_score": 0.6799664821308545,
        "pixel_mean_abs_error": 0.1152656078338623,
        "pixel_std_abs_error": 0.04227423667907715,
        "gradient_mean_abs_error": 0.06081237643957138,
        "gradient_std_abs_error": 0.029503244906663895,
        "local_patch_std_abs_error": 0.036031272262334824,
        "pairwise_distance_mean_abs_error": 0.061280496418476105,
        "feature_mmd": 0.6826052069664001,
        "reconstruction_mse": null,
        "pixel_mean_real": 0.6618775725364685,
        "pixel_mean_generated": 0.5466119647026062,
        "pixel_std_real": 0.16807186603546143,
        "pixel_std_generated": 0.12579762935638428,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 5.120318412780762,
        "nearest_neighbor_min": 4.416153907775879,
        "generation_seconds": 0.0022415416315197945,
        "notes": [
          "Reconstruction metrics evaluate autoencoding behavior, not unconditional generation quality.",
          "Nearest-neighbor diagnostics are memorization proxies, not privacy guarantees.",
          "Selection score is validation-only and combines pixel and simple gradient summary errors with a duplicate penalty; it does not use VAE ELBO or GAN discriminator loss.",
          "The handcrafted feature MMD is not a validated pathology quality metric and is not clinically meaningful by itself.",
          "No numeric threshold in this report establishes adequate image quality for clinical use or external release."
        ]
      }
    ],
    "validation_reliability": {
      "selection_score_mean": 0.6798998530239322,
      "selection_score_std": 0.0001344146100369661,
      "selection_score_min": 0.6797123588587509,
      "selection_score_max": 0.6800207180821909,
      "n_generation_seeds": 3,
      "practical_difference_threshold": 0.01
    },
    "checkpoint_path": "outputs/phase5c_pathmnist_reliable/image_gan_checkpoint.pt"
  },
  "image_vae": {
    "status": "trained",
    "config": {
      "image_shape": [
        28,
        28,
        3
      ],
      "latent_dim": 24,
      "hidden_channels": [
        16,
        32
      ],
      "learning_rate": 0.001,
      "batch_size": 64,
      "epochs": 5,
      "kl_weight": 1.0,
      "early_stopping_patience": 3,
      "seed": 2026,
      "device": "cpu",
      "reconstruction_assumption": "gaussian_unit_variance"
    },
    "training": {
      "history": {
        "train_loss": [
          2228.457977294922,
          2211.7002563476562,
          2193.2323455810547,
          2188.9945526123047,
          2186.745895385742
        ],
        "validation_loss": [
          2221.3397216796875,
          2198.5670776367188,
          2189.0623168945312,
          2186.6253051757812,
          2184.8577270507812
        ],
        "validation_reconstruction": [
          2220.3545532226562,
          2195.6784057617188,
          2186.6333618164062,
          2182.897705078125,
          2183.3090209960938
        ]
      },
      "best_validation_loss": 2184.8577270507812,
      "epochs_ran": 5,
      "checkpoint_path": "outputs/phase5c_pathmnist_reliable/image_vae_checkpoint.pt",
      "wall_seconds": 1.19178850017488,
      "device": "cpu",
      "failed": false,
      "failure_reason": null
    },
    "validation_evaluation": {
      "model_name": "image_vae",
      "selection_score": 0.7002887525616692,
      "pixel_mean_abs_error": 0.0682988166809082,
      "pixel_std_abs_error": 0.08950981497764587,
      "gradient_mean_abs_error": 0.013687863945960999,
      "gradient_std_abs_error": 0.02747713401913643,
      "local_patch_std_abs_error": 0.022813264280557632,
      "pairwise_distance_mean_abs_error": 0.11744822561740875,
      "feature_mmd": 0.4373190402984619,
      "reconstruction_mse": 0.018786534667015076,
      "pixel_mean_real": 0.6618775725364685,
      "pixel_mean_generated": 0.5935787558555603,
      "pixel_std_real": 0.16807186603546143,
      "pixel_std_generated": 0.07856205105781555,
      "duplicate_fraction": 0.0,
      "nearest_neighbor_mean": 3.118837833404541,
      "nearest_neighbor_min": 2.2839362621307373,
      "generation_seconds": 0.0019092499278485775,
      "notes": [
        "Reconstruction metrics evaluate autoencoding behavior, not unconditional generation quality.",
        "Nearest-neighbor diagnostics are memorization proxies, not privacy guarantees.",
        "Selection score is validation-only and combines pixel and simple gradient summary errors with a duplicate penalty; it does not use VAE ELBO or GAN discriminator loss.",
        "The handcrafted feature MMD is not a validated pathology quality metric and is not clinically meaningful by itself.",
        "No numeric threshold in this report establishes adequate image quality for clinical use or external release."
      ]
    },
    "validation_repeated_evaluations": [
      {
        "model_name": "image_vae",
        "selection_score": 0.7002887525616692,
        "pixel_mean_abs_error": 0.0682988166809082,
        "pixel_std_abs_error": 0.08950981497764587,
        "gradient_mean_abs_error": 0.013687863945960999,
        "gradient_std_abs_error": 0.02747713401913643,
        "local_patch_std_abs_error": 0.022813264280557632,
        "pairwise_distance_mean_abs_error": 0.11744822561740875,
        "feature_mmd": 0.4373190402984619,
        "reconstruction_mse": 0.018786534667015076,
        "pixel_mean_real": 0.6618775725364685,
        "pixel_mean_generated": 0.5935787558555603,
        "pixel_std_real": 0.16807186603546143,
        "pixel_std_generated": 0.07856205105781555,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 3.118837833404541,
        "nearest_neighbor_min": 2.2839362621307373,
        "generation_seconds": 0.0019092499278485775,
        "notes": [
          "Reconstruction metrics evaluate autoencoding behavior, not unconditional generation quality.",
          "Nearest-neighbor diagnostics are memorization proxies, not privacy guarantees.",
          "Selection score is validation-only and combines pixel and simple gradient summary errors with a duplicate penalty; it does not use VAE ELBO or GAN discriminator loss.",
          "The handcrafted feature MMD is not a validated pathology quality metric and is not clinically meaningful by itself.",
          "No numeric threshold in this report establishes adequate image quality for clinical use or external release."
        ]
      },
      {
        "model_name": "image_vae",
        "selection_score": 0.7058353471875735,
        "pixel_mean_abs_error": 0.06632518768310547,
        "pixel_std_abs_error": 0.08721110969781876,
        "gradient_mean_abs_error": 0.013646405190229416,
        "gradient_std_abs_error": 0.027352649718523026,
        "local_patch_std_abs_error": 0.022740624845027924,
        "pairwise_distance_mean_abs_error": 0.11438006162643433,
        "feature_mmd": 0.4224179983139038,
        "reconstruction_mse": 0.018331658095121384,
        "pixel_mean_real": 0.6618775725364685,
        "pixel_mean_generated": 0.595552384853363,
        "pixel_std_real": 0.16807186603546143,
        "pixel_std_generated": 0.08086075633764267,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 3.114124298095703,
        "nearest_neighbor_min": 2.1778697967529297,
        "generation_seconds": 0.0021642078645527363,
        "notes": [
          "Reconstruction metrics evaluate autoencoding behavior, not unconditional generation quality.",
          "Nearest-neighbor diagnostics are memorization proxies, not privacy guarantees.",
          "Selection score is validation-only and combines pixel and simple gradient summary errors with a duplicate penalty; it does not use VAE ELBO or GAN discriminator loss.",
          "The handcrafted feature MMD is not a validated pathology quality metric and is not clinically meaningful by itself.",
          "No numeric threshold in this report establishes adequate image quality for clinical use or external release."
        ]
      },
      {
        "model_name": "image_vae",
        "selection_score": 0.7048199589691294,
        "pixel_mean_abs_error": 0.06721198558807373,
        "pixel_std_abs_error": 0.08794417977333069,
        "gradient_mean_abs_error": 0.0133531354367733,
        "gradient_std_abs_error": 0.02734663151204586,
        "local_patch_std_abs_error": 0.02253156155347824,
        "pairwise_distance_mean_abs_error": 0.11340246349573135,
        "feature_mmd": 0.42944788932800293,
        "reconstruction_mse": 0.018749427050352097,
        "pixel_mean_real": 0.6618775725364685,
        "pixel_mean_generated": 0.5946655869483948,
        "pixel_std_real": 0.16807186603546143,
        "pixel_std_generated": 0.08012768626213074,
        "duplicate_fraction": 0.0,
        "nearest_neighbor_mean": 3.126739501953125,
        "nearest_neighbor_min": 2.3633182048797607,
        "generation_seconds": 0.0018906248733401299,
        "notes": [
          "Reconstruction metrics evaluate autoencoding behavior, not unconditional generation quality.",
          "Nearest-neighbor diagnostics are memorization proxies, not privacy guarantees.",
          "Selection score is validation-only and combines pixel and simple gradient summary errors with a duplicate penalty; it does not use VAE ELBO or GAN discriminator loss.",
          "The handcrafted feature MMD is not a validated pathology quality metric and is not clinically meaningful by itself.",
          "No numeric threshold in this report establishes adequate image quality for clinical use or external release."
        ]
      }
    ],
    "validation_reliability": {
      "selection_score_mean": 0.7036480195727908,
      "selection_score_std": 0.0024112596220710715,
      "selection_score_min": 0.7002887525616692,
      "selection_score_max": 0.7058353471875735,
      "n_generation_seeds": 3,
      "practical_difference_threshold": 0.01
    },
    "checkpoint_path": "outputs/phase5c_pathmnist_reliable/image_vae_checkpoint.pt"
  }
}
```

## Agent Decisions
```json
{
  "architecture": "Inspect -> Interpret -> Check Eligibility -> Plan -> Execute -> Evaluate -> Reflect -> Report",
  "eligibility": {
    "use_case": "internal_testing",
    "selected_method": "vae",
    "eligible_methods": [
      "gan",
      "vae"
    ],
    "rejected_methods": [],
    "method_eligibility": [
      {
        "method": "gan",
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
      }
    ],
    "ranking_metric": "selection_score",
    "selection_reason": "Selected highest selection_score among eligible methods; hard eligibility rules were applied before ranking."
  },
  "labels": "metadata_only_not_conditioning",
  "reconstruction_assumption": "gaussian_unit_variance",
  "selection_rule": "Train up to max_models total feasible eligible image candidates; select highest shared validation selection_score; evaluate held-out test only after selection.",
  "selection_metric": "1 / (1 + pixel mean error + pixel std error + 0.5 gradient mean error + 0.5 gradient std error + local patch std error + diversity error + 0.25 handcrafted feature MMD + duplicate fraction)",
  "adequacy_warning_threshold": 0.85,
  "absolute_evidence": {
    "selected_validation_score_mean": 0.7036480195727908,
    "warning_threshold": 0.85,
    "adequacy_claim": false,
    "quality_warning": true,
    "top_two_score_margin": 0.023748166548858585,
    "difference_practically_meaningful_by_heuristic": true,
    "interpretation": "Ranking is relative and heuristic; no score threshold establishes clinical realism, privacy, or external-release suitability."
  },
  "selected_candidate": "image_vae",
  "workflow_wall_seconds": 8.390011916868389,
  "limitations": [
    "Small image experiments are not evidence of clinically meaningful generative quality.",
    "No privacy guarantee is implemented.",
    "Image VAE and GAN are unconditional; labels are not used for conditioning.",
    "The handcrafted feature MMD is not KID/FID and is not a validated pathology quality metric."
  ]
}
```
