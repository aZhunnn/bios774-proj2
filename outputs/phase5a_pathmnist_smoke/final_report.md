# Phase 5A Image VAE Smoke Report

## Dataset Profile
```json
{
  "name": "pathmnist",
  "n_samples": 384,
  "split_counts": {
    "train": 256,
    "validation": 64,
    "test": 64
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
    "0": 39,
    "1": 47,
    "2": 36,
    "3": 46,
    "4": 34,
    "5": 42,
    "6": 42,
    "7": 36,
    "8": 62
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

## Model Configuration
```json
{
  "image_shape": [
    28,
    28,
    3
  ],
  "latent_dim": 16,
  "hidden_channels": [
    16,
    32
  ],
  "learning_rate": 0.001,
  "batch_size": 64,
  "epochs": 2,
  "kl_weight": 1.0,
  "early_stopping_patience": 3,
  "seed": 2026,
  "device": "cpu",
  "reconstruction_assumption": "gaussian_unit_variance"
}
```

## Training
```json
{
  "history": {
    "train_loss": [
      2220.2853393554688,
      2217.2435913085938
    ],
    "validation_loss": [
      2213.208251953125,
      2210.557373046875
    ],
    "validation_reconstruction": [
      2213.189453125,
      2210.499755859375
    ]
  },
  "best_validation_loss": 2210.557373046875,
  "epochs_ran": 2,
  "checkpoint_path": "outputs/phase5a_pathmnist_smoke/image_vae_checkpoint.pt",
  "wall_seconds": 0.12974737491458654,
  "device": "cpu",
  "failed": false,
  "failure_reason": null
}
```

## Evaluation
```json
{
  "reconstruction_mse": 0.041801873594522476,
  "pixel_mean_real": 0.6355350017547607,
  "pixel_mean_generated": 0.5233013033866882,
  "pixel_std_real": 0.1732720285654068,
  "pixel_std_generated": 0.029617933556437492,
  "duplicate_fraction": 0.0,
  "nearest_neighbor_mean": 5.754234313964844,
  "nearest_neighbor_min": 5.696342468261719,
  "generation_seconds": 0.0008316668681800365,
  "notes": [
    "Reconstruction metrics evaluate autoencoding behavior, not unconditional generation quality.",
    "Nearest-neighbor diagnostics are memorization proxies, not privacy guarantees."
  ]
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
      "vae"
    ],
    "rejected_methods": [],
    "method_eligibility": [
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
  "workflow_wall_seconds": 1.5018826248124242,
  "limitations": [
    "Smoke test uses a small subset and is not evidence of production generative quality.",
    "No privacy guarantee is implemented.",
    "Image VAE is unconditional; labels are not used for conditioning."
  ]
}
```
