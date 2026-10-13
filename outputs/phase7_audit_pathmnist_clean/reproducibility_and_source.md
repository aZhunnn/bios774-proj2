# PathMNIST Phase 7 Audit Reproducibility and Source

## Source and license

- Dataset: PathMNIST from MedMNIST.
- Local input: `data/medmnist/pathmnist.npz`.
- Source: MedMNIST PathMNIST, based on colorectal cancer histology patches from NCT-CRC-HE-100K and CRC-VAL-HE-7K.
- License recorded by the official MedMNIST metadata: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Modeling decision

- Objective: generate synthetic images for internal testing.
- Modeled variables: image pixels only.
- Label treatment: labels are metadata and are not used as conditioning variables.
- Subset used for this CPU-feasible audit: 256 training images, 64 validation images, and 64 held-out test images.
- Selection split: validation only with repeated generation seeds.
- Final held-out split: used only after model selection.

## Clean reproduction command

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /opt/miniconda3/bin/python -m generative_agent.cli data/medmnist/pathmnist.npz --modality image --objective 'Generate synthetic images for internal testing.' --output-dir outputs/phase7_audit_pathmnist_clean --epochs 2 --max-models 2 --seed 2026 --n-synthetic 32 --max-train-samples 256 --max-validation-samples 64 --max-test-samples 64 --image-latent-dim 16 --gan-latent-dim 32 --repeated-generation-seeds 2
```

## Audit note

The clean audit run reproduced the Phase 7 image validation and held-out test scores exactly. The selected model is `image_gan` under the stored validation-only selection rule. The report does not claim clinical realism, privacy protection, or external-release adequacy.
