# Deep-Learning-NYU

Project 1: ResNet on CIFAR-10 under 5M parameters (NYU Tandon Deep Learning, Fall 2026, Kaggle competition).
Team: Sakshi Sawant (Kaggle: Sakshi3003) and Harshal Pimpalshende (Kaggle: Noobacker).

Kaggle result of the final submission (13): **92.9% private / 93.3% public**.

## Code in this repository

Models are trained from scratch on the permitted CIFAR-10 training data only. The `sub*` notebooks are the originals exported from Kaggle, unchanged, outputs included.

| Submission | File | Notes |
|---|---|---|
| 2-4 (4.37M ResNet 80 ep; 3-model ensemble; 4.82M ResNet 150 ep) | `sub02-05_resnet_wrn_experiments.ipynb` | One notebook holding these experiments and the Wide ResNet code. It is the final saved version, so it shows the end state and not each submission separately. |
| 5 (Wide ResNet, stopped at 40 epochs) | `sub02-05_resnet_wrn_experiments.ipynb` | Last cells (WRN-28, 4,480,474 params). |
| 6 (Wide ResNet, 160 more epochs) | `sub06_wrn_recovery_160_epochs.ipynb` | Restores the epoch-41 checkpoint, then Mixup/CutMix training. |
| 8 (NYU-DL v2) | `sub08_nyu_dl_v2.ipynb` | Small ResNet-style model, 1,646,122 params. |
| 9 (fast baseline, 35 ep) | `sub09_fast_baseline_35ep.ipynb` | 4,327,754 params, one-cycle SGD, CutMix, EMA, flip TTA. |
| 10 (per-image normalisation, 50 ep) | `sub10_per_image_norm_50ep.ipynb` | Adds per-image standardisation and photometric augmentation. |
| 11 (80 ep, dropout 0.1) | `sub11_robust_aug_80ep.ipynb` | Stronger augmentation. |
| 12 (rerun of 11) | `sub12_rerun_80ep.ipynb` | Same configuration, rerun. Best documented run from a saved script: 85.9% private / 86.4% public. |
| Reproducible pipeline | `cifar10_reproducible_training.ipynb` | New, fully seeded pipeline for the robustness-focused approach (per-image normalisation, photometric augmentation, consistency loss, EMA). **Not the code of submission 13, and its full training run has not been completed**, so no accuracy is claimed for it here. Results will be added once a full run has finished. |

## Results of the Kaggle submissions

| # | What we submitted | Public | Private |
|---|---|---|---|
| 1 | ResNet-20 style, 272k params, 18 epochs | 78.8% | 76.5% |
| 2 | Wider ResNet, 4.37M, 80 epochs | 86.7% | 81.9% |
| 3 | Ensemble of three (over the 5M budget, not a valid entry) | 87.3% | 84.2% |
| 4 | Single ResNet 4.82M, 150 epochs | 86.7% | 83.2% |
| 5 | Wide ResNet, stopped at 40 epochs | 81.1% | 79.1% |
| 6 | Wide ResNet, +160 epochs | 88.0% | 86.1% |
| 7 | Pretrained BiT fine-tune (diagnostic only, not a valid entry) | 88.1% | 85.9% |
| 8 | NYU-DL v2 | 81.9% | 79.9% |
| 9 | Fast pipeline, 35 epochs | 84.3% | 82.1% |
| 10 | Per-image norm + photometric aug, 50 epochs | 86.2% | 83.8% |
| 11 | Stronger aug, dropout 0.1, 80 epochs | 86.6% | 83.8% |
| 12 | Rerun of 11 | 86.4% | 85.9% |
| 13 | Final submission (trained locally) | 93.3% | 92.9% |

## Notes on specific submissions

- **Submission 1** (ResNet-20 baseline): its notebook is not in this repository.
- **Submission 7** (pretrained BiT ResNetV2-50, ImageNet-21k): not included. It used external pretrained weights, which the competition rules do not allow. It was a diagnostic and is not counted as a result.
- **Submission 13**: trained locally, and the original training script is not in this repository. The 92.9% / 93.3% scores are as reported by Kaggle. `cifar10_reproducible_training.ipynb` is a separate, newly written pipeline and has not yet been shown to reproduce them.

See the project report for the full methodology and findings.
