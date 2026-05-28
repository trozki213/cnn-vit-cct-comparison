# Inductive Bias under Data Scarcity

**A parameter-matched comparison of CNN, ViT, and CCT architectures on CIFAR-10.**

Course project for *30562 — Machine Learning and Artificial Intelligence*, Bocconi University.

## Abstract

We compare three vision architectures — a ResNet-style **CNN**, a vanilla **Vision Transformer (ViT)**, and a **Compact Convolutional Transformer (CCT)** — on CIFAR-10 across five data fractions (10%, 25%, 50%, 75%, 100%) under two parameter budgets (~0.75 M and ~5 M) and two augmentation conditions. Beyond accuracy, we probe internal representations with **Centered Kernel Alignment (CKA)**, **linear probes**, and **mean attention distance**. Our central finding is a **low-data augmentation crossover**: contrary to the standard prediction, the CNN benefits more from random crop and flip at low data, while the ViT benefits more at high data. The CCT remains augmentation-independent throughout, consistent with its convolutional tokenizer pre-encoding the relevant spatial invariances.

See [`ML#13_report.pdf`](./ML%2313_report.pdf) for the full report.

## Authors

- Victor Antonescu — `victor.antonescu@studbocconi.it`
- Letizia Di Pietro — `letizia.dipietro@studbocconi.it`
- Martin Lukanov — `martin.lukanov@studbocconi.it`
- Alexandru Stoica — `alexandru.stoica@studbocconi.it`
- Antonio Troiano — `antonio.troiano@studbocconi.it`

## Repository structure

```
.
├── README.md
├── requirements.txt
├── ML#13_report.pdf              # 17-page report (main + appendix)
├── ML#13.ipynb                   # Analysis notebook (loads results, generates all figures)
├── models/                       # One script per architecture
│   ├── cnn_small_script.py       # ~0.76 M params
│   ├── cnn_large_script.py       # ~4.90 M params
│   ├── vit_small_script.py       # ~0.76 M params
│   ├── vit_large_script.py       # ~4.98 M params
│   ├── cct_small_script.py       # ~0.73 M params
│   └── cct_large_script.py       # ~5.26 M params
├── training_scripts/
│   └── train_hpc.py              # Entry point used on HPC to train all models
├── parameters_aug/               # 30 PyTorch checkpoints (.pt), with augmentation
├── parameters_noaug/             # 30 PyTorch checkpoints (.pt), without augmentation
├── resultsxepochs_aug/           # 30 JSON files: per-epoch {epoch, loss, acc} (aug)
└── resultsxepochs_noaug/         # 30 JSON files: per-epoch {epoch, loss, acc} (no aug)
```

The `dataset/` folder (CIFAR-10 raw archive, 163 MB) is **excluded** from the repo and downloaded automatically by `torchvision` on first run.

## Setup

```bash
# Clone
git clone <REPO_URL>
cd <REPO_NAME>

# Create env (optional)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Reproducing the analysis

The notebook is decoupled from training: it loads pre-saved JSON logs and `.pt` checkpoints, so all figures and tables can be reproduced on CPU in minutes.

```bash
jupyter notebook 'ML#13.ipynb'
```

- **Accuracy tables, learning curves, augmentation gain** → load from `resultsxepochs_*/` JSON files (fast, no model loading)
- **CKA, linear probes, mean attention distance, ERF saliency** → load `.pt` checkpoints from `parameters_*/` (under a minute per model on CPU)

## Re-training from scratch (optional, requires GPU)

```bash
python training_scripts/train_hpc.py --model cnn_large --augment   # or --no-augment
```

This trains the chosen architecture across all five data fractions (10–100%) for 150 epochs each. Available `--model` values: `{cnn,vit,cct}_{small,large}`.

Training hyperparameters (fixed across all 6 models, see report §A.4):
- Optimizer: AdamW, peak LR 1e-3, weight decay 5e-2
- Schedule: linear warm-up (~7 epochs) + cosine annealing
- Batch size 256, 150 epochs, label smoothing ε = 0.1
- Gradient clipping ‖g‖₂ ≤ 1.0, mixed precision (AMP)
- Single fixed seed (`torch.manual_seed(0)`)

## Headline results

Final test accuracy at epoch 150, with augmentation:

| Model | 10% | 25% | 50% | 75% | 100% |
|---|---:|---:|---:|---:|---:|
| CNN_large | 79.80 | 88.57 | 92.18 | 93.94 | **94.91** |
| ViT_large | 54.40 | 68.28 | 77.53 | 82.43 | 85.80 |
| CCT_large | 70.47 | 81.13 | 86.55 | 89.36 | 90.84 |
| CNN_small | 77.40 | 83.41 | 87.07 | 88.28 | 89.27 |
| ViT_small | 53.73 | 65.52 | 74.08 | 79.58 | 82.60 |
| CCT_small | 68.11 | 79.19 | 84.90 | 87.62 | **89.65** |

Full numerical results and ablations in the [report](./ML%2313_report.pdf).
