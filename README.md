# Pulmonary Anomaly Detection in Chest Radiographs

Unsupervised anomaly detection in chest X-rays using a Convolutional Autoencoder.
Trained exclusively on normal radiographs — no disease labels used during training.
Anomalies are detected as high reconstruction error at inference time.

## Project structure

```
pulmonary-anomaly-detection/
├── config.py                    # all hyperparameters and paths
├── pyproject.toml               # makes project pip-installable
├── requirements.txt
├── scripts/
│   ├── download_data.py         # Kaggle dataset download
│   ├── train.py                 # training entry point
│   ├── evaluate.py              # metrics + figures entry point
│   └── run_app.py               # Flask web app entry point
├── src/
│   ├── data/                    # transforms, datasets, dataloaders
│   ├── model/                   # encoder, decoder, autoencoder
│   ├── training/                # loss, callbacks, trainer
│   ├── evaluation/              # scorer, metrics, visualiser
│   └── utils/                   # seed, device, logger
└── app/
    ├── api/                     # Flask Blueprint, validators
    ├── services/                # inference service, image processor
    ├── templates/               # Jinja2 HTML templates
    └── static/                  # CSS, JS
```

## Setup

### 1. Install

```bash
git clone https://github.com/your-username/pulmonary-anomaly-detection
cd pulmonary-anomaly-detection
pip install -e .
```

The `-e .` install makes all `src/` and `app/` packages importable
from anywhere — no sys.path hacks needed.

### 2. Download dataset

```bash
# Place your kaggle.json at ~/.kaggle/kaggle.json first
python scripts/download_data.py
```

On Colab:
```python
from google.colab import files
files.upload()   # upload kaggle.json
import os, shutil
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
shutil.move("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
!python scripts/download_data.py
```

### 3. Train

```bash
python scripts/train.py
```

Trains only on NORMAL images (unsupervised). Saves best checkpoint
to `checkpoints/best_model.pth`. Training curves saved to `outputs/`.

### 4. Evaluate

```bash
python scripts/evaluate.py
```

Generates 9 figures in `outputs/` and saves `metrics.csv` and
`thresholds.json`. The thresholds file is loaded automatically
by the web app — no manual editing needed.

### 5. Run web app

```bash
python scripts/run_app.py
# Open http://localhost:5000
```

## How it works

The autoencoder learns to reconstruct normal chest X-rays.
At inference time, pathological features cannot be reconstructed
accurately — producing high pixel-wise MSE (the anomaly score).

```
Normal X-ray   → encoder → z → decoder → x̂  (low error)
Pneumonia X-ray → encoder → z → decoder → x̂  (high error = anomaly)
```

## Key results

After training, run `evaluate.py` to get:
- `outputs/roc_curve.png` — AUC-ROC
- `outputs/score_distribution.png` — score separation
- `outputs/latent_space_tsne.png` — learned representation
- `outputs/metrics.csv` — all scalar metrics

## Research note

This is a research prototype. Results must not be used for
clinical diagnosis or treatment decisions.

**Dataset:** Kermany et al. Chest X-Ray Images (Pneumonia).
Kaggle. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**Reference:** Baur et al. (2019). "Deep Autoencoding Models for
Unsupervised Anomaly Segmentation in Brain MR Images." MICCAI Workshop.
