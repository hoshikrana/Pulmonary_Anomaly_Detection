"""
scripts/evaluate.py — evaluation entry point.

Run:  python scripts/evaluate.py

Outputs (outputs/):
  training_curves.png, score_distribution.png, roc_curve.png,
  pr_curve.png, confusion_matrix.png, reconstruction_grid.png,
  heatmap_examples.png, latent_space_tsne.png, latent_space_pca.png,
  metrics.csv, thresholds.json
"""

import os
import sys

# Ensure the project root is in the Python path before importing config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

import config
from src.utils      import set_seed, print_device_info, get_device
from src.data       import get_dataloaders
from src.model      import ConvAutoencoder
from src.evaluation import (
    AnomalyScorer, compute_metrics, save_metrics_csv, save_thresholds,
    save_all_evaluation_figures, plot_reconstruction_grid, plot_heatmap_examples,
)


def main():
    print("\n" + "=" * 70)
    print("  PULMONARY ANOMALY DETECTION - EVALUATION SCRIPT")
    print("=" * 70)
    print(f"  Random Seed: {config.SEED}")
    print("=" * 70)

    set_seed(config.SEED)
    print_device_info()

    print("\n[1/6] Checking for trained model...")
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"❌ ERROR: No checkpoint found at {config.BEST_MODEL_PATH}")
        print("   Please run training first: python scripts/train.py --egx")
        sys.exit(1)
    print(f"✓ Found model checkpoint: {config.BEST_MODEL_PATH}")

    print("\n[2/6] Loading trained model...")
    model = ConvAutoencoder.load(config.BEST_MODEL_PATH, device=get_device())
    print("✓ Model loaded successfully")

    print("\n[3/6] Initializing anomaly scorer...")
    scorer = AnomalyScorer(model)
    print("✓ Anomaly scorer ready")

    print("\n[4/6] Loading test dataset...")
    _, _, test_loader = get_dataloaders()
    print("✓ Test data loaded")

    print("\n[5/6] Scoring test set...")
    scores, labels = scorer.score_loader(test_loader, desc="Test")
    print(f"✓ Scored {len(scores)} test samples")

    print("\n[6/6] Computing evaluation metrics...")
    result = compute_metrics(scores, labels)
    print("✓ Metrics computed")

    print("\n[Saving] Metrics CSV...")
    save_metrics_csv(result, path=os.path.join(config.OUTPUT_DIR, "metrics.csv"))
    print("✓ Metrics saved to CSV")

    print("\n[Saving] Thresholds for inference...")
    save_thresholds(result)
    print("✓ Thresholds saved for web app")

    print("\n[Generating] Evaluation figures...")
    _save_image_grids(model, test_loader)
    vectors, lat_labels = scorer.extract_latent_vectors(test_loader)
    save_all_evaluation_figures(result, vectors, lat_labels)
    print("✓ All figures generated")

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETED")
    print("=" * 70)
    print(f"  AUC-ROC: {result.auc_roc:.4f}")
    print(f"  AUC-PR:  {result.auc_pr:.4f}")
    print(f"  F1 Score: {result.f1:.4f}")
    print(f"  All outputs saved to: {config.OUTPUT_DIR}")
    print("=" * 70)


def _save_image_grids(model, test_loader):
    """Save reconstruction grid and heatmap examples."""
    device     = get_device()
    images, labels = next(iter(test_loader))
    images     = images.to(device)
    with torch.no_grad():
        x_hat, _ = model(images)

    plot_reconstruction_grid(
        originals=images[:8].cpu(),
        reconstructions=x_hat[:8].cpu(),
        labels=labels[:8].tolist(),
    )
    plot_heatmap_examples(
        originals=images[:6].cpu(),
        reconstructions=x_hat[:6].cpu(),
        labels=labels[:6].tolist(),
    )


if __name__ == "__main__":
    main()
