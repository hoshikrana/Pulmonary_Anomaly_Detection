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

import torch

import config
from src.utils      import set_seed, print_device_info, get_logger, get_device
from src.data       import get_dataloaders
from src.model      import ConvAutoencoder
from src.evaluation import (
    AnomalyScorer, compute_metrics, save_metrics_csv, save_thresholds,
    save_all_evaluation_figures, plot_reconstruction_grid, plot_heatmap_examples,
)

logger = get_logger(__name__, log_dir=config.OUTPUT_DIR + "/logs")


def main():
    set_seed(config.SEED)
    print_device_info()

    if not os.path.exists(config.BEST_MODEL_PATH):
        logger.error(f"No checkpoint at {config.BEST_MODEL_PATH}. Run train.py first.")
        sys.exit(1)

    logger.info(f"Loading model from {config.BEST_MODEL_PATH}")
    model  = ConvAutoencoder.load(config.BEST_MODEL_PATH, device=get_device())
    scorer = AnomalyScorer(model)

    logger.info("Loading test data...")
    _, _, test_loader = get_dataloaders()

    logger.info("Scoring test set...")
    scores, labels = scorer.score_loader(test_loader, desc="Test")

    logger.info("Computing metrics...")
    result = compute_metrics(scores, labels)
    save_metrics_csv(result, path=os.path.join(config.OUTPUT_DIR, "metrics.csv"))

    # Save percentile-based thresholds to disk — loaded by web app at startup
    save_thresholds(result)

    # Visual outputs
    logger.info("Generating figures...")
    _save_image_grids(model, test_loader)
    vectors, lat_labels = scorer.extract_latent_vectors(test_loader)
    save_all_evaluation_figures(result, vectors, lat_labels)

    logger.info(
        f"\nDone. AUC-ROC={result.auc_roc:.4f}  "
        f"AUC-PR={result.auc_pr:.4f}  F1={result.f1:.4f}"
    )
    logger.info(f"All outputs in: {config.OUTPUT_DIR}")


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
