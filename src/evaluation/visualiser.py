"""
src/evaluation/visualiser.py
─────────────────────────────
Generates all research report figures. Pure I/O: data in → PNG out.
No metric computation, no model code.

Figures:
  1. training_curves.png
  2. score_distribution.png
  3. roc_curve.png
  4. pr_curve.png
  5. confusion_matrix.png
  6. reconstruction_grid.png
  7. latent_space_tsne.png
  8. latent_space_pca.png
  9. heatmap_examples.png  ← was missing, now implemented
"""

import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

import config
from src.utils import get_logger

logger = get_logger(__name__)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.linewidth":    0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.5,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})

BLUE   = "#185FA5"
CORAL  = "#D85A30"
TEAL   = "#0F6E56"
PURPLE = "#534AB7"
GRAY   = "#888780"


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(config.OUTPUT_DIR, filename)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


def _denorm(t: torch.Tensor) -> np.ndarray:
    """Tensor [-1,1] → numpy [0,1]."""
    return np.clip(t.squeeze().cpu().numpy() * 0.5 + 0.5, 0, 1)


# ── 1. Training curves ────────────────────────────────────────────

def plot_training_curves(history: dict) -> str:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Curves", fontsize=13, y=1.02)

    axes[0].plot(epochs, history["train_loss"], color=BLUE,   lw=1.5, label="Train")
    axes[0].plot(epochs, history["val_loss"],   color=CORAL,  lw=1.5, label="Val", ls="--")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Reconstruction Loss")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, history["lr"], color=PURPLE, lw=1.5)
    axes[1].set(xlabel="Epoch", ylabel="LR", title="Learning Rate Schedule")
    axes[1].set_yscale("log")

    plt.tight_layout()
    return _save(fig, "training_curves.png")


# ── 2. Score distribution ─────────────────────────────────────────

def plot_score_distribution(normal_scores: np.ndarray,
                             anomaly_scores: np.ndarray,
                             threshold: float) -> str:
    bins = np.linspace(
        min(normal_scores.min(), anomaly_scores.min()),
        max(normal_scores.max(), anomaly_scores.max()),
        60,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(normal_scores,  bins=bins, alpha=0.65, color=TEAL,  density=True,
            label=f"Normal (n={len(normal_scores)})")
    ax.hist(anomaly_scores, bins=bins, alpha=0.65, color=CORAL, density=True,
            label=f"Anomaly (n={len(anomaly_scores)})")
    ax.axvline(threshold, color=GRAY, lw=1.2, ls="--",
               label=f"Threshold = {threshold:.5f}")
    ax.set(xlabel="Reconstruction Error (MSE)", ylabel="Density",
           title="Score Distribution: Normal vs Anomaly")
    ax.legend(frameon=False)
    plt.tight_layout()
    return _save(fig, "score_distribution.png")


# ── 3. ROC curve ──────────────────────────────────────────────────

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_roc: float) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"AUC = {auc_roc:.4f}")
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--", label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve", xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    return _save(fig, "roc_curve.png")


# ── 4. Precision-Recall curve ─────────────────────────────────────

def plot_pr_curve(prec: np.ndarray, rec: np.ndarray, auc_pr: float) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec, prec, color=CORAL, lw=2, label=f"AP = {auc_pr:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve",
           xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(frameon=False)
    plt.tight_layout()
    return _save(fig, "pr_curve.png")


# ── 5. Confusion matrix ───────────────────────────────────────────

def plot_confusion_matrix(tp: int, tn: int, fp: int, fn: int) -> str:
    total  = tp + tn + fp + fn
    matrix = np.array([[tn, fp], [fn, tp]])
    cell_labels = [
        [f"TN\n{tn}\n({tn/total*100:.1f}%)", f"FP\n{fp}\n({fp/total*100:.1f}%)"],
        [f"FN\n{fn}\n({fn/total*100:.1f}%)", f"TP\n{tp}\n({tp/total*100:.1f}%)"],
    ]
    cmap = LinearSegmentedColormap.from_list("cm", ["#E6F1FB", BLUE])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Normal", "Pred Anomaly"])
    ax.set_yticklabels(["Actual Normal", "Actual Anomaly"])
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "#2C2C2A"
            ax.text(j, i, cell_labels[i][j], ha="center", va="center",
                    fontsize=12, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return _save(fig, "confusion_matrix.png")


# ── 6. Reconstruction grid ────────────────────────────────────────

def plot_reconstruction_grid(originals: torch.Tensor,
                              reconstructions: torch.Tensor,
                              labels: List[int],
                              n_cols: int = 8) -> str:
    n = min(n_cols, originals.size(0))
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.5))
    fig.suptitle("Original (top) vs Reconstruction (bottom)\n"
                 "Green = normal  |  Red = anomaly", fontsize=10)

    for i in range(n):
        color = CORAL if labels[i] == 1 else TEAL
        for row, imgs in enumerate([originals, reconstructions]):
            ax = axes[row, i] if n > 1 else axes[row]
            ax.imshow(_denorm(imgs[i]), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2)

    plt.tight_layout()
    return _save(fig, "reconstruction_grid.png")


# ── 7 & 8. Latent space ───────────────────────────────────────────

def plot_latent_space(vectors: np.ndarray, labels: np.ndarray,
                      method: str = "tsne") -> str:
    from sklearn.preprocessing import StandardScaler
    scaled = StandardScaler().fit_transform(vectors)

    if method == "tsne":
        from sklearn.manifold import TSNE
        logger.info("Running t-SNE (1-2 min)...")
        embedded = TSNE(n_components=2, random_state=config.SEED,
                        perplexity=30, n_iter=1000).fit_transform(scaled)
        title = "Latent Space — t-SNE 2D Projection"
    else:
        from sklearn.decomposition import PCA
        pca      = PCA(n_components=2, random_state=config.SEED)
        embedded = pca.fit_transform(scaled)
        var      = pca.explained_variance_ratio_
        title    = (f"Latent Space — PCA 2D Projection "
                    f"(PC1={var[0]*100:.1f}%, PC2={var[1]*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, color, name in [(0, TEAL, "Normal"), (1, CORAL, "Anomaly")]:
        mask = labels == label
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=color, alpha=0.5, s=12, linewidths=0,
                   label=f"{name} (n={mask.sum()})")
    ax.set(xlabel="Component 1", ylabel="Component 2", title=title)
    ax.legend(frameon=False, markerscale=2)
    plt.tight_layout()
    return _save(fig, f"latent_space_{method}.png")


# ── 9. Heatmap examples ───────────────────────────────────────────

def plot_heatmap_examples(originals: torch.Tensor,
                           reconstructions: torch.Tensor,
                           labels: List[int],
                           n_cols: int = 6) -> str:
    """
    For each image: original | reconstruction | heatmap (error map).
    Red = high reconstruction error = anomalous region.
    """
    n   = min(n_cols, originals.size(0))
    fig, axes = plt.subplots(3, n, figsize=(n * 1.8, 5.5))
    row_titles = ["Original", "Reconstruction", "Error heatmap"]

    for i in range(n):
        orig  = _denorm(originals[i])
        recon = _denorm(reconstructions[i])
        error = (originals[i] - reconstructions[i]).pow(2).squeeze().cpu().numpy()
        error = (error - error.min()) / (error.max() - error.min() + 1e-8)

        color = CORAL if labels[i] == 1 else TEAL

        axes[0, i].imshow(orig,  cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(recon, cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(error, cmap="hot",  vmin=0, vmax=1)

        for row in range(3):
            axes[row, i].axis("off")
            for spine in axes[row, i].spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(1.5)

    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(title, fontsize=10, rotation=90, labelpad=4)

    fig.suptitle("Heatmap examples — green=normal, red=anomaly", fontsize=10)
    plt.tight_layout()
    return _save(fig, "heatmap_examples.png")


# ── All figures in one call ───────────────────────────────────────

def save_all_evaluation_figures(result,
                                 vectors: Optional[np.ndarray] = None,
                                 labels: Optional[np.ndarray]  = None) -> List[str]:
    """Generate and save all evaluation figures. Returns list of saved paths."""
    paths = [
        plot_score_distribution(result.normal_scores, result.anomaly_scores,
                                result.threshold),
        plot_roc_curve(result.fpr, result.tpr, result.auc_roc),
        plot_pr_curve(result.prec_curve, result.rec_curve, result.auc_pr),
        plot_confusion_matrix(result.tp, result.tn, result.fp, result.fn),
    ]
    if vectors is not None and labels is not None:
        paths.append(plot_latent_space(vectors, labels, method="tsne"))
        paths.append(plot_latent_space(vectors, labels, method="pca"))

    logger.info(f"All figures saved to: {config.OUTPUT_DIR}")
    return paths
