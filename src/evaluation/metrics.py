"""
src/evaluation/metrics.py
──────────────────────────
Computes all quantitative metrics. Returns EvaluationResult dataclass.
Thresholds are stored to disk — never hardcoded in inference.py.

Metrics:
  AUC-ROC     — primary; threshold-independent ranking metric
  AUC-PR      — secondary; better for imbalanced test sets
  Threshold   — via Youden's J: maximises sensitivity + specificity
  F1, Precision, Recall, Specificity, Accuracy at optimal threshold
"""

import json
import os
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix,
    f1_score, precision_score, recall_score,
)

import config


@dataclass
class EvaluationResult:
    """All evaluation metrics in one typed object."""

    # Core metrics
    auc_roc:     float = 0.0
    auc_pr:      float = 0.0

    # Optimal threshold (Youden's J)
    threshold:   float = 0.0

    # At optimal threshold
    accuracy:    float = 0.0
    f1:          float = 0.0
    precision:   float = 0.0
    recall:      float = 0.0
    specificity: float = 0.0

    # Confusion matrix
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    # Curve arrays (for plotting)
    fpr:        np.ndarray = field(default_factory=lambda: np.array([]))
    tpr:        np.ndarray = field(default_factory=lambda: np.array([]))
    thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    prec_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    rec_curve:  np.ndarray = field(default_factory=lambda: np.array([]))

    # Score distributions (for histogram)
    normal_scores:  np.ndarray = field(default_factory=lambda: np.array([]))
    anomaly_scores: np.ndarray = field(default_factory=lambda: np.array([]))

    def log_summary(self) -> None:
        print("=" * 50)
        print("  Evaluation Results")
        print("=" * 50)
        print(f"  AUC-ROC     : {self.auc_roc:.4f}")
        print(f"  AUC-PR      : {self.auc_pr:.4f}")
        print(f"  Threshold   : {self.threshold:.6f}")
        print(f"  Accuracy    : {self.accuracy:.4f}")
        print(f"  F1          : {self.f1:.4f}")
        print(f"  Precision   : {self.precision:.4f}")
        print(f"  Recall      : {self.recall:.4f}")
        print(f"  Specificity : {self.specificity:.4f}")
        print(f"  TP={self.tp} TN={self.tn} FP={self.fp} FN={self.fn}")
        print("=" * 50)

    def to_dict(self) -> dict:
        return {
            "auc_roc":     round(self.auc_roc,     4),
            "auc_pr":      round(self.auc_pr,       4),
            "threshold":   round(self.threshold,    6),
            "accuracy":    round(self.accuracy,     4),
            "f1":          round(self.f1,           4),
            "precision":   round(self.precision,    4),
            "recall":      round(self.recall,       4),
            "specificity": round(self.specificity,  4),
            "tp": self.tp, "tn": self.tn, "fp": self.fp, "fn": self.fn,
        }


class MetricsCalculator:
    """Computes all metrics from anomaly scores and ground-truth labels."""

    def __init__(self, scores: np.ndarray, labels: np.ndarray):
        if len(scores) != len(labels):
            raise ValueError(f"Length mismatch: scores={len(scores)}, labels={len(labels)}")
        self.scores = scores.astype(np.float32)
        self.labels = labels.astype(np.int32)

    def compute(self) -> EvaluationResult:
        r = EvaluationResult()

        # Threshold-independent
        r.auc_roc = float(roc_auc_score(self.labels, self.scores))
        r.auc_pr  = float(average_precision_score(self.labels, self.scores))

        # ROC curve + Youden's J threshold
        fpr, tpr, thresholds = roc_curve(self.labels, self.scores)
        r.fpr, r.tpr, r.thresholds = fpr, tpr, thresholds
        best_idx     = int(np.argmax(tpr - fpr))
        r.threshold  = float(thresholds[best_idx])

        # PR curve
        prec, rec, _ = precision_recall_curve(self.labels, self.scores)
        r.prec_curve = prec
        r.rec_curve  = rec

        # Metrics at threshold
        preds = (self.scores >= r.threshold).astype(np.int32)
        cm    = confusion_matrix(self.labels, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        r.tp, r.tn, r.fp, r.fn = int(tp), int(tn), int(fp), int(fn)
        r.accuracy    = float((tp + tn) / (tp + tn + fp + fn + 1e-8))
        r.f1          = float(f1_score(self.labels, preds, zero_division=0))
        r.precision   = float(precision_score(self.labels, preds, zero_division=0))
        r.recall      = float(recall_score(self.labels, preds, zero_division=0))
        r.specificity = float(tn / (tn + fp + 1e-8))

        # Score distributions
        r.normal_scores  = self.scores[self.labels == 0]
        r.anomaly_scores = self.scores[self.labels == 1]

        return r


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> EvaluationResult:
    """Compute, log, and return all metrics."""
    result = MetricsCalculator(scores, labels).compute()
    result.log_summary()
    return result


def save_metrics_csv(result: EvaluationResult, path: str) -> None:
    """Save scalar metrics to CSV for the research report."""
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = result.to_dict()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=d.keys())
        w.writeheader()
        w.writerow(d)
    print(f"Metrics CSV saved: {path}")


def save_thresholds(result: EvaluationResult) -> None:
    """
    Persist data-driven thresholds to disk.

    Uses percentile-based thresholds derived from the actual score
    distributions — not arbitrary multiples of the Youden threshold.

    THRESHOLD_NORMAL  = 50th percentile of normal scores
    THRESHOLD_ANOMALY = 95th percentile of normal scores

    The web app loads these at startup from THRESHOLDS_PATH.
    They are never hardcoded.
    """
    thresh = {
        "youden":           round(float(result.threshold),                    6),
        "threshold_normal": round(float(np.percentile(result.normal_scores, 50)), 6),
        "threshold_anomaly":round(float(np.percentile(result.normal_scores, 95)), 6),
    }
    os.makedirs(os.path.dirname(config.THRESHOLDS_PATH), exist_ok=True)
    with open(config.THRESHOLDS_PATH, "w") as f:
        json.dump(thresh, f, indent=2)
    print(f"Thresholds saved: {config.THRESHOLDS_PATH}")
    print(f"  threshold_normal  = {thresh['threshold_normal']}")
    print(f"  threshold_anomaly = {thresh['threshold_anomaly']}")
    print(f"  youden            = {thresh['youden']}")
