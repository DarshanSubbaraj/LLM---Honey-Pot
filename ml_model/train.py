"""
train.py
--------
End-to-end training script.

Usage
-----
  python train.py                     # train on built-in sample dataset
  python train.py --csv path/to/data.csv   # train on custom CSV
  python train.py --eval              # run with test-set evaluation

CSV format (custom data)
------------------------
  text   (str)  - the message
  label  (int)  - 1 = scam, 0 = legitimate

Output
------
  models/scam_detector.joblib   - saved model
  reports/training_report.txt   - metrics
  reports/confusion_matrix.png  - visualisation
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import textwrap

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# ── Path setup so imports work from any cwd ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data.sample_dataset import build_dataset
from models.scam_detector import ScamDetector
from nlp.sentiment_analyzer import SentimentAnalyzer

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(cm: list[list[int]], labels: list[str], path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Train] Confusion matrix saved -> {path}")


def _plot_label_distribution(df: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#4CAF50", "#F44336"]
    df["label"].value_counts().rename({0: "Legitimate", 1: "Scam"}).plot(
        kind="bar", ax=axes[0], color=colors, edgecolor="white", width=0.5
    )
    axes[0].set_title("Class Distribution", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    sent_counts = df[df["label"] == 1]["sentiment_label"].value_counts()
    sent_counts.plot(kind="bar", ax=axes[1],
                     color=["#E91E63", "#FF9800", "#9C27B0", "#607D8B"],
                     edgecolor="white", width=0.5)
    axes[1].set_title("Scam Sentiment Types", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle("Dataset Overview", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Train] Distribution plot saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train the scam detection + sentiment analysis model"
    )
    parser.add_argument("--csv",  type=str, help="Path to custom CSV dataset")
    parser.add_argument("--eval", action="store_true", help="Evaluate on held-out test set")
    parser.add_argument("--cv",   type=int, default=0,
                        help="Run k-fold cross-validation with k folds (0 = skip)")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Decision threshold for scam probability (default 0.50)")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.csv:
        print(f"[Train] Loading dataset from {args.csv} …")
        df = pd.read_csv(args.csv)
        assert {"text", "label"}.issubset(df.columns), "CSV must have 'text' and 'label' columns."
        if "sentiment_label" not in df.columns:
            df["sentiment_label"] = "neutral"
    else:
        print("[Train] Using built-in sample dataset …")
        df = build_dataset()

    print(f"[Train] Dataset size: {len(df)} rows | Scam: {df['label'].sum()} | Legit: {(df['label'] == 0).sum()}")

    # ── Visualise distribution ────────────────────────────────────────────────
    dist_path = os.path.join(REPORTS_DIR, "dataset_distribution.png")
    _plot_label_distribution(df, dist_path)

    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, stratify=labels, random_state=42
    )

    # ── Fit ───────────────────────────────────────────────────────────────────
    detector = ScamDetector(threshold=args.threshold)
    detector.fit(X_train, y_train)

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv_results: dict = {}
    if args.cv > 1:
        cv_results = detector.cross_validate(X_train, y_train, cv=args.cv)
        print(f"[Train] CV F1-macro: {cv_results['mean_f1_macro']:.4f} ± {cv_results['std_f1_macro']:.4f}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_results: dict = {}
    if args.eval:
        eval_results = detector.evaluate(X_test, y_test)
        print("\n[Train] === Test-set evaluation ===")
        print(f"  ROC-AUC   : {eval_results['roc_auc']}")
        print(f"  F1 (macro): {eval_results['f1_macro']}")
        cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        _plot_confusion_matrix(
            eval_results["confusion_matrix"],
            ["Legitimate", "Scam"],
            cm_path,
        )

    # ── Sentiment demo on test messages ──────────────────────────────────────
    analyzer = SentimentAnalyzer()
    print("\n[Train] === Sentiment Analysis demo (first 5 test samples) ===")
    for txt in X_test[:5]:
        result = analyzer.analyze(txt)
        print(f"  [{result['label'].upper():8s}] {txt[:70]!r}")

    # ── Save model ────────────────────────────────────────────────────────────
    detector.save()

    # ── Write text report ─────────────────────────────────────────────────────
    report_path = os.path.join(REPORTS_DIR, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Scam Detection Model — Training Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset size : {len(df)} rows\n")
        f.write(f"Scam samples : {df['label'].sum()}\n")
        f.write(f"Legit samples: {(df['label'] == 0).sum()}\n")
        f.write(f"Train / Test : {len(X_train)} / {len(X_test)}\n")
        f.write(f"Threshold    : {args.threshold}\n\n")

        if cv_results:
            f.write(f"Cross-validation ({args.cv}-fold)\n")
            f.write(f"  Mean F1-macro: {cv_results['mean_f1_macro']}\n")
            f.write(f"  Std  F1-macro: {cv_results['std_f1_macro']}\n")
            f.write(f"  Per-fold     : {cv_results['fold_scores']}\n\n")

        if eval_results:
            f.write("Test-set metrics\n")
            f.write(f"  ROC-AUC   : {eval_results['roc_auc']}\n")
            f.write(f"  F1 (macro): {eval_results['f1_macro']}\n\n")
            import json
            f.write("Classification report:\n")
            cr = eval_results["classification_report"]
            for cls, vals in cr.items():
                if isinstance(vals, dict):
                    f.write(f"  {cls}: precision={vals['precision']:.4f} "
                            f"recall={vals['recall']:.4f} "
                            f"f1={vals['f1-score']:.4f}\n")

    print(f"[Train] Report saved -> {report_path}")
    print("[Train] Done. OK")


if __name__ == "__main__":
    main()
