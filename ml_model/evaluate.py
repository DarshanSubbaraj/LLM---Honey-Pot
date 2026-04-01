"""
evaluate.py
-----------
Detailed evaluation script that produces:
  - Classification metrics (precision, recall, F1, ROC-AUC)
  - Confusion matrix plot
  - ROC curve plot
  - Per-sentiment-category breakdown
  - Ranked list of most-scammy features (top TF-IDF + feature importances)

Usage
-----
  python evaluate.py                        # evaluate on held-out split of sample data
  python evaluate.py --csv path/to/data.csv # evaluate on a custom dataset
"""

from __future__ import annotations

import io
import os
import sys

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from data.sample_dataset import build_dataset
from models.scam_detector import ScamDetector
from nlp.preprocessor import FeatureExtractor
from nlp.sentiment_analyzer import SentimentAnalyzer

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _roc_curve_plot(y_true, y_score, path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1976D2", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1976D2")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — Scam Detector", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Evaluate] ROC curve saved -> {path}")


def _sentiment_breakdown(texts, labels, sentiments, analyzer: SentimentAnalyzer) -> pd.DataFrame:
    records = []
    for txt, lbl, true_sent in zip(texts, labels, sentiments):
        pred = analyzer.analyze(txt)
        records.append({
            "text":          txt[:60],
            "true_label":    lbl,
            "true_sentiment":true_sent,
            "pred_sentiment":pred["label"],
            "confidence":    pred["confidence"],
            "correct":       true_sent == pred["label"],
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the scam detection model")
    parser.add_argument("--csv", type=str, help="Path to custom CSV with text, label columns")
    args = parser.parse_args()

    # Load data
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        df = build_dataset()

    texts     = df["text"].tolist()
    labels    = df["label"].tolist()
    sentiments = df.get("sentiment_label", pd.Series(["neutral"] * len(df))).tolist()

    X_train, X_test, y_train, y_test, _, sent_test = train_test_split(
        texts, labels, sentiments, test_size=0.25, stratify=labels, random_state=42
    )

    # Load or train detector
    model_path = os.path.join(os.path.dirname(__file__), "models", "scam_detector.joblib")
    if os.path.exists(model_path):
        print("[Evaluate] Loading saved model …")
        detector = ScamDetector.load(model_path)
    else:
        print("[Evaluate] No saved model found. Training from scratch …")
        detector = ScamDetector()
        detector.fit(X_train, y_train)
        detector.save(model_path)

    # Build test features & get probabilities
    X_feat   = detector._build_features(X_test, fit=False)
    y_prob   = detector.classifier.predict_proba(X_feat)[:, 1]
    y_pred   = (y_prob >= detector.threshold).astype(int)
    y_test_a = np.array(y_test)

    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test_a, y_pred, target_names=["Legitimate", "Scam"]))

    auc = roc_auc_score(y_test_a, y_prob)
    print(f"ROC-AUC: {auc:.4f}")

    # ROC curve
    _roc_curve_plot(y_test_a, y_prob, os.path.join(REPORTS_DIR, "roc_curve.png"))

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test_a, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Scam"], yticklabels=["Legit", "Scam"],
                ax=ax, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix_eval.png"), dpi=150)
    plt.close()

    # Sentiment breakdown
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS BREAKDOWN")
    print("=" * 60)
    analyzer = SentimentAnalyzer()
    sent_df  = _sentiment_breakdown(X_test, y_test, sent_test, analyzer)
    scam_sent = sent_df[sent_df["true_label"] == 1]
    accuracy  = scam_sent["correct"].mean()
    print(f"Sentiment accuracy (scam messages only): {accuracy:.1%}")
    print("\nPer-category counts:")
    print(scam_sent.groupby(["true_sentiment", "pred_sentiment"]).size().to_string())

    # Feature importance plot (Random Forest sub-estimator)
    try:
        rf_est = detector.classifier.named_estimators_["rf"]
        importances = rf_est.feature_importances_
        feat_names  = (
            detector.tfidf.get_feature_names_out().tolist()
            + FeatureExtractor.FEATURE_NAMES
        )
        top_n  = 20
        top_idx = np.argsort(importances)[-top_n:][::-1]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            [feat_names[i] for i in top_idx][::-1],
            importances[top_idx][::-1],
            color="#1976D2", edgecolor="white",
        )
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances (Random Forest)", fontweight="bold")
        plt.tight_layout()
        fi_path = os.path.join(REPORTS_DIR, "feature_importances.png")
        plt.savefig(fi_path, dpi=150)
        plt.close()
        print(f"\n[Evaluate] Feature importances saved -> {fi_path}")
    except Exception as e:
        print(f"[Evaluate] Feature importance plot skipped: {e}")

    print("\n[Evaluate] Done. OK")


if __name__ == "__main__":
    main()
