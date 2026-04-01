"""
scam_detector.py
----------------
Binary scam / legitimate classifier.

Pipeline
--------
  Raw text
      │
      ▼
  TF-IDF vectoriser  (unigrams + bigrams, max 5000 features)
      │
      ┴──── Feature extractor (22 engineered features from FeatureExtractor)
      │
      ▼
  Combined feature matrix
      │
      ▼
  Ensemble classifier
      ├─ Logistic Regression  (weight: 40%)
      ├─ Random Forest        (weight: 40%)
      └─ Linear SVM           (weight: 20%)
      │
      ▼
  Soft-voted probability → threshold 0.5 → label
"""

from __future__ import annotations

import io
import os
import sys
import warnings
from typing import Any

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from nlp.preprocessor import FeatureExtractor, lemmatise

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Model artefact paths
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "scam_detector.joblib")


# ─────────────────────────────────────────────────────────────────────────────
# ScamDetector
# ─────────────────────────────────────────────────────────────────────────────

class ScamDetector:
    """
    Trains and runs an ensemble scam-detection classifier.

    Attributes
    ----------
    tfidf      : fitted TF-IDF vectoriser
    extractor  : FeatureExtractor for engineered features
    classifier : fitted VotingClassifier ensemble
    threshold  : decision threshold (default 0.5)
    """

    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold
        self.tfidf     = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5_000,
            sublinear_tf=True,
            min_df=1,
        )
        self.extractor = FeatureExtractor()
        self.scaler    = StandardScaler()

        # Individual estimators
        lr  = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs")
        rf  = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                                     random_state=42, n_jobs=-1)
        svm = LinearSVC(C=0.5, class_weight="balanced", max_iter=2000)

        # Soft voting — LinearSVC doesn't support predict_proba so we wrap it with CalibratedClassifierCV
        from sklearn.calibration import CalibratedClassifierCV
        svm_cal = CalibratedClassifierCV(svm, cv=3, method="sigmoid")

        self.classifier = VotingClassifier(
            estimators=[("lr", lr), ("rf", rf), ("svm", svm_cal)],
            voting="soft",
            weights=[0.4, 0.4, 0.2],
        )

        self._is_fitted = False

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_features(self, texts: list[str], fit: bool = False) -> np.ndarray:
        """Combine TF-IDF and engineered features into a dense array."""
        lemmatised = [lemmatise(t) for t in texts]

        if fit:
            tfidf_mat = self.tfidf.fit_transform(lemmatised)
        else:
            tfidf_mat = self.tfidf.transform(lemmatised)

        eng_mat = self.extractor.batch_extract(texts)

        if fit:
            eng_mat = self.scaler.fit_transform(eng_mat)
        else:
            eng_mat = self.scaler.transform(eng_mat)

        # Combine sparse TF-IDF with dense engineered features
        if issparse(tfidf_mat):
            from scipy.sparse import csr_matrix
            combined = np.hstack([tfidf_mat.toarray(), eng_mat])
        else:
            combined = np.hstack([tfidf_mat, eng_mat])

        return combined

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, texts: list[str], labels: list[int]) -> "ScamDetector":
        """Train the detector on (texts, binary labels)."""
        print("[ScamDetector] Building feature matrix …")
        X = self._build_features(texts, fit=True)
        y = np.array(labels)

        print(f"[ScamDetector] Feature matrix shape: {X.shape}")
        print("[ScamDetector] Training ensemble classifier …")
        self.classifier.fit(X, y)
        self._is_fitted = True
        print("[ScamDetector] Training complete. OK")
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(self, texts: list[str], labels: list[int]) -> dict[str, Any]:
        """Return classification metrics on a held-out test set."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        X  = self._build_features(texts, fit=False)
        y  = np.array(labels)
        y_pred_proba = self.classifier.predict_proba(X)[:, 1]
        y_pred       = (y_pred_proba >= self.threshold).astype(int)

        report = classification_report(y, y_pred, target_names=["Legitimate", "Scam"], output_dict=True)
        cm     = confusion_matrix(y, y_pred).tolist()
        auc    = roc_auc_score(y, y_pred_proba)

        return {
            "classification_report": report,
            "confusion_matrix":      cm,
            "roc_auc":               round(auc, 4),
            "f1_macro":              round(f1_score(y, y_pred, average="macro"), 4),
        }

    def cross_validate(self, texts: list[str], labels: list[int], cv: int = 5) -> dict[str, Any]:
        """Run stratified k-fold cross-validation and return mean ± std."""
        print(f"[ScamDetector] Running {cv}-fold cross-validation …")
        X    = self._build_features(texts, fit=True)
        y    = np.array(labels)
        skf  = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.classifier, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
        return {
            "mean_f1_macro": round(scores.mean(), 4),
            "std_f1_macro":  round(scores.std(), 4),
            "fold_scores":   scores.tolist(),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict[str, Any]:
        """
        Predict whether a single message is a scam.

        Returns
        -------
        {
          "label":         "Scam" | "Legitimate",
          "confidence":    float,   # probability of being a scam
          "is_scam":       bool,
          "risk_level":    "HIGH" | "MEDIUM" | "LOW",
          "tfidf_top_tokens": list[str],  # most influential TF-IDF tokens
        }
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        X     = self._build_features([text], fit=False)
        proba = self.classifier.predict_proba(X)[0, 1]
        label = "Scam" if proba >= self.threshold else "Legitimate"

        risk = "HIGH" if proba >= 0.8 else ("MEDIUM" if proba >= 0.5 else "LOW")

        # Top TF-IDF tokens (from LR sub-estimator)
        lemmatised = lemmatise(text)
        top_tokens = self._top_tfidf_tokens(lemmatised, n=5)

        return {
            "label":             label,
            "confidence":        round(float(proba), 4),
            "is_scam":           label == "Scam",
            "risk_level":        risk,
            "tfidf_top_tokens":  top_tokens,
        }

    def _top_tfidf_tokens(self, lemmatised: str, n: int = 5) -> list[str]:
        """Return the top-n highest TF-IDF scoring tokens in the input."""
        tfidf_vec = self.tfidf.transform([lemmatised])
        feature_names = np.array(self.tfidf.get_feature_names_out())
        row = tfidf_vec.toarray()[0]
        top_idx = row.argsort()[-n:][::-1]
        return feature_names[top_idx[row[top_idx] > 0]].tolist()

    # ──────────────────────────────────────────────────────────────────────
    # Persist / load
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[ScamDetector] Model saved -> {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "ScamDetector":
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model at {path}")
        obj = joblib.load(path)
        print(f"[ScamDetector] Model loaded <- {path}")
        return obj
