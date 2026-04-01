"""
predict.py
----------
Unified inference interface for the Honey-Pot scam detection system.

Combines:
  1. ScamDetector  — is this message a scam?
  2. SentimentAnalyzer — what manipulation strategy is being used?

Usage (CLI)
-----------
  python predict.py "Congratulations! You've won $1,000,000!"
  python predict.py --file messages.txt
  python predict.py --interactive

Output example
--------------
  ╔══════════════════════════════════════════════╗
  ║  SCAM DETECTION REPORT                       ║
  ╠══════════════════════════════════════════════╣
  ║  Verdict      : ⚠  SCAM  (HIGH RISK)         ║
  ║  Confidence   : 0.9412                       ║
  ║  Sentiment    : GREEDY                       ║
  ║  Description  : Financial lure / prize bait  ║
  ║  Top tokens   : congratulations, win, prize  ║
  ╚══════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from models.scam_detector import ScamDetector
from nlp.sentiment_analyzer import SentimentAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────

def _load_detector() -> ScamDetector:
    model_path = os.path.join(os.path.dirname(__file__), "models", "scam_detector.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Trained model not found!\n"
            "Run: python train.py --eval\n"
            "to train and save the model first."
        )
    return ScamDetector.load(model_path)


_detector  = None
_analyzer  = SentimentAnalyzer()


def _get_detector() -> ScamDetector:
    global _detector
    if _detector is None:
        _detector = _load_detector()
    return _detector


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────

RISK_ICONS = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}
VERDICT_ICONS = {"Scam": "⚠️  SCAM", "Legitimate": "✅  LEGITIMATE"}


def analyze_message(text: str) -> dict:
    """
    Run full scam + sentiment analysis on a message.

    Returns
    -------
    A merged dict of detection and sentiment results:
    {
      "text":                 str,
      "label":                "Scam" | "Legitimate",
      "is_scam":              bool,
      "confidence":           float,
      "risk_level":           "HIGH" | "MEDIUM" | "LOW",
      "tfidf_top_tokens":     list[str],
      "sentiment_label":      str,
      "sentiment_description":str,
      "sentiment_scores":     dict,
      "textblob_polarity":    float,
      "textblob_subjectivity":float,
      "urgent_keywords":      list[str],
      "greed_keywords":       list[str],
      "trust_keywords":       list[str],
    }
    """
    detector  = _get_detector()
    det_result = detector.predict(text)
    sent_result = _analyzer.analyze(text)

    return {
        "text":                  text,
        # Detection
        "label":                 det_result["label"],
        "is_scam":               det_result["is_scam"],
        "confidence":            det_result["confidence"],
        "risk_level":            det_result["risk_level"],
        "tfidf_top_tokens":      det_result["tfidf_top_tokens"],
        # Sentiment
        "sentiment_label":       sent_result["label"],
        "sentiment_description": sent_result["description"],
        "sentiment_scores":      sent_result["scores"],
        "is_scam_sentiment":     sent_result["is_scam_sentiment"],
        "textblob_polarity":     sent_result["textblob_polarity"],
        "textblob_subjectivity": sent_result["textblob_subjectivity"],
        "urgent_keywords":       sent_result["urgent_keywords"],
        "greed_keywords":        sent_result["greed_keywords"],
        "trust_keywords":        sent_result["trust_keywords"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────────────────────

def pretty_print(result: dict) -> None:
    width = 62
    border = "═" * width
    print(f"\n╔{border}╗")
    print(f"║{'  SCAM DETECTION & SENTIMENT REPORT':^{width}}║")
    print(f"╠{border}╣")

    icon    = RISK_ICONS.get(result["risk_level"], "")
    verdict = VERDICT_ICONS.get(result["label"], result["label"])
    risk    = result["risk_level"]

    def row(label: str, value: str) -> None:
        line = f"  {label:<20} {value}"
        print(f"║{line:<{width}}║")

    row("Verdict", f"{icon}  {verdict}  [{risk} RISK]")
    row("Confidence", f"{result['confidence']:.4f}")
    row("Sentiment", result["sentiment_label"].upper())
    row("Description", result["sentiment_description"])

    if result["tfidf_top_tokens"]:
        row("Top tokens", ", ".join(result["tfidf_top_tokens"]))

    kws: list[str] = []
    kws += [f"[U] {k}" for k in result["urgent_keywords"][:3]]
    kws += [f"[G] {k}" for k in result["greed_keywords"][:3]]
    kws += [f"[T] {k}" for k in result["trust_keywords"][:3]]
    if kws:
        row("Keywords", "  ".join(kws))

    row("Polarity", f"{result['textblob_polarity']:+.3f} | Subjectivity: {result['textblob_subjectivity']:.3f}")
    print(f"╚{border}╝")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse a message for scam and emotional manipulation"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("message",       nargs="?", help="Message text to analyse")
    group.add_argument("--file",        type=str,  help="Path to text file with one message per line")
    group.add_argument("--interactive", action="store_true", help="Interactive mode (type messages)")
    args = parser.parse_args()

    if args.message:
        result = analyze_message(args.message)
        pretty_print(result)

    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            result = analyze_message(line)
            pretty_print(result)

    elif args.interactive:
        print("=" * 60)
        print("  Scam Detector — Interactive Mode")
        print("  Type a message and press Enter. Type 'quit' to exit.")
        print("=" * 60)
        while True:
            try:
                text = input("\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            result = analyze_message(text)
            pretty_print(result)


if __name__ == "__main__":
    main()
