"""
sentiment_analyzer.py
---------------------
Determines the *emotional strategy* (sentiment category) a scammer is using.

Categories
----------
  urgent   - Fear / pressure tactics ("Act NOW or your account will be closed!")
  greedy   - Financial lure / prize bait ("You won $1,000,000!")
  friendly - False trust / rapport building ("I chose you personally...")
  neutral  - No clear manipulation (used for legitimate messages too)

Methods used
------------
1. Rule-based lexicon classifier  (fast, interpretable)
2. spaCy dependency + POS patterns (structural analysis)
3. TextBlob polarity/subjectivity  (auxiliary signal)

The final sentiment is determined by a *weighted voting* scheme.
"""

from __future__ import annotations

import spacy
from textblob import TextBlob

from nlp.preprocessor import (
    NLP,
    URGENCY_WORDS,
    GREED_WORDS,
    TRUST_WORDS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sentiment categories
# ─────────────────────────────────────────────────────────────────────────────
SENTIMENTS = ["urgent", "greedy", "friendly", "neutral"]

# Sentiment descriptors for human-readable output
SENTIMENT_DESCRIPTIONS = {
    "urgent":   "High-pressure / fear-driven language",
    "greedy":   "Financial lure / prize bait",
    "friendly": "False trust-building / rapport",
    "neutral":  "No strong emotional manipulation",
}


# ─────────────────────────────────────────────────────────────────────────────
# Lexicon-based scorer
# ─────────────────────────────────────────────────────────────────────────────

def _lexicon_scores(tokens: list[str]) -> dict[str, float]:
    """
    Count keyword hits from each manipulation lexicon and normalise.
    Returns a dict {sentiment_category: score}.
    """
    lower = set(tokens)
    urgency_hits = len(lower & URGENCY_WORDS)
    greed_hits   = len(lower & GREED_WORDS)
    trust_hits   = len(lower & TRUST_WORDS)
    total        = max(urgency_hits + greed_hits + trust_hits, 1)

    return {
        "urgent":   urgency_hits / total,
        "greedy":   greed_hits   / total,
        "friendly": trust_hits   / total,
        "neutral":  0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Structural / syntactic scorer
# ─────────────────────────────────────────────────────────────────────────────

def _structural_scores(doc: spacy.tokens.Doc) -> dict[str, float]:
    """
    Use spaCy POS/dependency patterns to detect manipulation structures.

    Urgency signals
    ---------------
      - Imperative verbs (VERB with ROOT and no subject)
      - Very short sentences (≤ 5 tokens)

    Greed signals
    -------------
      - Money named-entities
      - CARDINAL entities (large numbers)

    Friendly signals
    ----------------
      - First-person pronouns (I, we, me, my)
      - Second-person pronouns (you, your)
    """
    scores: dict[str, float] = {"urgent": 0.0, "greedy": 0.0, "friendly": 0.0, "neutral": 0.0}

    ent_labels = {ent.label_ for ent in doc.ents}
    n_sents    = max(len(list(doc.sents)), 1)

    # Urgency: imperative verbs (ROOT verb + no nsubj dependent)
    imperative_count = 0
    short_sent_count = 0
    for sent in doc.sents:
        if len(list(sent)) <= 5:
            short_sent_count += 1
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                has_subject = any(child.dep_ in ("nsubj", "nsubjpass") for child in token.children)
                if not has_subject:
                    imperative_count += 1

    scores["urgent"] += (imperative_count / n_sents) * 0.5
    scores["urgent"] += (short_sent_count / n_sents) * 0.3

    # Greed: money/cardinal entities
    if "MONEY" in ent_labels:
        scores["greedy"] += 0.5
    if "CARDINAL" in ent_labels:
        scores["greedy"] += 0.3
    if "PERCENT" in ent_labels:
        scores["greedy"] += 0.2

    # Friendly: pronoun patterns
    first_person  = {"i", "we", "me", "my", "our", "us", "myself"}
    second_person = {"you", "your", "yourself", "you'd", "you'll", "you've"}
    lower_tokens  = [t.lower_ for t in doc]
    fp_count = sum(1 for t in lower_tokens if t in first_person)
    sp_count = sum(1 for t in lower_tokens if t in second_person)
    n_tok    = max(len(lower_tokens), 1)
    scores["friendly"] += min(fp_count / n_tok, 0.3)
    scores["friendly"] += min(sp_count / n_tok, 0.2)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# TextBlob auxiliary scorer
# ─────────────────────────────────────────────────────────────────────────────

def _textblob_scores(text: str) -> dict[str, float]:
    """
    Map TextBlob polarity/subjectivity to rough sentiment signals.
      - Very negative polarity → urgent (threat / fear)
      - Very positive polarity → greedy (prize / opportunity)
      - High subjectivity → friendly (personal / emotional language)
    """
    blob        = TextBlob(text)
    polarity    = blob.sentiment.polarity       # [-1, 1]
    subjectivity = blob.sentiment.subjectivity  # [0, 1]

    scores: dict[str, float] = {"urgent": 0.0, "greedy": 0.0, "friendly": 0.0, "neutral": 0.0}

    if polarity < -0.2:
        scores["urgent"]   = min(abs(polarity), 1.0) * 0.4
    if polarity > 0.3:
        scores["greedy"]   = min(polarity, 1.0) * 0.4
    if subjectivity > 0.5:
        scores["friendly"] = (subjectivity - 0.5) * 0.4

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Main SentimentAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Analyses the emotional manipulation strategy embedded in a piece of text.

    Usage
    -----
        analyzer = SentimentAnalyzer()
        result   = analyzer.analyze("Your account will be suspended NOW!")
        print(result)
    """

    # Weights for the three scoring methods
    WEIGHTS = {
        "lexicon":    0.50,
        "structural": 0.35,
        "textblob":   0.15,
    }

    def analyze(self, text: str) -> dict[str, Any]:
        """
        Returns a rich result dict:
        {
          "label":       str,        # dominant sentiment category
          "description": str,        # human-readable label description
          "confidence":  float,      # score of winning category [0, 1]
          "scores":      dict,       # {category: blended_score}
          "is_scam_sentiment": bool, # True if NOT neutral
          "textblob_polarity":  float,
          "textblob_subjectivity": float,
          "urgent_keywords": list[str],
          "greed_keywords":  list[str],
          "trust_keywords":  list[str],
        }
        """
        doc    = NLP(text)
        tokens = [t.lower_ for t in doc if t.is_alpha]

        lex_s  = _lexicon_scores(tokens)
        str_s  = _structural_scores(doc)
        tb_s   = _textblob_scores(text)
        blob   = TextBlob(text)

        # Weighted blend
        blended: dict[str, float] = {}
        for cat in SENTIMENTS:
            blended[cat] = (
                self.WEIGHTS["lexicon"]    * lex_s.get(cat, 0)
                + self.WEIGHTS["structural"] * str_s.get(cat, 0)
                + self.WEIGHTS["textblob"]   * tb_s.get(cat, 0)
            )

        # If no manipulation signal is found → neutral
        max_score = max(blended.values())
        if max_score < 0.05:
            label = "neutral"
        else:
            label = max(blended, key=blended.__getitem__)

        # Keyword evidence
        lower_set      = set(tokens)
        urgent_kws     = sorted(lower_set & URGENCY_WORDS)
        greed_kws      = sorted(lower_set & GREED_WORDS)
        trust_kws      = sorted(lower_set & TRUST_WORDS)

        return {
            "label":                  label,
            "description":            SENTIMENT_DESCRIPTIONS[label],
            "confidence":             round(blended.get(label, 0.0), 4),
            "scores":                 {k: round(v, 4) for k, v in blended.items()},
            "is_scam_sentiment":      label != "neutral",
            "textblob_polarity":      round(blob.sentiment.polarity, 4),
            "textblob_subjectivity":  round(blob.sentiment.subjectivity, 4),
            "urgent_keywords":        urgent_kws,
            "greed_keywords":         greed_kws,
            "trust_keywords":         trust_kws,
        }


# allow direct import of Any without re-importing typing
from typing import Any
