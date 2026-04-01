"""
preprocessor.py
---------------
spaCy-powered NLP text preprocessing pipeline.

Responsibilities
----------------
- Tokenise with spaCy (en_core_web_sm)
- Remove stopwords, punctuation, whitespace tokens
- Lemmatise tokens
- Extract rich NLP features:
    * POS tag distribution
    * Named-entity types
    * Sentence count, average sentence length
    * Presence of URLs and monetary amounts
    * Urgency-word density
    * Sentiment polarity / subjectivity (TextBlob fallback)
"""

from __future__ import annotations

import re
import string
from typing import Any

import numpy as np
import spacy
from textblob import TextBlob

# ---------------------------------------------------------------------------
# Load the spaCy model once at module level (en_core_web_sm must be installed:
#   python -m spacy download en_core_web_sm)
# ---------------------------------------------------------------------------
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' not found.\n"
        "Run: python -m spacy download en_core_web_sm"
    )

# ---------------------------------------------------------------------------
# Urgency / manipulation lexicon
# ---------------------------------------------------------------------------
URGENCY_WORDS = {
    "urgent", "immediately", "now", "hurry", "limited", "expire", "expires",
    "final", "notice", "warning", "alert", "attention", "act", "suspended",
    "verify", "confirm", "click", "call", "respond", "account", "risk",
    "penalty", "arrest", "illegal", "blocked", "locked", "compromised",
    "unauthorized", "violation", "deadline", "last", "chance",
}

GREED_WORDS = {
    "win", "won", "winner", "prize", "lottery", "claim", "free", "money",
    "cash", "reward", "bonus", "million", "thousand", "profit", "earn",
    "income", "investment", "guaranteed", "returns", "selected", "lucky",
}

TRUST_WORDS = {
    "friend", "trusted", "personally", "exclusive", "special", "secret",
    "confidential", "mutual", "recommend", "favour", "help",
}

URL_PATTERN = re.compile(r"http[s]?://\S+|www\.\S+", re.IGNORECASE)
MONEY_PATTERN = re.compile(r"\$[\d,]+|\d+[\s]?(million|thousand|USD|GBP)", re.IGNORECASE)
ALL_CAPS_PATTERN = re.compile(r"\b[A-Z]{3,}\b")
EXCLAMATION_PATTERN = re.compile(r"!")


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip URLs for clean token analysis."""
    text = URL_PATTERN.sub(" URL ", text)
    text = text.strip()
    return text


def lemmatise(text: str) -> str:
    """Return a space-joined string of lemmatised, filtered tokens."""
    doc = NLP(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.is_alpha
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Extracts a fixed-length numerical feature vector from raw text using spaCy.

    Feature groups
    --------------
    1.  Content statistics     (5 features)
    2.  Lexical manipulation   (3 features)
    3.  POS tag ratios         (4 features)
    4.  Named-entity flags     (6 features)
    5.  Sentiment signals      (2 features)
    """

    FEATURE_NAMES: list[str] = [
        # --- Content stats ---
        "token_count",
        "sentence_count",
        "avg_sentence_len",
        "url_count",
        "money_count",
        # --- Lexical manipulation ---
        "urgency_density",
        "greed_density",
        "trust_density",
        # --- POS ratios ---
        "noun_ratio",
        "verb_ratio",
        "adj_ratio",
        "adv_ratio",
        # --- NER flags ---
        "has_person",
        "has_org",
        "has_money_ent",
        "has_cardinal",
        "has_gpe",
        "has_date",
        # --- Sentiment (TextBlob fallback) ---
        "textblob_polarity",
        "textblob_subjectivity",
        # --- Surface signals ---
        "caps_word_ratio",
        "exclamation_count",
    ]

    def extract(self, text: str) -> np.ndarray:
        """Return a 1-D numpy array of features for `text`."""
        doc = NLP(text)
        tokens = [t for t in doc if not t.is_space]
        n_tokens = max(len(tokens), 1)
        sentences = list(doc.sents)
        n_sents = max(len(sentences), 1)
        avg_sent_len = n_tokens / n_sents

        # URL / money surface counts
        url_count = len(URL_PATTERN.findall(text))
        money_count = len(MONEY_PATTERN.findall(text))

        # Urgency / greed / trust lexicon density
        lower_tokens = set(t.lower_ for t in tokens if t.is_alpha)
        urgency_density = len(lower_tokens & URGENCY_WORDS) / n_tokens
        greed_density = len(lower_tokens & GREED_WORDS) / n_tokens
        trust_density = len(lower_tokens & TRUST_WORDS) / n_tokens

        # POS ratios
        pos_counts: dict[str, int] = {}
        for token in tokens:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        noun_ratio = pos_counts.get("NOUN", 0) / n_tokens
        verb_ratio = pos_counts.get("VERB", 0) / n_tokens
        adj_ratio = pos_counts.get("ADJ", 0) / n_tokens
        adv_ratio = pos_counts.get("ADV", 0) / n_tokens

        # Named-entity flags
        ent_labels = {ent.label_ for ent in doc.ents}
        has_person = float("PERSON" in ent_labels)
        has_org = float("ORG" in ent_labels)
        has_money_ent = float("MONEY" in ent_labels)
        has_cardinal = float("CARDINAL" in ent_labels)
        has_gpe = float("GPE" in ent_labels)
        has_date = float("DATE" in ent_labels)

        # Sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity          # [-1, 1]
        subjectivity = blob.sentiment.subjectivity  # [0, 1]

        # Surface signals
        caps_words = ALL_CAPS_PATTERN.findall(text)
        caps_word_ratio = len(caps_words) / n_tokens
        exclamation_count = len(EXCLAMATION_PATTERN.findall(text))

        return np.array([
            n_tokens, n_sents, avg_sent_len,
            url_count, money_count,
            urgency_density, greed_density, trust_density,
            noun_ratio, verb_ratio, adj_ratio, adv_ratio,
            has_person, has_org, has_money_ent, has_cardinal, has_gpe, has_date,
            polarity, subjectivity,
            caps_word_ratio, exclamation_count,
        ], dtype=float)

    def batch_extract(self, texts: list[str]) -> np.ndarray:
        """Vectorise feature extraction over a list of texts."""
        return np.vstack([self.extract(t) for t in texts])
