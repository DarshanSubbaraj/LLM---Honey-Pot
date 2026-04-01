# 🕵️ LLM Honey-Pot — Scam Detection & Sentiment Analysis

> A production-ready NLP pipeline that **detects scam messages** and **analyses the emotional manipulation strategy** (sentiment) used by the scammer — built with **spaCy**, **scikit-learn**, and **TextBlob**.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![spaCy](https://img.shields.io/badge/spaCy-3.7%2B-09a3d5?logo=spaCy)](https://spacy.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Architecture](#architecture)
4. [Features](#features)
5. [Installation](#installation)
6. [Training the Model](#training-the-model)
7. [Evaluating the Model](#evaluating-the-model)
8. [Running Predictions](#running-predictions)
9. [Using as a Python Library](#using-as-a-python-library)
10. [Training on Custom Data](#training-on-custom-data)
11. [Sentiment Categories](#sentiment-categories)
12. [Feature Engineering](#feature-engineering)
13. [Model Performance](#model-performance)
14. [Deployment Guide](#deployment-guide)
15. [Troubleshooting](#troubleshooting)

---

## Overview

This ML model is part of the **LLM Honey-Pot** project — a system for detecting and analysing scammer behaviour. It does two things:

| Task | Output |
|------|--------|
| **Scam Detection** | `Scam` / `Legitimate` + confidence score + risk level (`HIGH`/`MEDIUM`/`LOW`) |
| **Sentiment Analysis** | `urgent` / `greedy` / `friendly` / `neutral` — the scammer's manipulation strategy |

---

## Project Structure

```
LLM---Honey-Pot/
├── ml_model/
│   ├── data/
│   │   ├── __init__.py
│   │   └── sample_dataset.py        # 60-sample built-in labelled dataset
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── preprocessor.py          # spaCy tokeniser + 22-feature extractor
│   │   └── sentiment_analyzer.py   # 3-method sentiment engine
│   ├── models/
│   │   ├── __init__.py
│   │   └── scam_detector.py         # Ensemble classifier (LR + RF + SVM)
│   ├── reports/                     # Auto-generated plots & metrics (after training)
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── feature_importances.png
│   │   ├── dataset_distribution.png
│   │   └── training_report.txt
│   ├── train.py                     # End-to-end training script
│   ├── evaluate.py                  # Detailed evaluation with plots
│   ├── predict.py                   # Inference CLI
│   ├── __init__.py
│   ├── requirements.txt
│   └── README.md
├── .gitignore
└── ml model for sentiment analyis scam detetion   # Original notes file
```

---

## Architecture

```
          Raw Message Text
                │
    ┌───────────▼───────────┐
    │  spaCy NLP Pipeline   │   tokenise · POS tag · NER · dependency parse
    └───────────┬───────────┘
                │
       ┌────────┴──────────┐
       │                   │
  TF-IDF Vectoriser     22 Engineered Features
  (1-2 grams,           Content  : token count, sentence len, URLs, money refs
   5000 features,       Lexical  : urgency / greed / trust keyword density
   lemmatised)          POS      : noun / verb / adj / adv ratio
                        NER      : PERSON, ORG, MONEY, CARDINAL, GPE, DATE
                        Sentiment: TextBlob polarity & subjectivity
                        Surface  : ALL-CAPS ratio, exclamation count
       │                   │
       └────────┬───────────┘
                │
    ┌───────────▼──────────────────────────────┐
    │           Ensemble Classifier             │
    │   Logistic Regression     (40% weight)   │
    │   Random Forest           (40% weight)   │
    │   Linear SVM (calibrated) (20% weight)   │
    │         Soft-voted probability            │
    └──────┬───────────────────────────────────┘
           │
    ┌──────▼──────────┐        ┌─────────────────────────────┐
    │  Scam Detector  │        │     Sentiment Analyzer       │
    │  Scam/Legit     │        │  urgent/greedy/friendly/     │
    │  HIGH/MED/LOW   │        │  neutral + confidence        │
    └─────────────────┘        └─────────────────────────────┘
```

---

## Features

- **spaCy NLP pipeline** — tokenisation, lemmatisation, POS tagging, NER, dependency parsing
- **22 hand-engineered features** — urgency/greed/trust lexicons, POS ratios, named-entity flags, surface signals
- **TF-IDF** with unigrams + bigrams (5 000 features, lemmatised, sublinear TF)
- **Ensemble classifier** — Logistic Regression + Random Forest + SVM with soft voting
- **Multi-method sentiment** — lexicon scoring + spaCy structural patterns + TextBlob polarity
- **Stratified k-fold cross-validation** with F1-macro reporting
- **Full evaluation suite** — confusion matrix, ROC curve, feature importance plots
- **Interactive CLI** — single message, batch file, or interactive REPL mode

---

## Installation

### Prerequisites

- Python **3.9 or higher**
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/DarshanSubbaraj/LLM---Honey-Pot.git
cd LLM---Honey-Pot/ml_model
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

> **Windows users:** Always run Python scripts with the `-X utf8` flag to avoid encoding errors:
> ```bash
> python -X utf8 train.py --eval
> ```

---

## Training the Model

Run from inside the `ml_model/` directory:

```bash
# Basic training (uses built-in 60-sample dataset)
python -X utf8 train.py

# Train + evaluate on a held-out test set
python -X utf8 train.py --eval

# Train + evaluate + 5-fold cross-validation
python -X utf8 train.py --eval --cv 5

# Custom decision threshold (default is 0.50)
python -X utf8 train.py --eval --threshold 0.45

# Train on your own CSV dataset
python -X utf8 train.py --csv path/to/your_data.csv --eval --cv 5
```

**What `train.py` produces:**

| Output | Location |
|--------|----------|
| Saved model | `models/scam_detector.joblib` |
| Training report | `reports/training_report.txt` |
| Confusion matrix | `reports/confusion_matrix.png` |
| Dataset distribution plot | `reports/dataset_distribution.png` |

> **Note:** You must train the model before running `predict.py` or `evaluate.py`.

---

## Evaluating the Model

```bash
python -X utf8 evaluate.py

# Evaluate on a custom CSV
python -X utf8 evaluate.py --csv path/to/your_data.csv
```

**What `evaluate.py` produces:**

| Output | Location |
|--------|----------|
| Classification report (console) | Terminal |
| ROC-AUC score (console) | Terminal |
| Confusion matrix | `reports/confusion_matrix_eval.png` |
| ROC curve | `reports/roc_curve.png` |
| Feature importances | `reports/feature_importances.png` |
| Sentiment breakdown | Terminal |

---

## Running Predictions

### Single message

```bash
python -X utf8 predict.py "Congratulations! You have won $1,000,000. Click here to claim your prize!"
```

**Example output:**

```
╔══════════════════════════════════════════════════════════════╗
║               SCAM DETECTION & SENTIMENT REPORT              ║
╠══════════════════════════════════════════════════════════════╣
║  Verdict              [HIGH RISK]  SCAM                      ║
║  Confidence           0.8646                                 ║
║  Sentiment            GREEDY                                 ║
║  Description          Financial lure / prize bait            ║
║  Top tokens           lucky winner, prize, winner, claim     ║
║  Keywords             [U] expires  [U] now  [G] claim  [G] prize ║
║  Polarity             +0.417 | Subjectivity: 0.833           ║
╚══════════════════════════════════════════════════════════════╝
```

### Batch mode — analyse a file (one message per line)

```bash
python -X utf8 predict.py --file messages.txt
```

**Format of `messages.txt`:**

```
Your account has been suspended. Verify now!
Hi, can we reschedule our meeting to 3pm?
You have won a $500 Amazon gift card. Claim now!
```

### Interactive REPL mode

```bash
python -X utf8 predict.py --interactive
```

Type any message at the `>` prompt and press Enter. Type `quit` to exit.

---

## Using as a Python Library

```python
import sys
sys.path.insert(0, "path/to/ml_model")

from predict import analyze_message

result = analyze_message("Your account will be CLOSED in 24 hours! Act immediately!")

# Detection result
print(result["label"])            # "Scam"
print(result["is_scam"])          # True
print(result["confidence"])       # 0.9007
print(result["risk_level"])       # "HIGH"
print(result["tfidf_top_tokens"]) # ["account", "close", "hour", ...]

# Sentiment result
print(result["sentiment_label"])       # "urgent"
print(result["sentiment_description"]) # "High-pressure / fear-driven language"
print(result["urgent_keywords"])       # ["immediately", "account", "closed"]
print(result["greed_keywords"])        # []
print(result["textblob_polarity"])     # 0.0
print(result["textblob_subjectivity"]) # 0.0
```

### Using components individually

```python
from models.scam_detector import ScamDetector
from nlp.sentiment_analyzer import SentimentAnalyzer

# Load saved detector
detector = ScamDetector.load("models/scam_detector.joblib")
print(detector.predict("You won $1M! Click here!"))

# Sentiment only (no model needed)
analyzer = SentimentAnalyzer()
print(analyzer.analyze("Act NOW or your account will be suspended!"))
```

---

## Training on Custom Data

Prepare a CSV file with the following columns:

| Column | Type | Required | Values |
|--------|------|----------|--------|
| `text` | str | Yes | The raw message text |
| `label` | int | Yes | `1` = Scam, `0` = Legitimate |
| `sentiment_label` | str | No | `urgent` / `greedy` / `friendly` / `neutral` |

**Example CSV:**

```csv
text,label,sentiment_label
"Your account is suspended. Verify now!",1,urgent
"You won $10000. Claim your prize!",1,greedy
"Hi, are you free for coffee tomorrow?",0,neutral
```

**Train on it:**

```bash
python -X utf8 train.py --csv data/my_dataset.csv --eval --cv 5
```

---

## Sentiment Categories

| Label | Scammer Strategy | Linguistic Signals |
|-------|------------------|--------------------|
| `urgent` | Fear / pressure tactics | Imperative verbs, short sentences, urgency words like *immediately*, *suspended*, *act now* |
| `greedy` | Financial lure / prize bait | Money entities, CARDINAL numbers, words like *win*, *prize*, *claim*, *free* |
| `friendly` | False trust-building | First/second-person pronouns, trust words like *personally*, *exclusive*, *friend* |
| `neutral` | No manipulation detected | Absence of manipulation signals (normal messages) |

---

## Feature Engineering

The model extracts **22 numerical features** from each message using spaCy:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `token_count` | Total number of tokens |
| 2 | `sentence_count` | Number of sentences |
| 3 | `avg_sentence_len` | Average tokens per sentence |
| 4 | `url_count` | Number of detected URLs |
| 5 | `money_count` | Monetary mentions (`$`, `million`, etc.) |
| 6 | `urgency_density` | Urgency keyword hits / total tokens |
| 7 | `greed_density` | Greed keyword hits / total tokens |
| 8 | `trust_density` | Trust keyword hits / total tokens |
| 9 | `noun_ratio` | Nouns / total tokens |
| 10 | `verb_ratio` | Verbs / total tokens |
| 11 | `adj_ratio` | Adjectives / total tokens |
| 12 | `adv_ratio` | Adverbs / total tokens |
| 13 | `has_person` | PERSON named entity present (0/1) |
| 14 | `has_org` | ORG named entity present (0/1) |
| 15 | `has_money_ent` | MONEY named entity present (0/1) |
| 16 | `has_cardinal` | CARDINAL entity present (0/1) |
| 17 | `has_gpe` | Location (GPE) entity present (0/1) |
| 18 | `has_date` | DATE entity present (0/1) |
| 19 | `textblob_polarity` | TextBlob sentiment polarity [-1, 1] |
| 20 | `textblob_subjectivity` | TextBlob subjectivity [0, 1] |
| 21 | `caps_word_ratio` | ALL-CAPS words / total tokens |
| 22 | `exclamation_count` | Number of `!` characters |

---

## Model Performance

Results on the built-in 58-message dataset (train/test split 75/25, stratified):

| Metric | Score |
|--------|-------|
| ROC-AUC | **1.0000** |
| F1-macro (test set) | **0.9321** |
| F1-macro (5-fold CV, mean) | **0.8820** |
| F1-macro (5-fold CV, std) | **±0.0715** |
| Sentiment accuracy (scam only) | **100%** |

> Performance will vary with larger, real-world datasets. The built-in dataset is intentionally small and clear-cut. Train on more data for production use.

---

## Deployment Guide

### Option A — Local deployment (development)

```bash
git clone https://github.com/DarshanSubbaraj/LLM---Honey-Pot.git
cd LLM---Honey-Pot/ml_model
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -X utf8 train.py --eval
python -X utf8 predict.py --interactive
```

### Option B — Integrate into a Flask / FastAPI API

```python
# api.py (example)
from fastapi import FastAPI
from pydantic import BaseModel
import sys
sys.path.insert(0, "ml_model")
from predict import analyze_message

app = FastAPI(title="Scam Detector API")

class Message(BaseModel):
    text: str

@app.post("/analyze")
def analyze(msg: Message):
    return analyze_message(msg.text)
```

```bash
pip install fastapi uvicorn
uvicorn api:app --reload
# POST to http://localhost:8000/analyze
```

### Option C — Docker deployment

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY ml_model/ ./ml_model/
COPY ml_model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm
ENV PYTHONIOENCODING=utf-8
CMD ["python", "-X", "utf8", "ml_model/train.py", "--eval"]
```

```bash
docker build -t scam-detector .
docker run scam-detector
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OSError: spaCy model 'en_core_web_sm' not found` | Run `python -m spacy download en_core_web_sm` |
| `UnicodeEncodeError: 'charmap' codec` | Run with `python -X utf8 script.py` or set `PYTHONIOENCODING=utf-8` |
| `FileNotFoundError: No saved model` | Run `python -X utf8 train.py --eval` first |
| `ModuleNotFoundError` | Make sure you're running from inside `ml_model/` or add it to `sys.path` |
| Low accuracy on custom data | Add more balanced training samples; use `--cv 5` to diagnose |
| `imbalanced-learn` warnings | Dataset is small; add `--threshold 0.45` to reduce false negatives |

---

## License

MIT — Part of the **LLM Honey-Pot** research project.

---

*Generated and maintained as part of the LLM Honey-Pot scam analysis system.*
