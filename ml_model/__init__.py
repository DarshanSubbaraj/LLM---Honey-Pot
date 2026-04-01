"""
__init__.py — exposes the public API of the ml_model package.
"""

from models.scam_detector import ScamDetector
from nlp.sentiment_analyzer import SentimentAnalyzer
from predict import analyze_message

__all__ = ["ScamDetector", "SentimentAnalyzer", "analyze_message"]
