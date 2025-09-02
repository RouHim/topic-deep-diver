"""
Source Analysis Engine for Topic Deep Diver.

This module provides comprehensive source quality assessment including:
- Credibility scoring based on domain authority, recency, and author expertise
- Bias detection using ML models and sentiment analysis
- Content deduplication with similarity algorithms
- Quality metrics and performance monitoring
"""

from .engine import SourceAnalysisEngine
from .credibility_scorer import CredibilityScorer
from .bias_detector import BiasDetector
from .deduplication_engine import DeduplicationEngine
from .models import SourceAnalysisResult, CredibilityScore, BiasAnalysis, DeduplicationResult

__all__ = [
    "SourceAnalysisEngine",
    "CredibilityScorer",
    "BiasDetector",
    "DeduplicationEngine",
    "SourceAnalysisResult",
    "CredibilityScore",
    "BiasAnalysis",
    "DeduplicationResult",
]