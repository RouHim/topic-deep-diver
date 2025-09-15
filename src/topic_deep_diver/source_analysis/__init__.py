"""
Source Analysis Engine for Topic Deep Diver.

This module provides comprehensive source quality assessment including:
- Credibility scoring based on domain authority, recency, and author expertise
- Bias detection using ML models and sentiment analysis
- Content deduplication with similarity algorithms
- Quality metrics and performance monitoring
"""

from .bias_detector import BiasDetector
from .credibility_scorer import CredibilityScorer
from .deduplication_engine import DeduplicationEngine
from .engine import SourceAnalysisEngine
from .models import (
    BiasAnalysis,
    CredibilityScore,
    DeduplicationResult,
    SourceAnalysisResult,
)

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
