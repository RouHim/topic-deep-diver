"""
Information Synthesis Engine for Topic Deep Diver.

This module provides comprehensive synthesis of multi-source research findings,
including aggregation, narrative generation, citation tracking, and gap analysis.
"""

from .engine import SynthesisEngine
from .models import (
    Citation,
    CitationFormat,
    CitationResult,
    ConsensusLevel,
    GapAnalysisResult,
    KnowledgeGap,
    NarrativeResult,
    NarrativeSection,
    NarrativeType,
    SynthesisConfig,
    SynthesisMetrics,
    SynthesisResult,
    TopicCluster,
    AggregationResult,
)

__all__ = [
    "Citation",
    "CitationFormat",
    "CitationResult",
    "ConsensusLevel",
    "GapAnalysisResult",
    "KnowledgeGap",
    "NarrativeResult",
    "NarrativeSection",
    "NarrativeType",
    "SynthesisConfig",
    "SynthesisEngine",
    "SynthesisMetrics",
    "SynthesisResult",
    "TopicCluster",
    "AggregationResult",
]