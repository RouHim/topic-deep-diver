"""
Data models for source analysis results and configurations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class BiasType(Enum):
    """Types of bias that can be detected."""

    POLITICAL = "political"
    COMMERCIAL = "commercial"
    IDEOLOGICAL = "ideological"
    SENTIMENT = "sentiment"
    NONE = "none"


class SourceQuality(Enum):
    """Quality levels for sources."""

    HIGH = "high"
    RELIABLE = "reliable"
    MODERATE = "moderate"
    LOW = "low"
    UNRELIABLE = "unreliable"


@dataclass
class CredibilityScore:
    """Credibility score for a source."""

    overall_score: float  # 0.0 to 1.0
    domain_authority: float  # 0.0 to 1.0
    recency_score: float  # 0.0 to 1.0
    author_expertise: float  # 0.0 to 1.0
    citation_count: int | None = None
    cross_reference_score: float = 0.0  # 0.0 to 1.0
    quality_level: SourceQuality = SourceQuality.MODERATE
    confidence: float = 0.5  # 0.0 to 1.0
    factors: dict[str, Any] = field(default_factory=dict)


@dataclass
class BiasAnalysis:
    """Bias analysis results for a source."""

    bias_type: BiasType
    bias_score: float  # 0.0 to 1.0 (0 = no bias, 1 = high bias)
    political_bias: str | None = None  # e.g., "left", "right", "center"
    commercial_bias: bool = False
    sentiment_score: float = 0.0  # -1.0 to 1.0
    perspective_diversity: float = 0.5  # 0.0 to 1.0
    detected_indicators: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class DeduplicationResult:
    """Results of deduplication analysis."""

    is_duplicate: bool
    similarity_score: float  # 0.0 to 1.0
    cluster_id: str | None = None
    duplicate_sources: list[str] = field(default_factory=list)
    content_freshness: float = 1.0  # 0.0 to 1.0 (1.0 = most recent)
    redundancy_level: str = "low"  # "low", "medium", "high"


@dataclass
class SourceAnalysisResult:
    """Complete analysis result for a source."""

    source_id: str
    url: str
    title: str
    content: str | None = None
    credibility: CredibilityScore = field(
        default_factory=lambda: CredibilityScore(0.5, 0.5, 0.5, 0.5)
    )
    bias: BiasAnalysis = field(default_factory=lambda: BiasAnalysis(BiasType.NONE, 0.0))
    deduplication: DeduplicationResult = field(
        default_factory=lambda: DeduplicationResult(False, 0.0)
    )
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def quality_score(self) -> float:
        """Calculate overall quality score combining credibility and bias."""
        # Weight credibility more heavily than bias
        credibility_weight = 0.7
        bias_penalty = self.bias.bias_score * 0.3
        return self.credibility.overall_score * credibility_weight - bias_penalty

    @property
    def should_include(self) -> bool:
        """Determine if source should be included in research."""
        return (
            self.credibility.overall_score >= 0.6
            and self.bias.bias_score <= 0.7
            and not self.deduplication.is_duplicate
        )


@dataclass
class AnalysisConfig:
    """Configuration for source analysis."""

    credibility_weights: dict[str, float] = field(
        default_factory=lambda: {
            "domain_authority": 0.3,
            "recency": 0.2,
            "author_expertise": 0.25,
            "citations": 0.15,
            "cross_reference": 0.1,
        }
    )

    bias_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "political_bias": 0.7,
            "commercial_bias": 0.6,
            "sentiment_extreme": 0.8,
        }
    )

    deduplication_settings: dict[str, Any] = field(
        default_factory=lambda: {
            "similarity_threshold": 0.85,
            "min_content_length": 100,
            "freshness_decay_days": 30,
        }
    )

    performance_limits: dict[str, float] = field(
        default_factory=lambda: {
            "max_analysis_time_ms": 5000,  # 5 seconds
            "cache_ttl_seconds": 3600,  # 1 hour
            "batch_size": 10,
        }
    )

    domain_authority_scores: dict[str, float] = field(
        default_factory=lambda: {
            # Academic domains
            ".edu": 0.95,
            ".ac.uk": 0.95,
            ".ac.au": 0.95,
            ".edu.au": 0.95,
            # Government domains
            ".gov": 0.90,
            ".gov.uk": 0.90,
            ".gov.au": 0.90,
            # International organizations
            ".org": 0.80,
            ".int": 0.85,
            # Scholarly domains
            "scholar.google": 0.90,
            "pubmed.ncbi.nlm.nih.gov": 0.95,
            "arxiv.org": 0.85,
            "nature.com": 0.90,
            "science.org": 0.90,
            "ieee.org": 0.85,
            "acm.org": 0.85,
            # News domains
            "reuters.com": 0.85,
            "bbc.com": 0.85,
            "ap.org": 0.85,
            "npr.org": 0.80,
            "pbs.org": 0.80,
            "bloomberg.com": 0.75,
            "wsj.com": 0.75,
            "nytimes.com": 0.80,
            "guardian.com": 0.75,
            # Default for other domains
            "default": 0.5,
        }
    )


@dataclass
class AnalysisMetrics:
    """Performance and quality metrics for analysis."""

    total_sources_analyzed: int = 0
    average_processing_time_ms: float = 0.0
    credibility_score_distribution: dict[str, int] = field(default_factory=dict)
    bias_detection_accuracy: float = 0.0
    deduplication_effectiveness: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
