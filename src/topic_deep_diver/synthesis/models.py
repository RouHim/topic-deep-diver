"""
Data models for information synthesis results and configurations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class CitationFormat(Enum):
    """Supported citation formats."""

    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    IEEE = "ieee"


class NarrativeType(Enum):
    """Types of narrative structures."""

    CHRONOLOGICAL = "chronological"
    THEMATIC = "thematic"
    PROBLEM_SOLUTION = "problem_solution"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"


class ConsensusLevel(Enum):
    """Levels of consensus among sources."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    CONFLICTING = "conflicting"


@dataclass
class SynthesisConfig:
    """Configuration for information synthesis."""

    # Aggregation settings
    min_sources_for_consensus: int = 3
    consensus_threshold: float = 0.7  # 0.0 to 1.0
    credibility_weight_factor: float = 0.8
    recency_weight_factor: float = 0.2

    # Narrative settings
    max_narrative_length: int = 5000  # characters
    narrative_type: NarrativeType = NarrativeType.ANALYTICAL
    include_executive_summary: bool = True
    include_key_findings: bool = True

    # Citation settings
    citation_format: CitationFormat = CitationFormat.APA
    include_evidence_strength: bool = True
    preserve_links: bool = True

    # Gap analysis settings
    gap_detection_threshold: float = 0.3
    max_followup_questions: int = 5

    # Performance settings
    max_processing_time_ms: float = 30000  # 30 seconds
    batch_size: int = 10


@dataclass
class TopicCluster:
    """Represents a cluster of related information."""

    cluster_id: str
    topic: str
    subtopics: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)  # source IDs
    consensus_level: ConsensusLevel = ConsensusLevel.MODERATE
    confidence_score: float = 0.5  # 0.0 to 1.0
    key_claims: List[str] = field(default_factory=list)
    conflicting_claims: List[str] = field(default_factory=list)
    average_credibility: float = 0.5


@dataclass
class AggregationResult:
    """Results of multi-source information aggregation."""

    total_sources: int
    processed_sources: int
    topic_clusters: List[TopicCluster] = field(default_factory=list)
    consensus_overview: Dict[str, Any] = field(default_factory=dict)
    timeline_events: List[Dict[str, Any]] = field(default_factory=list)
    fact_validation_results: Dict[str, bool] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """Represents a citation with full attribution."""

    citation_id: str
    source_id: str
    source_url: str
    source_title: str
    author_info: Optional[Dict[str, Any]] = None
    publication_date: Optional[str] = None
    citation_text: str = ""
    evidence_strength: float = 0.5  # 0.0 to 1.0
    format_variants: Dict[CitationFormat, str] = field(default_factory=dict)
    page_references: List[str] = field(default_factory=list)
    archived_url: Optional[str] = None


@dataclass
class CitationResult:
    """Results of citation tracking and formatting."""

    total_citations: int
    citations: List[Citation] = field(default_factory=list)
    citation_map: Dict[str, Citation] = field(default_factory=dict)  # claim_id -> citation
    format_used: CitationFormat = CitationFormat.APA
    orphaned_claims: List[str] = field(default_factory=list)  # claims without citations
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeSection:
    """A section of the generated narrative."""

    title: str
    content: str
    section_type: str  # "introduction", "body", "conclusion", etc.
    key_points: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)  # citation IDs
    word_count: int = 0


@dataclass
class NarrativeResult:
    """Results of narrative generation."""

    title: str
    executive_summary: Optional[str] = None
    sections: List[NarrativeSection] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    narrative_type: NarrativeType = NarrativeType.ANALYTICAL
    total_word_count: int = 0
    balanced_perspective_score: float = 0.5  # 0.0 to 1.0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGap:
    """Represents a gap in the synthesized knowledge."""

    gap_id: str
    description: str
    gap_type: str  # "missing_perspective", "insufficient_evidence", "temporal_gap", etc.
    severity: float = 0.5  # 0.0 to 1.0
    related_topics: List[str] = field(default_factory=list)
    suggested_questions: List[str] = field(default_factory=list)
    potential_sources: List[str] = field(default_factory=list)


@dataclass
class GapAnalysisResult:
    """Results of gap identification and analysis."""

    total_gaps: int
    knowledge_gaps: List[KnowledgeGap] = field(default_factory=list)
    coverage_score: float = 0.5  # 0.0 to 1.0 (1.0 = complete coverage)
    followup_questions: List[str] = field(default_factory=list)
    recommended_searches: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


@dataclass
class SynthesisResult:
    """Complete synthesis result combining all components."""

    synthesis_id: str
    topic: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Component results
    aggregation: AggregationResult = field(default_factory=lambda: AggregationResult(0, 0))
    narrative: NarrativeResult = field(default_factory=lambda: NarrativeResult(""))
    citations: CitationResult = field(default_factory=lambda: CitationResult(0))
    gaps: GapAnalysisResult = field(default_factory=lambda: GapAnalysisResult(0))

    # Overall quality metrics
    overall_quality_score: float = 0.5  # 0.0 to 1.0
    completeness_score: float = 0.5  # 0.0 to 1.0
    coherence_score: float = 0.5  # 0.0 to 1.0

    # Processing metadata
    total_processing_time_ms: float = 0.0
    config_used: SynthesisConfig = field(default_factory=SynthesisConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Determine if synthesis was successful."""
        return (
            self.overall_quality_score >= 0.7
            and self.completeness_score >= 0.6
            and len(self.narrative.sections) > 0
        )


@dataclass
class SynthesisMetrics:
    """Performance and quality metrics for synthesis."""

    total_syntheses_performed: int = 0
    average_processing_time_ms: float = 0.0
    average_quality_score: float = 0.0
    average_completeness_score: float = 0.0
    synthesis_success_rate: float = 0.0
    common_gap_types: Dict[str, int] = field(default_factory=dict)
    narrative_type_distribution: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)