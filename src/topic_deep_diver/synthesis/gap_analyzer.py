"""
Gap analyzer for identifying knowledge gaps and missing information.

Handles gap detection algorithms, missing perspective identification,
follow-up question generation, and automated gap-filling research triggers.
"""

import time
from typing import Any, Dict, List

from ..logging_config import get_logger
from .models import (
    AggregationResult,
    GapAnalysisResult,
    KnowledgeGap,
    NarrativeResult,
    SynthesisConfig,
)

logger = get_logger(__name__)


class GapAnalyzer:
    """Analyzes knowledge gaps in synthesized information."""

    def __init__(self, config: SynthesisConfig):
        self.config = config
        self.logger = logger

    async def analyze_gaps(
        self,
        topic: str,
        sources: List[Dict[str, Any]],
        aggregation: AggregationResult,
        narrative: NarrativeResult,
    ) -> GapAnalysisResult:
        """
        Analyze gaps in the synthesized knowledge.

        Args:
            topic: Research topic
            sources: Original source data
            aggregation: Aggregation results
            narrative: Narrative results

        Returns:
            GapAnalysisResult with identified gaps
        """
        start_time = time.time()

        try:
            self.logger.info(f"Analyzing knowledge gaps for topic: {topic}")

            knowledge_gaps = []
            followup_questions = []
            recommended_searches = []

            # Detect content gaps
            content_gaps = self._detect_content_gaps(topic, sources, aggregation)
            knowledge_gaps.extend(content_gaps)

            # Detect perspective gaps
            perspective_gaps = self._detect_perspective_gaps(aggregation)
            knowledge_gaps.extend(perspective_gaps)

            # Detect temporal gaps
            temporal_gaps = self._detect_temporal_gaps(aggregation)
            knowledge_gaps.extend(temporal_gaps)

            # Generate follow-up questions
            followup_questions = self._generate_followup_questions(knowledge_gaps)

            # Generate recommended searches
            recommended_searches = self._generate_recommended_searches(knowledge_gaps, topic)

            # Calculate coverage score
            coverage_score = self._calculate_coverage_score(
                sources, aggregation, knowledge_gaps
            )

            processing_time = (time.time() - start_time) * 1000

            return GapAnalysisResult(
                total_gaps=len(knowledge_gaps),
                knowledge_gaps=knowledge_gaps,
                coverage_score=coverage_score,
                followup_questions=followup_questions,
                recommended_searches=recommended_searches,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            self.logger.error(f"Error analyzing gaps: {e}")
            processing_time = (time.time() - start_time) * 1000
            return GapAnalysisResult(
                total_gaps=0,
                knowledge_gaps=[],
                coverage_score=0.5,
                followup_questions=[],
                recommended_searches=[],
                processing_time_ms=processing_time,
            )

    def _detect_content_gaps(
        self,
        topic: str,
        sources: List[Dict[str, Any]],
        aggregation: AggregationResult,
    ) -> List[KnowledgeGap]:
        """Detect gaps in content coverage."""
        gaps = []

        # Check for minimum cluster coverage
        if len(aggregation.topic_clusters) < 3:
            gaps.append(KnowledgeGap(
                gap_id="content_coverage",
                description="Limited topic coverage - fewer than 3 major topic areas identified",
                gap_type="insufficient_evidence",
                severity=0.7,
                related_topics=[topic],
                suggested_questions=["What are the main subtopics in this area?"],
                potential_sources=["academic databases", "review articles"],
            ))

        # Check for source diversity
        unique_domains = set()
        for source in sources:
            url = source.get("url", "")
            if url:
                # Extract domain (simple approach)
                domain = url.split("//")[-1].split("/")[0] if "//" in url else url
                unique_domains.add(domain)

        if len(unique_domains) < 3:
            gaps.append(KnowledgeGap(
                gap_id="source_diversity",
                description="Limited source diversity - information primarily from few domains",
                gap_type="missing_perspective",
                severity=0.6,
                related_topics=[topic],
                suggested_questions=["What other domains cover this topic?"],
                potential_sources=["alternative news sources", "international perspectives"],
            ))

        return gaps

    def _detect_perspective_gaps(self, aggregation: AggregationResult) -> List[KnowledgeGap]:
        """Detect gaps in perspective coverage."""
        gaps = []

        # Check for conflicting viewpoints
        conflicting_clusters = [
            cluster for cluster in aggregation.topic_clusters
            if cluster.consensus_level.value == "conflicting"
        ]

        if conflicting_clusters:
            for cluster in conflicting_clusters[:2]:  # Limit to top 2
                gaps.append(KnowledgeGap(
                    gap_id=f"perspective_{cluster.cluster_id}",
                    description=f"Conflicting perspectives on {cluster.topic} need resolution",
                    gap_type="missing_perspective",
                    severity=0.8,
                    related_topics=[cluster.topic],
                    suggested_questions=[
                        f"Why do sources disagree on {cluster.topic}?",
                        f"What evidence supports each perspective on {cluster.topic}?"
                    ],
                    potential_sources=["expert opinions", "meta-analyses"],
                ))

        # Check for consensus levels
        low_consensus_clusters = [
            cluster for cluster in aggregation.topic_clusters
            if cluster.consensus_level.value == "low"
        ]

        if len(low_consensus_clusters) > len(aggregation.topic_clusters) * 0.5:
            gaps.append(KnowledgeGap(
                gap_id="consensus_gap",
                description="Low overall consensus indicates need for more authoritative sources",
                gap_type="insufficient_evidence",
                severity=0.5,
                related_topics=[cluster.topic for cluster in low_consensus_clusters],
                suggested_questions=["What are the most reliable sources on this topic?"],
                potential_sources=["peer-reviewed journals", "government reports"],
            ))

        return gaps

    def _detect_temporal_gaps(self, aggregation: AggregationResult) -> List[KnowledgeGap]:
        """Detect gaps in temporal coverage."""
        gaps = []

        # Check timeline coverage
        if not aggregation.timeline_events:
            gaps.append(KnowledgeGap(
                gap_id="temporal_coverage",
                description="No temporal information available - missing historical context",
                gap_type="temporal_gap",
                severity=0.4,
                related_topics=["historical context"],
                suggested_questions=["How has this topic evolved over time?"],
                potential_sources=["historical archives", "longitudinal studies"],
            ))
        elif len(aggregation.timeline_events) < 5:
            gaps.append(KnowledgeGap(
                gap_id="temporal_depth",
                description="Limited temporal depth - few key events identified",
                gap_type="temporal_gap",
                severity=0.3,
                related_topics=["recent developments"],
                suggested_questions=["What are the most recent developments?"],
                potential_sources=["recent publications", "news archives"],
            ))

        return gaps

    def _generate_followup_questions(self, knowledge_gaps: List[KnowledgeGap]) -> List[str]:
        """Generate follow-up questions based on identified gaps."""
        questions = []

        for gap in knowledge_gaps:
            questions.extend(gap.suggested_questions)

        # Limit to configured maximum
        return questions[: self.config.max_followup_questions]

    def _generate_recommended_searches(
        self, knowledge_gaps: List[KnowledgeGap], topic: str
    ) -> List[str]:
        """Generate recommended search queries."""
        searches = []

        for gap in knowledge_gaps:
            for source_type in gap.potential_sources:
                search = f"{topic} {gap.description.lower()} {source_type}"
                searches.append(search)

        return searches[:5]  # Limit to 5 recommendations

    def _calculate_coverage_score(
        self,
        sources: List[Dict[str, Any]],
        aggregation: AggregationResult,
        knowledge_gaps: List[KnowledgeGap],
    ) -> float:
        """Calculate overall coverage score."""
        base_score = 0.5

        # Factor in number of sources
        source_factor = min(1.0, len(sources) / 10)  # Optimal: 10+ sources

        # Factor in topic clusters
        cluster_factor = min(1.0, len(aggregation.topic_clusters) / 5)  # Optimal: 5+ clusters

        # Factor in consensus
        consensus_info = aggregation.consensus_overview
        consensus_factor = 0.5
        if consensus_info:
            high_consensus_pct = consensus_info.get("high_consensus_percentage", 0)
            consensus_factor = high_consensus_pct * 0.8 + 0.2  # Some consensus is expected

        # Penalty for gaps
        gap_penalty = len(knowledge_gaps) * 0.1

        coverage_score = (
            base_score * 0.2 +
            source_factor * 0.3 +
            cluster_factor * 0.3 +
            consensus_factor * 0.2
        ) - gap_penalty

        return max(0.0, min(1.0, coverage_score))