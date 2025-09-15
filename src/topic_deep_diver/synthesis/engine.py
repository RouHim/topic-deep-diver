"""
Main synthesis engine that orchestrates information aggregation, narrative generation,
citation tracking, and gap analysis.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger

from .aggregator import Aggregator
from .citation_tracker import CitationTracker
from .gap_analyzer import GapAnalyzer
from .models import (
    AggregationResult,
    CitationResult,
    GapAnalysisResult,
    NarrativeResult,
    SynthesisConfig,
    SynthesisMetrics,
    SynthesisResult,
)
from .narrative_generator import NarrativeGenerator

logger = get_logger(__name__)


class SynthesisEngine:
    """Main engine for comprehensive information synthesis."""

    def __init__(self, config: Optional[SynthesisConfig] = None):
        self.config = config or SynthesisConfig()
        self.logger = logger

        # Initialize synthesis components
        self.aggregator = Aggregator(self.config)
        self.narrative_generator = NarrativeGenerator(self.config)
        self.citation_tracker = CitationTracker(self.config)
        self.gap_analyzer = GapAnalyzer(self.config)

        # Performance tracking
        self.metrics = SynthesisMetrics()

        # Synthesis cache for performance
        self._synthesis_cache: Dict[str, SynthesisResult] = {}
        self._cache_timestamps: Dict[str, float] = {}

        self.logger.info("SynthesisEngine initialized with all components")

    async def synthesize(
        self,
        synthesis_id: str,
        topic: str,
        sources: List[Dict[str, Any]],
        use_cache: bool = True,
    ) -> SynthesisResult:
        """
        Perform comprehensive synthesis of research findings.

        Args:
            synthesis_id: Unique identifier for this synthesis
            topic: Research topic
            sources: List of analyzed source data
            use_cache: Whether to use cached results

        Returns:
            Complete SynthesisResult
        """
        start_time = time.time()

        # Check cache first
        if use_cache:
            cached_result = self._get_cached_result(synthesis_id)
            if cached_result:
                self.logger.debug(f"Using cached synthesis for {synthesis_id}")
                return cached_result

        try:
            self.logger.info(f"Starting comprehensive synthesis for topic: {topic}")

            # Run synthesis components in parallel where possible
            aggregation_task = self.aggregator.aggregate_sources(topic, sources)
            citation_task = self.citation_tracker.track_citations(sources)

            # Execute parallel tasks
            aggregation_result, citation_result = await asyncio.gather(
                aggregation_task,
                citation_task,
                return_exceptions=True,
            )

            # Handle exceptions from components
            if isinstance(aggregation_result, Exception):
                self.logger.error(f"Aggregation failed: {aggregation_result}")
                aggregation_result = AggregationResult(
                    total_sources=len(sources),
                    processed_sources=0,
                    metadata={"error": str(aggregation_result)}
                )

            if isinstance(citation_result, Exception):
                self.logger.error(f"Citation tracking failed: {citation_result}")
                citation_result = CitationResult(
                    total_citations=0,
                    metadata={"error": str(citation_result)}
                )

            # Cast to proper types after exception handling
            from typing import cast
            aggregation_result = cast(AggregationResult, aggregation_result)
            citation_result = cast(CitationResult, citation_result)

            # Generate narrative based on aggregation results
            narrative_result = await self.narrative_generator.generate_narrative(
                topic, aggregation_result, citation_result
            )

            # Analyze gaps
            gap_result = await self.gap_analyzer.analyze_gaps(
                topic, sources, aggregation_result, narrative_result
            )

            # Calculate overall quality metrics
            quality_score = self._calculate_quality_score(
                aggregation_result, narrative_result, citation_result, gap_result
            )
            completeness_score = self._calculate_completeness_score(
                aggregation_result, gap_result
            )
            coherence_score = self._calculate_coherence_score(narrative_result)

            # Create comprehensive result
            result = SynthesisResult(
                synthesis_id=synthesis_id,
                topic=topic,
                aggregation=aggregation_result,
                narrative=narrative_result,
                citations=citation_result,
                gaps=gap_result,
                overall_quality_score=quality_score,
                completeness_score=completeness_score,
                coherence_score=coherence_score,
                total_processing_time_ms=(time.time() - start_time) * 1000,
                config_used=self.config,
                metadata={
                    "synthesis_version": "1.0.0",
                    "components_used": ["aggregation", "narrative", "citations", "gaps"],
                    "cache_used": False,
                },
            )

            # Cache the result
            self._cache_result(synthesis_id, result)

            # Update metrics
            self._update_metrics(result)

            self.logger.info(
                f"Synthesis completed for {topic}: "
                f"quality={result.overall_quality_score:.3f}, "
                f"completeness={result.completeness_score:.3f}, "
                f"time={result.total_processing_time_ms:.1f}ms"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error synthesizing {topic}: {e}")

            # Return minimal result on error
            processing_time = (time.time() - start_time) * 1000
            return SynthesisResult(
                synthesis_id=synthesis_id,
                topic=topic,
                total_processing_time_ms=processing_time,
                metadata={"error": str(e)},
            )

    def _calculate_quality_score(
        self,
        aggregation: AggregationResult,
        narrative: NarrativeResult,
        citations: CitationResult,
        gaps: GapAnalysisResult,
    ) -> float:
        """Calculate overall quality score."""
        # Weight different components
        aggregation_weight = 0.3
        narrative_weight = 0.3
        citation_weight = 0.2
        gap_weight = 0.2

        # Calculate component scores
        aggregation_score = min(1.0, aggregation.processed_sources / max(1, aggregation.total_sources))
        narrative_score = min(1.0, len(narrative.sections) / 5)  # Expect at least 5 sections
        citation_score = min(1.0, citations.total_citations / max(1, aggregation.total_sources))
        gap_score = gaps.coverage_score

        return (
            aggregation_score * aggregation_weight +
            narrative_score * narrative_weight +
            citation_score * citation_weight +
            gap_score * gap_weight
        )

    def _calculate_completeness_score(
        self, aggregation: AggregationResult, gaps: GapAnalysisResult
    ) -> float:
        """Calculate completeness score based on coverage and gaps."""
        base_completeness = min(1.0, len(aggregation.topic_clusters) / 5)  # Expect at least 5 clusters
        gap_penalty = gaps.total_gaps * 0.1  # Penalty for each gap
        return max(0.0, base_completeness - gap_penalty)

    def _calculate_coherence_score(self, narrative: NarrativeResult) -> float:
        """Calculate coherence score based on narrative structure."""
        if len(narrative.sections) == 0:
            return 0.0

        # Simple coherence based on section count and balance
        section_score = min(1.0, len(narrative.sections) / 5)
        balance_score = narrative.balanced_perspective_score

        return (section_score + balance_score) / 2

    def _get_cached_result(self, synthesis_id: str) -> Optional[SynthesisResult]:
        """Get cached synthesis result if still valid."""
        if synthesis_id not in self._synthesis_cache:
            return None

        cache_timestamp = self._cache_timestamps.get(synthesis_id, 0)
        cache_ttl = self.config.max_processing_time_ms / 1000  # Use processing time as TTL

        if time.time() - cache_timestamp > cache_ttl:
            # Cache expired
            del self._synthesis_cache[synthesis_id]
            del self._cache_timestamps[synthesis_id]
            return None

        return self._synthesis_cache[synthesis_id]

    def _cache_result(self, synthesis_id: str, result: SynthesisResult) -> None:
        """Cache synthesis result."""
        self._synthesis_cache[synthesis_id] = result
        self._cache_timestamps[synthesis_id] = time.time()

    def _update_metrics(self, result: SynthesisResult) -> None:
        """Update performance metrics."""
        self.metrics.total_syntheses_performed += 1
        self.metrics.average_processing_time_ms = (
            (
                self.metrics.average_processing_time_ms
                * (self.metrics.total_syntheses_performed - 1)
            )
            + result.total_processing_time_ms
        ) / self.metrics.total_syntheses_performed

        self.metrics.average_quality_score = (
            (
                self.metrics.average_quality_score
                * (self.metrics.total_syntheses_performed - 1)
            )
            + result.overall_quality_score
        ) / self.metrics.total_syntheses_performed

        self.metrics.average_completeness_score = (
            (
                self.metrics.average_completeness_score
                * (self.metrics.total_syntheses_performed - 1)
            )
            + result.completeness_score
        ) / self.metrics.total_syntheses_performed

        # Update success rate
        success = 1.0 if result.is_successful else 0.0
        self.metrics.synthesis_success_rate = (
            self.metrics.synthesis_success_rate * (self.metrics.total_syntheses_performed - 1)
            + success
        ) / self.metrics.total_syntheses_performed

        # Update narrative type distribution
        narrative_type = result.narrative.narrative_type.value
        self.metrics.narrative_type_distribution[narrative_type] = (
            self.metrics.narrative_type_distribution.get(narrative_type, 0) + 1
        )

        self.metrics.last_updated = result.timestamp

    def get_metrics(self) -> SynthesisMetrics:
        """Get current performance metrics."""
        return self.metrics

    def clear_cache(self) -> None:
        """Clear synthesis cache."""
        cache_size = len(self._synthesis_cache)
        self._synthesis_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info(f"Cleared synthesis cache ({cache_size} entries)")

    def update_config(self, new_config: SynthesisConfig) -> None:
        """Update synthesis configuration."""
        self.config = new_config
        self.aggregator.config = new_config
        self.narrative_generator.config = new_config
        self.citation_tracker.config = new_config
        self.gap_analyzer.config = new_config
        self.logger.info("Synthesis configuration updated")