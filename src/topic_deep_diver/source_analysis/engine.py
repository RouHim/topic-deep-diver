"""
Main source analysis engine that orchestrates credibility, bias, and deduplication analysis.
"""

import asyncio
import hashlib
import time
from typing import Any

from ..logging_config import get_logger
from .bias_detector import BiasDetector
from .credibility_scorer import CredibilityScorer
from .deduplication_engine import DeduplicationEngine
from .models import (
    AnalysisConfig,
    AnalysisMetrics,
    SourceAnalysisResult,
)

logger = get_logger(__name__)


class SourceAnalysisEngine:
    """Main engine for comprehensive source analysis."""

    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig()
        self.logger = logger

        # Initialize analysis components
        self.credibility_scorer = CredibilityScorer(self.config)
        self.bias_detector = BiasDetector(self.config)
        self.deduplication_engine = DeduplicationEngine(self.config)

        # Performance tracking
        self.metrics = AnalysisMetrics()

        # Analysis cache for performance
        self._analysis_cache: dict[str, SourceAnalysisResult] = {}
        self._cache_timestamps: dict[str, float] = {}

        self.logger.info("SourceAnalysisEngine initialized with all components")

    async def analyze_source(
        self,
        source_id: str,
        url: str,
        title: str,
        content: str | None = None,
        published_date: str | None = None,
        author_info: dict[str, Any] | None = None,
        citation_count: int | None = None,
        existing_sources: list[dict[str, Any]] | None = None,
        use_cache: bool = True,
    ) -> SourceAnalysisResult:
        """
        Perform comprehensive analysis of a source.

        Args:
            source_id: Unique identifier for the source
            url: Source URL
            title: Source title
            content: Source content (optional)
            published_date: Publication date
            author_info: Author information
            citation_count: Number of citations
            existing_sources: List of existing sources for deduplication
            use_cache: Whether to use cached results

        Returns:
            Complete SourceAnalysisResult
        """
        start_time = time.time()

        # Check cache first
        if use_cache:
            cached_result = self._get_cached_result(source_id)
            if cached_result:
                self.logger.debug(f"Using cached analysis for {source_id}")
                return cached_result

        try:
            self.logger.info(f"Starting comprehensive analysis for source: {source_id}")

            # Run all analysis components in parallel
            credibility_task = self.credibility_scorer.score_source(
                url=url,
                title=title,
                content=content,
                published_date=published_date,
                author_info=author_info,
                citation_count=citation_count,
            )

            bias_task = self.bias_detector.analyze_bias(
                title=title, content=content, url=url
            )

            deduplication_task = self.deduplication_engine.analyze_duplicates(
                source_id=source_id,
                title=title,
                content=content,
                published_date=published_date,
                existing_sources=existing_sources,
            )

            # Execute all tasks concurrently
            credibility_result, bias_result, deduplication_result = (
                await asyncio.gather(credibility_task, bias_task, deduplication_task)
            )

            # Create comprehensive result
            result = SourceAnalysisResult(
                source_id=source_id,
                url=url,
                title=title,
                content=content,
                credibility=credibility_result,
                bias=bias_result,
                deduplication=deduplication_result,
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "analysis_version": "1.0.0",
                    "components_used": ["credibility", "bias", "deduplication"],
                    "cache_used": False,
                },
            )

            # Cache the result
            self._cache_result(source_id, result)

            # Update metrics
            self._update_metrics(result)

            self.logger.info(
                f"Analysis completed for {source_id}: "
                f"credibility={result.credibility.overall_score:.3f}, "
                f"bias={result.bias.bias_score:.3f}, "
                f"duplicate={result.deduplication.is_duplicate}, "
                f"time={result.processing_time_ms:.1f}ms"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing source {source_id}: {e}")

            # Return minimal result on error
            processing_time = (time.time() - start_time) * 1000
            return SourceAnalysisResult(
                source_id=source_id,
                url=url,
                title=title,
                processing_time_ms=processing_time,
                metadata={"error": str(e)},
            )

    async def analyze_sources_batch(
        self,
        sources: list[dict[str, Any]],
        existing_sources: list[dict[str, Any]] | None = None,
        max_concurrent: int = 5,
    ) -> list[SourceAnalysisResult]:
        """
        Analyze multiple sources in batch with controlled concurrency.

        Args:
            sources: List of source dictionaries
            existing_sources: Existing sources for deduplication
            max_concurrent: Maximum concurrent analyses

        Returns:
            List of analysis results
        """
        self.logger.info(f"Starting batch analysis of {len(sources)} sources")

        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def analyze_with_semaphore(
            source_data: dict[str, Any],
        ) -> SourceAnalysisResult:
            async with semaphore:
                # Extract source_id with fallback to url or generated unique ID
                source_id = source_data.get("source_id") or source_data.get("url")
                if not source_id:
                    # Generate unique ID based on URL hash and timestamp
                    url = source_data.get("url", "")
                    timestamp = str(int(time.time()))
                    unique_hash = hashlib.md5(f"{url}{timestamp}".encode()).hexdigest()[
                        :8
                    ]
                    source_id = f"unknown_{unique_hash}"

                return await self.analyze_source(
                    source_id=source_id,
                    url=source_data.get("url", ""),
                    title=source_data.get("title", ""),
                    content=source_data.get("content"),
                    published_date=source_data.get("published_date"),
                    author_info=source_data.get("author_info"),
                    citation_count=source_data.get("citation_count"),
                    existing_sources=existing_sources,
                )

        # Create analysis tasks
        tasks = [analyze_with_semaphore(source) for source in sources]

        # Execute with controlled concurrency
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch analysis failed for source {i}: {result}")
                # Create error result
                source_data = sources[i]
                results.append(
                    SourceAnalysisResult(
                        source_id=source_data.get(
                            "source_id", source_data.get("url", f"error_{i}")
                        ),
                        url=source_data.get("url", ""),
                        title=source_data.get("title", "Analysis Failed"),
                        metadata={"batch_error": str(result)},
                    )
                )
            else:
                results.append(result)  # type: ignore[arg-type]

        self.logger.info(f"Batch analysis completed: {len(results)} results")
        return results

    def _get_cached_result(self, source_id: str) -> SourceAnalysisResult | None:
        """Get cached analysis result if still valid."""
        if source_id not in self._analysis_cache:
            return None

        cache_timestamp = self._cache_timestamps.get(source_id, 0)
        cache_ttl = self.config.performance_limits["cache_ttl_seconds"]

        if time.time() - cache_timestamp > cache_ttl:
            # Cache expired
            del self._analysis_cache[source_id]
            del self._cache_timestamps[source_id]
            return None

        return self._analysis_cache[source_id]

    def _cache_result(self, source_id: str, result: SourceAnalysisResult) -> None:
        """Cache analysis result."""
        self._analysis_cache[source_id] = result
        self._cache_timestamps[source_id] = time.time()

    def _update_metrics(self, result: SourceAnalysisResult) -> None:
        """Update performance metrics."""
        self.metrics.total_sources_analyzed += 1
        self.metrics.average_processing_time_ms = (
            (
                self.metrics.average_processing_time_ms
                * (self.metrics.total_sources_analyzed - 1)
            )
            + result.processing_time_ms
        ) / self.metrics.total_sources_analyzed

        # Update credibility distribution
        quality_level = result.credibility.quality_level.value
        self.metrics.credibility_score_distribution[quality_level] = (
            self.metrics.credibility_score_distribution.get(quality_level, 0) + 1
        )

        # Update cache hit rate (exponential moving average)
        cache_hit = 1.0 if result.metadata.get("cache_used") else 0.0
        alpha = 0.9  # Smoothing factor
        self.metrics.cache_hit_rate = (
            self.metrics.cache_hit_rate * alpha + cache_hit * (1 - alpha)
        )

        self.metrics.last_updated = result.analysis_timestamp

    def get_metrics(self) -> AnalysisMetrics:
        """Get current performance metrics."""
        return self.metrics

    def clear_cache(self) -> None:
        """Clear analysis cache."""
        cache_size = len(self._analysis_cache)
        self._analysis_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info(f"Cleared analysis cache ({cache_size} entries)")

    def update_config(self, new_config: AnalysisConfig) -> None:
        """Update analysis configuration."""
        self.config = new_config
        self.credibility_scorer.config = new_config
        self.bias_detector.config = new_config
        self.deduplication_engine.config = new_config
        self.logger.info("Analysis configuration updated")
