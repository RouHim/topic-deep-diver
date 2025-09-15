"""
Tests for the Information Synthesis Engine.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.topic_deep_diver.synthesis.engine import SynthesisEngine
from src.topic_deep_diver.synthesis.models import (
    AggregationResult,
    CitationResult,
    GapAnalysisResult,
    NarrativeResult,
    SynthesisConfig,
    SynthesisResult,
)


class TestSynthesisEngine:
    """Test cases for SynthesisEngine."""

    @pytest.fixture
    def synthesis_config(self):
        """Create test synthesis configuration."""
        return SynthesisConfig(
            min_sources_for_consensus=2,
            consensus_threshold=0.6,
            max_narrative_length=1000,
        )

    @pytest.fixture
    def synthesis_engine(self, synthesis_config):
        """Create test synthesis engine."""
        return SynthesisEngine(synthesis_config)

    @pytest.fixture
    def mock_sources(self):
        """Create mock source data for testing."""
        return [
            {
                "source_id": "source_1",
                "url": "https://example.com/article1",
                "title": "Test Article 1",
                "content": "This is a comprehensive article about artificial intelligence and its applications.",
                "credibility": {"overall_score": 0.8},
                "type": "academic",
            },
            {
                "source_id": "source_2",
                "url": "https://example.com/article2",
                "title": "Test Article 2",
                "content": "Another perspective on AI development and future implications.",
                "credibility": {"overall_score": 0.7},
                "type": "news",
            },
            {
                "source_id": "source_3",
                "url": "https://example.com/article3",
                "title": "Test Article 3",
                "content": "Technical analysis of machine learning algorithms.",
                "credibility": {"overall_score": 0.9},
                "type": "academic",
            },
        ]

    @pytest.mark.asyncio
    async def test_synthesis_engine_initialization(self, synthesis_engine):
        """Test that SynthesisEngine initializes correctly."""
        assert synthesis_engine.config is not None
        assert synthesis_engine.aggregator is not None
        assert synthesis_engine.narrative_generator is not None
        assert synthesis_engine.citation_tracker is not None
        assert synthesis_engine.gap_analyzer is not None
        assert synthesis_engine.metrics is not None

    @pytest.mark.asyncio
    async def test_synthesize_basic_functionality(self, synthesis_engine, mock_sources):
        """Test basic synthesis functionality."""
        topic = "Artificial Intelligence"
        synthesis_id = "test_synthesis_001"

        result = await synthesis_engine.synthesize(synthesis_id, topic, mock_sources)

        assert isinstance(result, SynthesisResult)
        assert result.synthesis_id == synthesis_id
        assert result.topic == topic
        assert result.overall_quality_score >= 0.0
        assert result.overall_quality_score <= 1.0
        assert result.aggregation.total_sources == len(mock_sources)
        assert len(result.narrative.sections) > 0
        assert result.citations.total_citations >= 0
        assert result.gaps.total_gaps >= 0

    @pytest.mark.asyncio
    async def test_synthesize_with_empty_sources(self, synthesis_engine):
        """Test synthesis with empty source list."""
        topic = "Empty Topic"
        synthesis_id = "test_empty"

        result = await synthesis_engine.synthesize(synthesis_id, topic, [])

        assert isinstance(result, SynthesisResult)
        assert result.aggregation.total_sources == 0
        assert result.aggregation.processed_sources == 0
        assert result.overall_quality_score >= 0.0

    @pytest.mark.asyncio
    async def test_synthesize_cache_functionality(self, synthesis_engine, mock_sources):
        """Test synthesis caching functionality."""
        topic = "Cache Test Topic"
        synthesis_id = "test_cache"

        # First synthesis
        result1 = await synthesis_engine.synthesize(synthesis_id, topic, mock_sources)
        assert not result1.metadata.get("cache_used", False)

        # Modify config to have shorter cache TTL for testing
        original_ttl = synthesis_engine.config.max_processing_time_ms
        synthesis_engine.config.max_processing_time_ms = 60000  # 60 seconds

        # Second synthesis with same ID (should use cache)
        result2 = await synthesis_engine.synthesize(synthesis_id, topic, mock_sources)
        assert result2.metadata.get("cache_used") == True
        assert result1.synthesis_id == result2.synthesis_id

        # Restore original TTL
        synthesis_engine.config.max_processing_time_ms = original_ttl

    @pytest.mark.asyncio
    async def test_quality_score_calculation(self, synthesis_engine):
        """Test quality score calculation logic."""
        # Mock aggregation result
        aggregation = AggregationResult(
            total_sources=5,
            processed_sources=5,
            topic_clusters=[],
            consensus_overview={"overall_consensus": "high"},
            timeline_events=[],
            fact_validation_results={},
            processing_time_ms=100.0,
        )

        # Mock narrative result
        narrative = NarrativeResult(
            title="Test Narrative",
            sections=[MagicMock(word_count=100)] * 3,
            key_findings=["Finding 1", "Finding 2"],
            conclusions=["Conclusion 1"],
            total_word_count=300,
            processing_time_ms=50.0,
        )

        # Mock citation result
        citations = CitationResult(
            total_citations=3,
            citations=[],
            citation_map={},
            processing_time_ms=25.0,
        )

        # Mock gap result
        gaps = GapAnalysisResult(
            total_gaps=1,
            knowledge_gaps=[],
            coverage_score=0.8,
            followup_questions=[],
            recommended_searches=[],
            processing_time_ms=30.0,
        )

        quality_score = synthesis_engine._calculate_quality_score(
            aggregation, narrative, citations, gaps
        )

        assert 0.0 <= quality_score <= 1.0
        assert quality_score > 0.5  # Should be reasonably high with good inputs

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, synthesis_engine, mock_sources):
        """Test that synthesis metrics are properly tracked."""
        initial_syntheses = synthesis_engine.metrics.total_syntheses_performed

        await synthesis_engine.synthesize("metrics_test", "Metrics Topic", mock_sources)

        assert synthesis_engine.metrics.total_syntheses_performed == initial_syntheses + 1
        assert synthesis_engine.metrics.average_quality_score >= 0.0
        assert synthesis_engine.metrics.average_processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_cache_clearing(self, synthesis_engine):
        """Test cache clearing functionality."""
        # Add something to cache
        synthesis_engine._synthesis_cache["test"] = MagicMock()
        synthesis_engine._cache_timestamps["test"] = 1234567890

        assert len(synthesis_engine._synthesis_cache) > 0

        synthesis_engine.clear_cache()

        assert len(synthesis_engine._synthesis_cache) == 0
        assert len(synthesis_engine._cache_timestamps) == 0

    @pytest.mark.asyncio
    async def test_config_updates(self, synthesis_engine):
        """Test configuration update functionality."""
        new_config = SynthesisConfig(
            min_sources_for_consensus=5,
            consensus_threshold=0.8,
            max_narrative_length=2000,
        )

        synthesis_engine.update_config(new_config)

        assert synthesis_engine.config.min_sources_for_consensus == 5
        assert synthesis_engine.config.consensus_threshold == 0.8
        assert synthesis_engine.config.max_narrative_length == 2000

    def test_get_metrics(self, synthesis_engine):
        """Test metrics retrieval."""
        metrics = synthesis_engine.get_metrics()

        assert hasattr(metrics, 'total_syntheses_performed')
        assert hasattr(metrics, 'average_processing_time_ms')
        assert hasattr(metrics, 'average_quality_score')
        assert hasattr(metrics, 'synthesis_success_rate')


class TestAggregator:
    """Test cases for Aggregator component."""

    @pytest.fixture
    def synthesis_config(self):
        """Create test synthesis configuration."""
        from src.topic_deep_diver.synthesis.models import SynthesisConfig
        return SynthesisConfig()

    @pytest.fixture
    def aggregator(self, synthesis_config):
        """Create test aggregator."""
        from src.topic_deep_diver.synthesis.aggregator import Aggregator
        return Aggregator(synthesis_config)

    @pytest.mark.asyncio
    async def test_aggregate_sources(self, aggregator, mock_sources):
        """Test source aggregation."""
        topic = "Test Topic"
        result = await aggregator.aggregate_sources(topic, mock_sources)

        assert isinstance(result, AggregationResult)
        assert result.total_sources == len(mock_sources)
        assert result.processed_sources == len(mock_sources)
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_aggregate_empty_sources(self, aggregator):
        """Test aggregation with empty source list."""
        result = await aggregator.aggregate_sources("Empty Topic", [])

        assert result.total_sources == 0
        assert result.processed_sources == 0


class TestNarrativeGenerator:
    """Test cases for NarrativeGenerator component."""

    @pytest.fixture
    def synthesis_config(self):
        """Create test synthesis configuration."""
        from src.topic_deep_diver.synthesis.models import SynthesisConfig
        return SynthesisConfig()

    @pytest.fixture
    def narrative_generator(self, synthesis_config):
        """Create test narrative generator."""
        from src.topic_deep_diver.synthesis.narrative_generator import NarrativeGenerator
        return NarrativeGenerator(synthesis_config)

    @pytest.mark.asyncio
    async def test_generate_narrative(self, narrative_generator):
        """Test narrative generation."""
        topic = "Test Topic"
        aggregation = AggregationResult(
            total_sources=3,
            processed_sources=3,
            topic_clusters=[],
            consensus_overview={},
            timeline_events=[],
            fact_validation_results={},
            processing_time_ms=100.0,
        )
        citations = CitationResult(
            total_citations=2,
            citations=[],
            citation_map={},
            processing_time_ms=50.0,
        )

        result = await narrative_generator.generate_narrative(topic, aggregation, citations)

        assert isinstance(result, NarrativeResult)
        assert result.title is not None
        assert len(result.sections) > 0
        assert result.total_word_count >= 0


class TestCitationTracker:
    """Test cases for CitationTracker component."""

    @pytest.fixture
    def synthesis_config(self):
        """Create test synthesis configuration."""
        from src.topic_deep_diver.synthesis.models import SynthesisConfig
        return SynthesisConfig()

    @pytest.fixture
    def citation_tracker(self, synthesis_config):
        """Create test citation tracker."""
        from src.topic_deep_diver.synthesis.citation_tracker import CitationTracker
        return CitationTracker(synthesis_config)

    @pytest.mark.asyncio
    async def test_track_citations(self, citation_tracker, mock_sources):
        """Test citation tracking."""
        result = await citation_tracker.track_citations(mock_sources)

        assert isinstance(result, CitationResult)
        assert result.total_citations == len(mock_sources)
        assert len(result.citations) == len(mock_sources)
        assert result.processing_time_ms > 0


class TestGapAnalyzer:
    """Test cases for GapAnalyzer component."""

    @pytest.fixture
    def synthesis_config(self):
        """Create test synthesis configuration."""
        from src.topic_deep_diver.synthesis.models import SynthesisConfig
        return SynthesisConfig()

    @pytest.fixture
    def gap_analyzer(self, synthesis_config):
        """Create test gap analyzer."""
        from src.topic_deep_diver.synthesis.gap_analyzer import GapAnalyzer
        return GapAnalyzer(synthesis_config)

    @pytest.mark.asyncio
    async def test_analyze_gaps(self, gap_analyzer):
        """Test gap analysis."""
        topic = "Test Topic"
        sources = []
        aggregation = AggregationResult(
            total_sources=0,
            processed_sources=0,
            topic_clusters=[],
            consensus_overview={},
            timeline_events=[],
            fact_validation_results={},
            processing_time_ms=100.0,
        )
        narrative = NarrativeResult(
            title="Test Narrative",
            sections=[],
            key_findings=[],
            conclusions=[],
            total_word_count=0,
            processing_time_ms=50.0,
        )

        result = await gap_analyzer.analyze_gaps(topic, sources, aggregation, narrative)

        assert isinstance(result, GapAnalysisResult)
        assert result.total_gaps >= 0
        assert 0.0 <= result.coverage_score <= 1.0
        assert result.processing_time_ms > 0


# Performance and integration tests
@pytest.mark.performance
class TestSynthesisPerformance:
    """Performance tests for synthesis engine."""

    @pytest.fixture
    def synthesis_config(self):
        """Create test synthesis configuration."""
        from src.topic_deep_diver.synthesis.models import SynthesisConfig
        return SynthesisConfig()

    @pytest.fixture
    def synthesis_engine(self, synthesis_config):
        """Create test synthesis engine."""
        from src.topic_deep_diver.synthesis.engine import SynthesisEngine
        return SynthesisEngine(synthesis_config)

    @pytest.fixture
    def large_source_set(self):
        """Create a large set of mock sources for performance testing."""
        return [
            {
                "source_id": f"source_{i}",
                "url": f"https://example.com/article{i}",
                "title": f"Test Article {i}",
                "content": f"This is test content for article {i} about the topic.",
                "credibility": {"overall_score": 0.5 + (i % 5) * 0.1},
                "type": "web",
            }
            for i in range(50)
        ]

    @pytest.mark.asyncio
    async def test_large_scale_synthesis(self, synthesis_engine, large_source_set):
        """Test synthesis with large number of sources."""
        import time

        start_time = time.time()
        result = await synthesis_engine.synthesize(
            "performance_test", "Performance Test Topic", large_source_set
        )
        end_time = time.time()

        processing_time = end_time - start_time

        assert result.aggregation.total_sources == len(large_source_set)
        assert processing_time < 30.0  # Should complete within 30 seconds
        assert result.overall_quality_score >= 0.0

    @pytest.mark.asyncio
    async def test_memory_usage(self, synthesis_engine, large_source_set):
        """Test memory usage during synthesis."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        result = await synthesis_engine.synthesize(
            "memory_test", "Memory Test Topic", large_source_set
        )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory

        assert memory_used < 500  # Should use less than 500MB additional memory
        assert result is not None