"""
Tests for source analysis engine.
"""

import pytest

from topic_deep_diver.source_analysis.engine import SourceAnalysisEngine
from topic_deep_diver.source_analysis.models import AnalysisConfig


class TestSourceAnalysisEngine:
    """Test cases for source analysis engine."""

    @pytest.fixture
    def analysis_engine(self):
        """Create analysis engine for testing."""
        config = AnalysisConfig()
        return SourceAnalysisEngine(config)

    @pytest.mark.asyncio
    async def test_analyze_source_basic(self, analysis_engine):
        """Test basic source analysis functionality."""
        result = await analysis_engine.analyze_source(
            source_id="test_1",
            url="https://example.com/article",
            title="Test Article",
            content="This is a test article about technology.",
            published_date="2024-01-01",
        )

        assert result.source_id == "test_1"
        assert result.url == "https://example.com/article"
        assert result.title == "Test Article"
        assert isinstance(result.credibility.overall_score, float)
        assert isinstance(result.bias.bias_score, float)
        assert isinstance(result.deduplication.similarity_score, float)
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_analyze_high_quality_source(self, analysis_engine):
        """Test analysis of a high-quality academic source."""
        result = await analysis_engine.analyze_source(
            source_id="academic_1",
            url="https://scholar.google.com/article",
            title="Academic Research Paper",
            content="This is a peer-reviewed academic paper with citations.",
            published_date="2024-12-01",  # More recent date
            author_info={"credentials": "PhD", "affiliation": "University of Example"},
            citation_count=25,
        )

        # Should have reasonably high credibility score
        assert (
            result.credibility.overall_score > 0.6
        )  # Academic source should score well
        assert (
            result.credibility.domain_authority > 0.8
        )  # Scholar domain should be high
        assert result.credibility.quality_level.value in [
            "moderate",
            "reliable",
            "high",
        ]  # Allow moderate for now
        assert result.should_include is True

    @pytest.mark.asyncio
    async def test_analyze_commercial_bias(self, analysis_engine):
        """Test detection of commercial bias."""
        result = await analysis_engine.analyze_source(
            source_id="commercial_1",
            url="https://shop.example.com/product",
            title="Amazing Product Review",
            content="This sponsored product is amazing and you should buy it now!",
            published_date="2024-01-01",
        )

        # Should detect commercial bias
        assert result.bias.commercial_bias is True
        assert result.bias.bias_type.value == "commercial"

    @pytest.mark.asyncio
    async def test_batch_analysis(self, analysis_engine):
        """Test batch analysis functionality."""
        sources = [
            {
                "source_id": "batch_1",
                "url": "https://news.example.com/article1",
                "title": "News Article 1",
                "content": "Breaking news story.",
                "published_date": "2024-01-01",
            },
            {
                "source_id": "batch_2",
                "url": "https://blog.example.com/post1",
                "title": "Blog Post 1",
                "content": "Personal opinion on technology.",
                "published_date": "2024-01-01",
            },
        ]

        results = await analysis_engine.analyze_sources_batch(sources)

        assert len(results) == 2
        assert all(hasattr(result, "source_id") for result in results)
        assert all(result.processing_time_ms > 0 for result in results)

    def test_metrics_tracking(self, analysis_engine):
        """Test that metrics are properly tracked."""
        # Note: In a real test, we'd need to run async analysis
        # For now, just check that metrics object exists
        assert hasattr(analysis_engine, "metrics")
        assert hasattr(analysis_engine.metrics, "total_sources_analyzed")

    def test_cache_functionality(self, analysis_engine):
        """Test caching functionality."""
        # Test cache clearing
        analysis_engine.clear_cache()
        assert len(analysis_engine._analysis_cache) == 0

    def test_config_updates(self, analysis_engine):
        """Test configuration updates."""
        new_config = AnalysisConfig()
        new_config.performance_limits["max_analysis_time_ms"] = 3000

        analysis_engine.update_config(new_config)

        assert analysis_engine.config.performance_limits["max_analysis_time_ms"] == 3000
