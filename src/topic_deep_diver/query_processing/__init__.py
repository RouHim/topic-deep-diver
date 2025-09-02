"""
Query Processing Engine for Topic Deep Diver.

This module provides intelligent query processing capabilities including:
- Topic decomposition and analysis
- Question type identification
- Search strategy generation
- Query prioritization and planning
"""

try:
    from .engine import QueryProcessingEngine
    from .models import (
        QueryAnalysis,
        QueryPlan,
        QuestionType,
        ResearchScope,
        ScopeConfig,
        SearchEngine,
        SearchStrategy,
        SubQuestion,
        TaxonomyNode,
    )
    from .nlp_processor import NLPProcessor
    from .strategy_planner import StrategyPlanner
    from .taxonomy_generator import TaxonomyGenerator

    __all__ = [
        "QueryProcessingEngine",
        "QueryAnalysis",
        "SubQuestion",
        "SearchStrategy",
        "ResearchScope",
        "QueryPlan",
        "QuestionType",
        "SearchEngine",
        "TaxonomyNode",
        "ScopeConfig",
        "NLPProcessor",
        "TaxonomyGenerator",
        "StrategyPlanner",
    ]
except ImportError as e:
    # Handle missing dependencies gracefully
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"Query processing dependencies not available: {e}")
    logger.info("Install NLP dependencies with: uv sync --dev")

    # Provide dummy classes for type hints
    class _DummyQueryProcessingEngine:
        pass

    # Define only in except block to avoid redefinition
    __all__ = ["_DummyQueryProcessingEngine"]
