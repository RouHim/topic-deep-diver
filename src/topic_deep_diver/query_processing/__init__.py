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
        SubQuestion,
        SearchStrategy,
        ResearchScope,
        QueryPlan,
        QuestionType,
        SearchEngine,
        TaxonomyNode,
        ScopeConfig
    )
    from .nlp_processor import NLPProcessor
    from .taxonomy_generator import TaxonomyGenerator
    from .strategy_planner import StrategyPlanner

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
    QueryProcessingEngine = _DummyQueryProcessingEngine
    __all__ = ["QueryProcessingEngine"]
