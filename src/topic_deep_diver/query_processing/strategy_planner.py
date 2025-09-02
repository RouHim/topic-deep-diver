"""
Search strategy planning and optimization.
"""

import time

from ..logging_config import get_logger
from .models import (
    QueryAnalysis,
    QueryPlan,
    ResearchScope,
    ScopeConfig,
    SearchEngine,
    SearchStrategy,
    SubQuestion,
)

logger = get_logger(__name__)


class StrategyPlanner:
    """Plans and optimizes search strategies for research queries."""

    def __init__(self) -> None:
        self.scope_configs = self._initialize_scope_configs()

    def _initialize_scope_configs(self) -> dict[str, ScopeConfig]:
        """Initialize default scope configurations."""
        return {
            "quick": ScopeConfig(
                name="quick",
                max_sources=15,
                time_limit=120,  # 2 minutes
                engines=[SearchEngine.SEARXNG],
                depth_preference="breadth",
            ),
            "comprehensive": ScopeConfig(
                name="comprehensive",
                max_sources=40,
                time_limit=300,  # 5 minutes
                engines=[SearchEngine.SEARXNG, SearchEngine.GOOGLE_SCHOLAR],
                depth_preference="balanced",
            ),
            "academic": ScopeConfig(
                name="academic",
                max_sources=80,
                time_limit=600,  # 10 minutes
                engines=[
                    SearchEngine.SEARXNG,
                    SearchEngine.GOOGLE_SCHOLAR,
                    SearchEngine.SEMANTIC_SCHOLAR,
                    SearchEngine.PUBMED,
                    SearchEngine.ARXIV,
                ],
                depth_preference="depth",
            ),
        }

    def create_query_plan(
        self, topic: str, scope: ResearchScope, analysis: QueryAnalysis
    ) -> QueryPlan:
        """
        Create a comprehensive query plan for research.

        Args:
            topic: Research topic
            scope: Research scope
            analysis: Query analysis results

        Returns:
            Complete query plan
        """
        logger.info(f"Creating query plan for topic: {topic} with scope: {scope.value}")

        # Get scope configuration
        scope_config = self.scope_configs.get(
            scope.value, self.scope_configs["comprehensive"]
        )

        # Generate search strategies for each sub-question
        strategies = []
        for sub_question in analysis.sub_questions:
            strategy = self._create_search_strategy(
                sub_question, scope_config, analysis
            )
            strategies.append(strategy)

        # Calculate execution order based on dependencies and importance
        priority_order = self._calculate_priority_order(analysis.sub_questions)

        # Estimate total time and sources
        total_time = sum(strategy.time_limit for strategy in strategies)
        total_sources = sum(strategy.max_results for strategy in strategies)

        # Cap at scope limits
        total_time = min(total_time, scope_config.time_limit)
        total_sources = min(total_sources, scope_config.max_sources)

        plan = QueryPlan(
            topic=topic,
            scope=scope,
            analysis=analysis,
            strategies=strategies,
            total_estimated_time=total_time,
            total_estimated_sources=total_sources,
            priority_order=priority_order,
            created_at=time.time(),
        )

        logger.info(
            f"Query plan created with {len(strategies)} strategies, "
            f"estimated time: {total_time}s, sources: {total_sources}"
        )

        return plan

    def _create_search_strategy(
        self,
        sub_question: SubQuestion,
        scope_config: ScopeConfig,
        analysis: QueryAnalysis,
    ) -> SearchStrategy:
        """
        Create a search strategy for a specific sub-question.

        Args:
            sub_question: Sub-question to create strategy for
            scope_config: Scope configuration
            analysis: Overall query analysis

        Returns:
            Search strategy
        """
        # Select appropriate engines based on question type and domain
        engines = self._select_search_engines(sub_question, scope_config, analysis)

        # Generate optimized search query
        query = self._optimize_search_query(sub_question)

        # Determine search scope and limits
        search_scope = self._determine_search_scope(sub_question, scope_config)
        max_results = self._calculate_max_results(sub_question, scope_config)
        time_limit = self._calculate_time_limit(sub_question, scope_config)

        # Create filters based on question type and domain
        filters = self._create_search_filters(sub_question, analysis)

        strategy = SearchStrategy(
            query=query,
            engines=engines,
            scope=search_scope,
            max_results=max_results,
            time_limit=time_limit,
            filters=filters,
        )

        return strategy

    def _select_search_engines(
        self,
        sub_question: SubQuestion,
        scope_config: ScopeConfig,
        analysis: QueryAnalysis,
    ) -> list[SearchEngine]:
        """
        Select appropriate search engines for a sub-question.

        Args:
            sub_question: Sub-question
            scope_config: Scope configuration
            analysis: Query analysis

        Returns:
            List of selected search engines
        """
        available_engines = scope_config.engines.copy()

        # Add domain-specific engines
        if sub_question.domain:
            domain_engines = self._get_domain_engines(sub_question.domain)
            available_engines.extend(domain_engines)

        # Prioritize based on question type
        if sub_question.question_type.value == "factual":
            # Prefer general search engines for factual questions
            priority = [SearchEngine.SEARXNG]
        elif sub_question.question_type.value in ["analytical", "explanatory"]:
            # Prefer academic engines for analytical questions
            priority = [SearchEngine.GOOGLE_SCHOLAR, SearchEngine.SEMANTIC_SCHOLAR]
        else:
            # Use scope default for other types
            priority = available_engines[:2]

        # Ensure we have at least one engine
        if not available_engines:
            available_engines = [SearchEngine.SEARXNG]

        # Return top engines, limited to 3
        selected = []
        for engine in priority:
            if engine in available_engines and engine not in selected:
                selected.append(engine)

        # Fill with remaining engines if needed
        for engine in available_engines:
            if engine not in selected and len(selected) < 3:
                selected.append(engine)

        return selected

    def _get_domain_engines(self, domain: str) -> list[SearchEngine]:
        """
        Get domain-specific search engines.

        Args:
            domain: Domain name

        Returns:
            List of domain-specific engines
        """
        domain_mapping = {
            "science": [SearchEngine.SEMANTIC_SCHOLAR, SearchEngine.ARXIV],
            "health": [SearchEngine.PUBMED, SearchEngine.SEMANTIC_SCHOLAR],
            "technology": [SearchEngine.SEARXNG, SearchEngine.ARXIV],
            "business": [SearchEngine.SEARXNG],
            "social": [SearchEngine.SEARXNG, SearchEngine.SEMANTIC_SCHOLAR],
        }

        return domain_mapping.get(domain, [])

    def _optimize_search_query(self, sub_question: SubQuestion) -> str:
        """
        Optimize search query for better results.

        Args:
            sub_question: Sub-question

        Returns:
            Optimized search query
        """
        query = sub_question.question

        # Remove question marks and normalize
        query = query.replace("?", "").strip()

        # Add important keywords
        if sub_question.keywords:
            # Add top keywords that aren't already in the query
            query_lower = query.lower()
            additional_keywords = []
            for keyword in sub_question.keywords[:3]:  # Limit to top 3
                if keyword.lower() not in query_lower:
                    additional_keywords.append(keyword)

            if additional_keywords:
                query += " " + " ".join(additional_keywords)

        return query

    def _determine_search_scope(
        self, sub_question: SubQuestion, scope_config: ScopeConfig
    ) -> str:
        """
        Determine search scope for a sub-question.

        Args:
            sub_question: Sub-question
            scope_config: Scope configuration

        Returns:
            Search scope string
        """
        # Base scope on question type and importance
        if sub_question.question_type.value in ["factual", "predictive"]:
            base_scope = "focused"
        elif sub_question.question_type.value in ["analytical", "explanatory"]:
            base_scope = "broad"
        else:
            base_scope = "balanced"

        # Adjust based on importance
        if sub_question.importance_score > 0.8:
            scope = "comprehensive"
        elif sub_question.importance_score > 0.6:
            scope = base_scope
        else:
            scope = "quick"

        return scope

    def _calculate_max_results(
        self, sub_question: SubQuestion, scope_config: ScopeConfig
    ) -> int:
        """
        Calculate maximum results for a sub-question.

        Args:
            sub_question: Sub-question
            scope_config: Scope configuration

        Returns:
            Maximum number of results
        """
        # Base calculation on importance and scope
        base_results = scope_config.max_sources // len(scope_config.engines)

        # Adjust based on importance
        importance_multiplier = 0.5 + (sub_question.importance_score * 0.5)
        max_results = int(base_results * importance_multiplier)

        # Ensure reasonable bounds
        return max(5, min(max_results, 50))

    def _calculate_time_limit(
        self, sub_question: SubQuestion, scope_config: ScopeConfig
    ) -> int:
        """
        Calculate time limit for a sub-question.

        Args:
            sub_question: Sub-question
            scope_config: Scope configuration

        Returns:
            Time limit in seconds
        """
        # Base time per engine
        base_time = scope_config.time_limit // len(scope_config.engines)

        # Adjust based on importance
        importance_multiplier = 0.7 + (sub_question.importance_score * 0.3)
        time_limit = int(base_time * importance_multiplier)

        return max(30, min(time_limit, 300))  # 30s to 5min

    def _create_search_filters(
        self, sub_question: SubQuestion, analysis: QueryAnalysis
    ) -> dict:
        """
        Create search filters based on question and analysis.

        Args:
            sub_question: Sub-question
            analysis: Query analysis

        Returns:
            Dictionary of search filters
        """
        filters = {}

        # Add date filters for time-sensitive topics
        if any(
            word in analysis.original_topic.lower()
            for word in ["recent", "latest", "current", "new", "2024", "2025"]
        ):
            filters["date_range"] = "past_year"

        # Add language filter
        filters["language"] = "en"

        # Add domain-specific filters
        if sub_question.domain:
            filters["domain"] = sub_question.domain

        return filters

    def _calculate_priority_order(self, sub_questions: list[SubQuestion]) -> list[str]:
        """
        Calculate execution order based on dependencies and importance.

        Args:
            sub_questions: List of sub-questions

        Returns:
            List of question IDs in execution order
        """
        # Simple priority ordering based on importance score
        # In a more complex implementation, this would consider dependencies
        ordered = sorted(sub_questions, key=lambda q: q.importance_score, reverse=True)

        # Return stable identifiers based on original question order
        # Use question ID if available, otherwise use question text
        return [
            (
                str(getattr(q, "id", q.question))
                if getattr(q, "id", None) is not None
                else q.question
            )
            for q in ordered
        ]
