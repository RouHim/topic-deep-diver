"""
Main query processing engine that orchestrates all components.
"""

import time

from ..logging_config import get_logger
from .models import QueryAnalysis, QueryPlan, ResearchScope, SubQuestion
from .nlp_processor import NLPProcessor
from .strategy_planner import StrategyPlanner
from .taxonomy_generator import TaxonomyGenerator

logger = get_logger(__name__)


class QueryProcessingEngine:
    """Main engine for processing and analyzing research queries."""

    def __init__(self) -> None:
        self.nlp_processor = NLPProcessor()
        self.taxonomy_generator = TaxonomyGenerator()
        self.strategy_planner = StrategyPlanner()

    async def process_query(
        self, topic: str, scope: ResearchScope = ResearchScope.COMPREHENSIVE
    ) -> QueryPlan:
        """
        Process a research topic and generate a complete query plan.

        Args:
            topic: Research topic to process
            scope: Research scope (quick, comprehensive, academic)

        Returns:
            Complete query plan with analysis and strategies
        """
        start_time = time.time()
        logger.info(
            f"Starting query processing for topic: {topic} with scope: {scope.value}"
        )

        try:
            # Step 1: Analyze the topic
            analysis = await self._analyze_topic(topic)

            # Step 2: Generate taxonomy (optional for future enhancements)
            # taxonomy = self.taxonomy_generator.generate_taxonomy(analysis)  # Uncomment when needed

            # Step 3: Create query plan
            plan = self.strategy_planner.create_query_plan(topic, scope, analysis)

            # Step 4: Enhance analysis with taxonomy insights
            analysis.domains = self.taxonomy_generator.identify_relevant_domains(topic)

            processing_time = time.time() - start_time
            logger.info(f"Query processing completed in {processing_time:.2f}s")

            return plan

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    async def _analyze_topic(self, topic: str) -> QueryAnalysis:
        """
        Analyze a research topic using NLP techniques.

        Args:
            topic: Research topic

        Returns:
            Query analysis results
        """
        logger.debug(f"Analyzing topic: {topic}")

        # Extract key concepts
        key_concepts = self.nlp_processor.extract_key_concepts(topic)

        # Identify question type
        question_type = self.nlp_processor.identify_question_type(topic)

        # Extract keywords (for potential future use)
        _keywords = self.nlp_processor.extract_keywords(topic)

        # Calculate complexity
        complexity_score = self.nlp_processor.calculate_complexity_score(topic)

        # Generate sub-questions
        sub_question_texts = self.nlp_processor.split_into_sub_questions(topic)

        # Create sub-question objects
        sub_questions = []
        for _i, sq_text in enumerate(sub_question_texts):
            sq_type = self.nlp_processor.identify_question_type(sq_text)
            importance = self._calculate_subquestion_importance(
                sq_text, topic, key_concepts
            )

            sub_question = SubQuestion(
                question=sq_text,
                question_type=sq_type,
                importance_score=importance,
                keywords=self.nlp_processor.extract_keywords(sq_text, max_keywords=5),
                domain=None,  # Will be set by taxonomy generator
            )
            sub_questions.append(sub_question)

        # Estimate sources needed
        estimated_sources = self._estimate_sources_needed(
            complexity_score, len(sub_questions)
        )

        # Identify domains
        domains = self.taxonomy_generator.identify_relevant_domains(topic)

        analysis = QueryAnalysis(
            original_topic=topic,
            key_concepts=key_concepts,
            question_type=question_type,
            domains=domains,
            complexity_score=complexity_score,
            estimated_sources=estimated_sources,
            sub_questions=sub_questions,
        )

        logger.debug(
            f"Topic analysis complete: {len(key_concepts)} concepts, "
            f"{len(sub_questions)} sub-questions, complexity: {complexity_score:.2f}"
        )

        return analysis

    def _calculate_subquestion_importance(
        self, sub_question: str, topic: str, key_concepts: list
    ) -> float:
        """
        Calculate importance score for a sub-question.

        Args:
            sub_question: Sub-question text
            topic: Original topic
            key_concepts: Key concepts from topic

        Returns:
            Importance score (0.0 to 1.0)
        """
        sq_lower = sub_question.lower()

        # Base score from overlap with original topic
        overlap_score = 0.5

        # Bonus for containing key concepts
        concept_matches = sum(
            1 for concept in key_concepts if concept.lower() in sq_lower
        )
        concept_bonus = min(concept_matches * 0.1, 0.3)

        # Bonus for question words that indicate depth
        depth_indicators = ["why", "how", "explain", "analyze", "compare"]
        depth_bonus = 0.2 if any(word in sq_lower for word in depth_indicators) else 0.0

        importance = overlap_score + concept_bonus + depth_bonus
        return min(importance, 1.0)

    def _estimate_sources_needed(
        self, complexity_score: float, num_subquestions: int
    ) -> int:
        """
        Estimate the number of sources needed for a topic.

        Args:
            complexity_score: Topic complexity (0.0 to 1.0)
            num_subquestions: Number of sub-questions

        Returns:
            Estimated number of sources
        """
        # Base estimate
        base_sources = 10

        # Adjust for complexity
        complexity_multiplier = 1.0 + (complexity_score * 2.0)  # 1.0 to 3.0

        # Adjust for number of sub-questions
        subquestion_multiplier = 1.0 + (num_subquestions * 0.1)  # 1.0 to 2.0

        estimated = int(base_sources * complexity_multiplier * subquestion_multiplier)

        # Reasonable bounds
        return max(5, min(estimated, 100))

    async def refine_plan(
        self, plan: QueryPlan, feedback: dict | None = None
    ) -> QueryPlan:
        """
        Refine an existing query plan based on feedback or new information.

        Args:
            plan: Existing query plan
            feedback: Optional feedback data

        Returns:
            Refined query plan
        """
        logger.info("Refining query plan based on feedback")

        # For now, return the original plan
        # In a full implementation, this would adjust strategies based on feedback
        return plan

    def get_processing_stats(self) -> dict:
        """
        Get statistics about query processing performance.

        Returns:
            Dictionary with processing statistics
        """
        # Placeholder for performance metrics
        return {
            "total_queries_processed": 0,
            "average_processing_time": 0.0,
            "success_rate": 1.0,
        }
