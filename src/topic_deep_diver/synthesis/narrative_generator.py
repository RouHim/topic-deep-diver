"""
Narrative generator for creating coherent stories from aggregated information.

Handles story structure creation, balanced perspective presentation,
executive summaries, key findings extraction, and conclusion synthesis.
"""

import time
from collections import defaultdict
from typing import Any, Dict, List

from ..logging_config import get_logger
from .models import (
    AggregationResult,
    CitationResult,
    NarrativeResult,
    NarrativeSection,
    NarrativeType,
    SynthesisConfig,
)

logger = get_logger(__name__)


class NarrativeGenerator:
    """Generates coherent narratives from aggregated research data."""

    def __init__(self, config: SynthesisConfig):
        self.config = config
        self.logger = logger

    async def generate_narrative(
        self,
        topic: str,
        aggregation: AggregationResult,
        citations: CitationResult,
    ) -> NarrativeResult:
        """
        Generate a coherent narrative from aggregated data.

        Args:
            topic: Research topic
            aggregation: Results from aggregation
            citations: Citation tracking results

        Returns:
            NarrativeResult with generated content
        """
        start_time = time.time()

        try:
            self.logger.info(f"Generating narrative for topic: {topic}")

            # Generate title
            title = self._generate_title(topic, aggregation)

            # Generate executive summary if requested
            executive_summary = None
            if self.config.include_executive_summary:
                executive_summary = self._generate_executive_summary(topic, aggregation)

            # Generate main sections based on narrative type
            sections = self._generate_sections(topic, aggregation, citations)

            # Extract key findings
            key_findings = []
            if self.config.include_key_findings:
                key_findings = self._extract_key_findings(aggregation, sections)

            # Generate conclusions
            conclusions = self._generate_conclusions(aggregation, sections)

            # Calculate balanced perspective score
            balanced_score = self._calculate_balanced_perspective_score(aggregation)

            processing_time = (time.time() - start_time) * 1000
            total_word_count = sum(section.word_count for section in sections)

            return NarrativeResult(
                title=title,
                executive_summary=executive_summary,
                sections=sections,
                key_findings=key_findings,
                conclusions=conclusions,
                narrative_type=self.config.narrative_type,
                total_word_count=total_word_count,
                balanced_perspective_score=balanced_score,
                processing_time_ms=processing_time,
                metadata={
                    "narrative_structure": self.config.narrative_type.value,
                    "sections_generated": len(sections),
                    "citations_integrated": len(citations.citations),
                }
            )

        except Exception as e:
            self.logger.error(f"Error generating narrative: {e}")
            processing_time = (time.time() - start_time) * 1000
            return NarrativeResult(
                title=f"Analysis of {topic}",
                sections=[],
                key_findings=[],
                conclusions=[],
                total_word_count=0,
                processing_time_ms=processing_time,
                metadata={"error": str(e)}
            )

    def _generate_title(self, topic: str, aggregation: AggregationResult) -> str:
        """Generate an appropriate title for the narrative."""
        if aggregation.topic_clusters:
            main_topic = aggregation.topic_clusters[0].topic
            return f"Comprehensive Analysis: {main_topic}"
        else:
            return f"Research Synthesis: {topic}"

    def _generate_executive_summary(
        self, topic: str, aggregation: AggregationResult
    ) -> str:
        """Generate an executive summary of the research."""
        if not aggregation.topic_clusters:
            return f"This synthesis examines {topic} based on available sources."

        total_sources = aggregation.total_sources
        main_topics = [cluster.topic for cluster in aggregation.topic_clusters[:3]]

        summary_parts = [
            f"This comprehensive synthesis examines {topic} drawing from {total_sources} sources.",
        ]

        if main_topics:
            summary_parts.append(f"Key areas covered include: {', '.join(main_topics)}.")

        consensus_info = aggregation.consensus_overview.get("overall_consensus", "mixed")
        if consensus_info == "high":
            summary_parts.append("The analysis reveals strong consensus among sources.")
        elif consensus_info == "moderate":
            summary_parts.append("Sources show moderate agreement on key findings.")
        else:
            summary_parts.append("Sources present diverse perspectives requiring careful consideration.")

        return " ".join(summary_parts)

    def _generate_sections(
        self,
        topic: str,
        aggregation: AggregationResult,
        citations: CitationResult,
    ) -> List[NarrativeSection]:
        """Generate main content sections based on narrative type."""
        sections = []

        if self.config.narrative_type == NarrativeType.CHRONOLOGICAL:
            sections = self._generate_chronological_narrative(aggregation, citations)
        elif self.config.narrative_type == NarrativeType.THEMATIC:
            sections = self._generate_thematic_narrative(aggregation, citations)
        elif self.config.narrative_type == NarrativeType.PROBLEM_SOLUTION:
            sections = self._generate_problem_solution_narrative(aggregation, citations)
        elif self.config.narrative_type == NarrativeType.COMPARATIVE:
            sections = self._generate_comparative_narrative(aggregation, citations)
        else:  # ANALYTICAL (default)
            sections = self._generate_analytical_narrative(aggregation, citations)

        return sections

    def _generate_analytical_narrative(
        self,
        aggregation: AggregationResult,
        citations: CitationResult,
    ) -> List[NarrativeSection]:
        """Generate an analytical narrative structure."""
        sections = []

        # Introduction section
        intro_content = self._generate_introduction(aggregation)
        sections.append(NarrativeSection(
            title="Introduction",
            content=intro_content,
            section_type="introduction",
            key_points=["Research overview", "Methodology summary", "Scope definition"],
            citations=[],
            word_count=len(intro_content.split())
        ))

        # Background section
        background_clusters = [c for c in aggregation.topic_clusters if "background" in c.topic.lower()]
        if background_clusters:
            background_content = self._generate_section_content(background_clusters[0])
            sections.append(NarrativeSection(
                title="Background and Context",
                content=background_content,
                section_type="background",
                key_points=["Historical context", "Current state", "Key developments"],
                citations=self._get_citations_for_cluster(background_clusters[0], citations),
                word_count=len(background_content.split())
            ))

        # Main findings section
        findings_content = self._generate_findings_section(aggregation)
        sections.append(NarrativeSection(
            title="Key Findings",
            content=findings_content,
            section_type="findings",
            key_points=["Major discoveries", "Consensus areas", "Emerging trends"],
            citations=[],
            word_count=len(findings_content.split())
        ))

        # Analysis section
        analysis_content = self._generate_analysis_section(aggregation)
        sections.append(NarrativeSection(
            title="Analysis and Implications",
            content=analysis_content,
            section_type="analysis",
            key_points=["Pattern identification", "Implication assessment", "Future considerations"],
            citations=[],
            word_count=len(analysis_content.split())
        ))

        return sections

    def _generate_chronological_narrative(
        self,
        aggregation: AggregationResult,
        citations: CitationResult,
    ) -> List[NarrativeSection]:
        """Generate a chronological narrative structure."""
        sections = []

        # Sort timeline events
        timeline_events = sorted(
            aggregation.timeline_events,
            key=lambda x: x.get("date", ""),
            reverse=True  # Most recent first
        )

        # Group events by time periods
        periods = self._group_events_by_periods(timeline_events)

        for period, events in periods.items():
            period_content = self._generate_period_content(events, aggregation)
            sections.append(NarrativeSection(
                title=f"Period: {period}",
                content=period_content,
                section_type="chronological",
                key_points=[f"Key events in {period}"],
                citations=[],
                word_count=len(period_content.split())
            ))

        return sections

    def _generate_thematic_narrative(
        self,
        aggregation: AggregationResult,
        citations: CitationResult,
    ) -> List[NarrativeSection]:
        """Generate a thematic narrative structure."""
        sections = []

        for cluster in aggregation.topic_clusters:
            theme_content = self._generate_section_content(cluster)
            sections.append(NarrativeSection(
                title=cluster.topic,
                content=theme_content,
                section_type="thematic",
                key_points=[f"Key aspects of {cluster.topic}"],
                citations=self._get_citations_for_cluster(cluster, citations),
                word_count=len(theme_content.split())
            ))

        return sections

    def _generate_problem_solution_narrative(
        self,
        aggregation: AggregationResult,
        citations: CitationResult,
    ) -> List[NarrativeSection]:
        """Generate a problem-solution narrative structure."""
        sections = []

        # Identify problems and solutions from clusters
        problem_clusters = [c for c in aggregation.topic_clusters if self._is_problem_cluster(c)]
        solution_clusters = [c for c in aggregation.topic_clusters if self._is_solution_cluster(c)]

        # Problems section
        if problem_clusters:
            problems_content = self._generate_problems_content(problem_clusters)
            sections.append(NarrativeSection(
                title="Identified Problems and Challenges",
                content=problems_content,
                section_type="problems",
                key_points=["Core issues", "Challenges identified", "Problem scope"],
                citations=[],
                word_count=len(problems_content.split())
            ))

        # Solutions section
        if solution_clusters:
            solutions_content = self._generate_solutions_content(solution_clusters)
            sections.append(NarrativeSection(
                title="Proposed Solutions and Approaches",
                content=solutions_content,
                section_type="solutions",
                key_points=["Solution strategies", "Implementation approaches", "Expected outcomes"],
                citations=[],
                word_count=len(solutions_content.split())
            ))

        return sections

    def _generate_comparative_narrative(
        self,
        aggregation: AggregationResult,
        citations: CitationResult,
    ) -> List[NarrativeSection]:
        """Generate a comparative narrative structure."""
        sections = []

        # Compare different perspectives or approaches
        perspectives = self._identify_perspectives(aggregation.topic_clusters)

        for perspective, clusters in perspectives.items():
            comparison_content = self._generate_comparison_content(clusters)
            sections.append(NarrativeSection(
                title=f"Perspective: {perspective}",
                content=comparison_content,
                section_type="comparative",
                key_points=[f"Key aspects of {perspective} approach"],
                citations=[],
                word_count=len(comparison_content.split())
            ))

        return sections

    def _generate_introduction(self, aggregation: AggregationResult) -> str:
        """Generate introduction content."""
        total_sources = aggregation.total_sources
        total_clusters = len(aggregation.topic_clusters)

        intro = f"This analysis synthesizes information from {total_sources} sources "
        intro += f"covering {total_clusters} key topic areas. "

        if aggregation.consensus_overview:
            consensus = aggregation.consensus_overview.get("overall_consensus", "mixed")
            intro += f"The sources show {consensus} levels of consensus on major findings."

        return intro

    def _generate_section_content(self, cluster) -> str:
        """Generate content for a topic cluster."""
        content_parts = []

        # Add key claims
        for i, claim in enumerate(cluster.key_claims[:3]):
            content_parts.append(f"Key finding {i+1}: {claim}")

        # Add consensus information
        if cluster.consensus_level.value == "high":
            content_parts.append("This area shows strong agreement among sources.")
        elif cluster.consensus_level.value == "conflicting":
            content_parts.append("Sources present conflicting viewpoints on this topic.")

        return " ".join(content_parts)

    def _generate_findings_section(self, aggregation: AggregationResult) -> str:
        """Generate key findings content."""
        findings = []

        for cluster in aggregation.topic_clusters[:5]:  # Top 5 clusters
            finding = f"• {cluster.topic}: "
            if cluster.consensus_level.value == "high":
                finding += "Strong consensus among sources"
            else:
                finding += "Mixed perspectives identified"
            findings.append(finding)

        return "\n".join(findings)

    def _generate_analysis_section(self, aggregation: AggregationResult) -> str:
        """Generate analysis content."""
        analysis_parts = []

        # Overall patterns
        consensus_info = aggregation.consensus_overview
        if consensus_info:
            overall = consensus_info.get("overall_consensus", "mixed")
            analysis_parts.append(f"Overall, the analysis reveals {overall} consensus patterns across sources.")

        # Implications
        analysis_parts.append("These findings suggest several important implications for the field.")

        return " ".join(analysis_parts)

    def _extract_key_findings(
        self, aggregation: AggregationResult, sections: List[NarrativeSection]
    ) -> List[str]:
        """Extract key findings from the analysis."""
        findings = []

        # Extract from high-consensus clusters
        for cluster in aggregation.topic_clusters:
            if cluster.consensus_level.value in ["high", "moderate"]:
                findings.extend(cluster.key_claims[:2])  # Top 2 claims per cluster

        return findings[:10]  # Limit to top 10 findings

    def _generate_conclusions(
        self, aggregation: AggregationResult, sections: List[NarrativeSection]
    ) -> List[str]:
        """Generate conclusion statements."""
        conclusions = []

        # Overall conclusion
        if aggregation.topic_clusters:
            main_topic = aggregation.topic_clusters[0].topic
            conclusions.append(f"The analysis of {main_topic} reveals several key insights.")

        # Consensus-based conclusion
        consensus_info = aggregation.consensus_overview
        if consensus_info and consensus_info.get("overall_consensus") == "high":
            conclusions.append("Strong consensus among sources provides confidence in the findings.")
        else:
            conclusions.append("Diverse perspectives highlight the complexity of the topic.")

        return conclusions

    def _calculate_balanced_perspective_score(self, aggregation: AggregationResult) -> float:
        """Calculate how balanced the perspectives are."""
        if not aggregation.topic_clusters:
            return 0.5

        # Simple balance calculation based on consensus distribution
        consensus_info = aggregation.consensus_overview
        if not consensus_info:
            return 0.5

        high_pct = consensus_info.get("high_consensus_percentage", 0)
        conflicting_pct = consensus_info.get("conflicting_percentage", 0)

        # Balance score: high consensus is good, but some conflicting views show balance
        balance_score = high_pct * 0.7 + (1 - conflicting_pct) * 0.3

        return min(1.0, max(0.0, balance_score))

    def _get_citations_for_cluster(self, cluster, citations: CitationResult) -> List[str]:
        """Get citation IDs relevant to a cluster."""
        # Placeholder - would match citations to cluster content
        return []

    def _group_events_by_periods(self, timeline_events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group timeline events by time periods."""
        # Simple grouping by year (placeholder)
        periods = defaultdict(list)
        for event in timeline_events:
            date = event.get("date", "")
            if date:
                # Extract year (simple approach)
                year = date[:4] if len(date) >= 4 else "unknown"
                periods[year].append(event)
        return dict(periods)

    def _generate_period_content(self, events: List[Dict[str, Any]], aggregation: AggregationResult) -> str:
        """Generate content for a time period."""
        content_parts = [f"This period includes {len(events)} key events."]
        for event in events[:3]:  # Top 3 events
            content_parts.append(f"• {event.get('title', 'Event')}")
        return " ".join(content_parts)

    def _is_problem_cluster(self, cluster) -> bool:
        """Determine if a cluster represents problems/challenges."""
        problem_keywords = ["problem", "challenge", "issue", "difficulty", "obstacle"]
        return any(keyword in cluster.topic.lower() for keyword in problem_keywords)

    def _is_solution_cluster(self, cluster) -> bool:
        """Determine if a cluster represents solutions/approaches."""
        solution_keywords = ["solution", "approach", "method", "strategy", "resolution"]
        return any(keyword in cluster.topic.lower() for keyword in solution_keywords)

    def _generate_problems_content(self, problem_clusters) -> str:
        """Generate content for problems section."""
        content_parts = []
        for cluster in problem_clusters[:3]:
            content_parts.append(f"• {cluster.topic}: {cluster.key_claims[0] if cluster.key_claims else 'Identified issue'}")
        return "\n".join(content_parts)

    def _generate_solutions_content(self, solution_clusters) -> str:
        """Generate content for solutions section."""
        content_parts = []
        for cluster in solution_clusters[:3]:
            content_parts.append(f"• {cluster.topic}: {cluster.key_claims[0] if cluster.key_claims else 'Proposed approach'}")
        return "\n".join(content_parts)

    def _identify_perspectives(self, topic_clusters) -> Dict[str, List]:
        """Identify different perspectives in the clusters."""
        # Simple perspective identification (placeholder)
        perspectives = defaultdict(list)
        for cluster in topic_clusters:
            # Categorize by topic keywords
            if "method" in cluster.topic.lower():
                perspectives["methodological"].append(cluster)
            elif "result" in cluster.topic.lower():
                perspectives["empirical"].append(cluster)
            else:
                perspectives["general"].append(cluster)
        return dict(perspectives)

    def _generate_comparison_content(self, clusters) -> str:
        """Generate comparative content for clusters."""
        content_parts = [f"Comparing {len(clusters)} different approaches:"]
        for cluster in clusters:
            content_parts.append(f"• {cluster.topic}: {cluster.consensus_level.value} consensus")
        return " ".join(content_parts)