"""
Aggregator for multi-source information synthesis.

Handles content clustering, consensus detection, credibility weighting,
timeline organization, and fact validation.
"""

import time
from collections import defaultdict
from typing import Any, Dict, List

from ..logging_config import get_logger
from .models import (
    AggregationResult,
    ConsensusLevel,
    SynthesisConfig,
    TopicCluster,
)

logger = get_logger(__name__)


class Aggregator:
    """Handles aggregation of information from multiple sources."""

    def __init__(self, config: SynthesisConfig):
        self.config = config
        self.logger = logger

    async def aggregate_sources(
        self, topic: str, sources: List[Dict[str, Any]]
    ) -> AggregationResult:
        """
        Aggregate information from multiple sources.

        Args:
            topic: Research topic
            sources: List of source data dictionaries

        Returns:
            AggregationResult with clustered information
        """
        start_time = time.time()

        try:
            self.logger.info(f"Aggregating {len(sources)} sources for topic: {topic}")

            if not sources:
                return AggregationResult(
                    total_sources=0,
                    processed_sources=0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    metadata={"message": "No sources provided"}
                )

            # Extract claims and information from sources
            claims_data = self._extract_claims_from_sources(sources)

            # Cluster claims by topic and subtopic
            topic_clusters = self._cluster_claims_by_topic(claims_data, topic)

            # Analyze consensus and conflicts
            consensus_overview = self._analyze_consensus(topic_clusters)

            # Organize timeline events
            timeline_events = self._organize_timeline_events(sources)

            # Validate facts across sources
            fact_validation_results = self._validate_facts_across_sources(claims_data)

            processing_time = (time.time() - start_time) * 1000

            return AggregationResult(
                total_sources=len(sources),
                processed_sources=len(sources),
                topic_clusters=topic_clusters,
                consensus_overview=consensus_overview,
                timeline_events=timeline_events,
                fact_validation_results=fact_validation_results,
                processing_time_ms=processing_time,
                metadata={
                    "clustering_method": "semantic_similarity",
                    "consensus_threshold": self.config.consensus_threshold,
                }
            )

        except Exception as e:
            self.logger.error(f"Error aggregating sources: {e}")
            processing_time = (time.time() - start_time) * 1000
            return AggregationResult(
                total_sources=len(sources),
                processed_sources=0,
                processing_time_ms=processing_time,
                metadata={"error": str(e)}
            )

    def _extract_claims_from_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract claims and key information from sources."""
        claims_data = []

        for source in sources:
            source_id = source.get("source_id", source.get("url", ""))
            content = source.get("content", "")
            title = source.get("title", "")
            credibility = source.get("credibility", {}).get("overall_score", 0.5)

            # Simple claim extraction (in real implementation, use NLP)
            claims = self._extract_simple_claims(content, title)

            for claim in claims:
                claims_data.append({
                    "source_id": source_id,
                    "claim": claim,
                    "credibility": credibility,
                    "content": content,
                    "title": title,
                })

        return claims_data

    def _extract_simple_claims(self, content: str, title: str) -> List[str]:
        """Simple claim extraction from content."""
        # This is a placeholder - real implementation would use NLP
        claims = []

        # Split content into sentences (simple approach)
        sentences = content.split('.') if content else []

        # Extract sentences that might contain claims
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:  # Reasonable claim length
                claims.append(sentence)

        # Add title as a key claim if not too long
        if title and len(title) < 100:
            claims.insert(0, title)

        return claims[:10]  # Limit claims per source

    def _cluster_claims_by_topic(
        self, claims_data: List[Dict[str, Any]], main_topic: str
    ) -> List[TopicCluster]:
        """Cluster claims into topic groups."""
        # Simple clustering based on keyword matching
        # Real implementation would use semantic similarity

        clusters = defaultdict(list)
        cluster_counter = 0

        for claim_data in claims_data:
            claim = claim_data["claim"].lower()

            # Simple topic detection (placeholder)
            if any(word in claim for word in ["introduction", "overview", "background"]):
                topic_key = "background"
            elif any(word in claim for word in ["method", "approach", "technique"]):
                topic_key = "methodology"
            elif any(word in claim for word in ["result", "finding", "outcome"]):
                topic_key = "results"
            elif any(word in claim for word in ["conclusion", "summary", "implication"]):
                topic_key = "conclusions"
            else:
                topic_key = f"topic_{cluster_counter}"
                cluster_counter += 1

            clusters[topic_key].append(claim_data)

        # Convert to TopicCluster objects
        topic_clusters = []
        for topic_key, cluster_claims in clusters.items():
            if len(cluster_claims) >= self.config.min_sources_for_consensus:
                consensus_level = self._determine_consensus_level(cluster_claims)

                cluster = TopicCluster(
                    cluster_id=f"cluster_{len(topic_clusters)}",
                    topic=topic_key.replace("_", " ").title(),
                    subtopics=[],  # Would be determined by deeper analysis
                    sources=[claim["source_id"] for claim in cluster_claims],
                    consensus_level=consensus_level,
                    confidence_score=self._calculate_cluster_confidence(cluster_claims),
                    key_claims=[claim["claim"] for claim in cluster_claims[:5]],  # Top claims
                    conflicting_claims=self._identify_conflicting_claims(cluster_claims),
                    average_credibility=sum(c["credibility"] for c in cluster_claims) / len(cluster_claims)
                )
                topic_clusters.append(cluster)

        return topic_clusters

    def _determine_consensus_level(self, cluster_claims: List[Dict[str, Any]]) -> ConsensusLevel:
        """Determine consensus level for a cluster."""
        if len(cluster_claims) < self.config.min_sources_for_consensus:
            return ConsensusLevel.LOW

        # Simple consensus calculation (placeholder)
        # Real implementation would use semantic similarity
        unique_claims = len(set(claim["claim"] for claim in cluster_claims))
        consensus_ratio = 1.0 - (unique_claims / len(cluster_claims))

        if consensus_ratio >= self.config.consensus_threshold:
            return ConsensusLevel.HIGH
        elif consensus_ratio >= self.config.consensus_threshold * 0.7:
            return ConsensusLevel.MODERATE
        else:
            return ConsensusLevel.CONFLICTING

    def _calculate_cluster_confidence(self, cluster_claims: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for a cluster."""
        if not cluster_claims:
            return 0.0

        # Weight by credibility and consensus
        avg_credibility = sum(c["credibility"] for c in cluster_claims) / len(cluster_claims)
        consensus_factor = 1.0 if len(cluster_claims) >= self.config.min_sources_for_consensus else 0.5

        return min(1.0, avg_credibility * consensus_factor)

    def _identify_conflicting_claims(self, cluster_claims: List[Dict[str, Any]]) -> List[str]:
        """Identify claims that conflict with the majority."""
        # Placeholder implementation
        # Real implementation would use contradiction detection
        return []  # No conflicts detected in simple implementation

    def _analyze_consensus(self, topic_clusters: List[TopicCluster]) -> Dict[str, Any]:
        """Analyze overall consensus across all clusters."""
        if not topic_clusters:
            return {"overall_consensus": "insufficient_data"}

        consensus_counts = defaultdict(int)
        for cluster in topic_clusters:
            consensus_counts[cluster.consensus_level.value] += 1

        total_clusters = len(topic_clusters)
        high_consensus_pct = consensus_counts.get("high", 0) / total_clusters
        moderate_consensus_pct = consensus_counts.get("moderate", 0) / total_clusters
        low_consensus_pct = consensus_counts.get("low", 0) / total_clusters
        conflicting_pct = consensus_counts.get("conflicting", 0) / total_clusters

        overall_consensus = "high" if high_consensus_pct > 0.5 else "moderate" if moderate_consensus_pct > 0.5 else "low"

        return {
            "overall_consensus": overall_consensus,
            "consensus_distribution": dict(consensus_counts),
            "high_consensus_percentage": high_consensus_pct,
            "conflicting_percentage": conflicting_pct,
            "total_clusters": total_clusters,
        }

    def _organize_timeline_events(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Organize sources by timeline."""
        # Extract dates from sources and sort chronologically
        events = []

        for source in sources:
            published_date = source.get("published_date")
            if published_date:
                events.append({
                    "date": published_date,
                    "source_id": source.get("source_id", source.get("url", "")),
                    "title": source.get("title", ""),
                    "type": "publication",
                })

        # Sort by date (placeholder - would need proper date parsing)
        events.sort(key=lambda x: x["date"])

        return events

    def _validate_facts_across_sources(self, claims_data: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Validate facts by cross-referencing across sources."""
        # Placeholder implementation
        # Real implementation would use fact-checking algorithms
        validation_results = {}

        # Group claims by similar content
        claim_groups = defaultdict(list)
        for claim_data in claims_data:
            # Simple grouping by first few words
            key = claim_data["claim"][:50].lower()
            claim_groups[key].append(claim_data)

        # Validate claims that appear in multiple sources
        for claim_key, claims in claim_groups.items():
            if len(claims) >= 2:  # Claim appears in multiple sources
                # Simple validation: if sources have high credibility, consider validated
                avg_credibility = sum(c["credibility"] for c in claims) / len(claims)
                validation_results[claim_key] = avg_credibility >= 0.7

        return validation_results