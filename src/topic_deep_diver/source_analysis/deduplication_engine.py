"""
Content deduplication engine using text similarity algorithms.
"""

import hashlib
import re
from collections import defaultdict
from datetime import datetime
from typing import Any

from ..logging_config import get_logger
from .models import AnalysisConfig, DeduplicationResult

logger = get_logger(__name__)


class DeduplicationEngine:
    """Main deduplication engine for content similarity analysis."""

    # Class-level constant for stop words to avoid recreation
    STOP_WORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
        }
    )

    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig()
        self.logger = logger

        # In-memory storage for similarity tracking
        self._content_hashes: dict[str, str] = {}
        self._similarity_clusters: dict[str, list[str]] = defaultdict(list)

    async def analyze_duplicates(
        self,
        source_id: str,
        title: str,
        content: str | None = None,
        published_date: str | None = None,
        existing_sources: list[dict[str, Any]] | None = None,
    ) -> DeduplicationResult:
        """
        Analyze content for duplicates and near-duplicates.

        Args:
            source_id: Unique identifier for the source
            title: Source title
            content: Source content (optional)
            published_date: Publication date
            existing_sources: List of existing sources to compare against

        Returns:
            DeduplicationResult with similarity analysis
        """
        try:
            # Prepare content for comparison
            content_to_analyze = self._prepare_content(title, content)

            if (
                not content_to_analyze
                or len(content_to_analyze)
                < self.config.deduplication_settings["min_content_length"]
            ):
                return DeduplicationResult(
                    is_duplicate=False, similarity_score=0.0, content_freshness=1.0
                )

            # Calculate content freshness
            content_freshness = self._calculate_content_freshness(published_date)

            # Find similar content
            similar_sources, max_similarity = await self._find_similar_content(
                source_id, content_to_analyze, existing_sources
            )

            # Determine if content is duplicate
            similarity_threshold = self.config.deduplication_settings[
                "similarity_threshold"
            ]
            is_duplicate = max_similarity >= similarity_threshold

            # Calculate redundancy level
            redundancy_level = self._calculate_redundancy_level(max_similarity)

            # Generate cluster ID if duplicate
            cluster_id = None
            if is_duplicate and similar_sources:
                cluster_id = self._generate_cluster_id(similar_sources[0])

            result = DeduplicationResult(
                is_duplicate=is_duplicate,
                similarity_score=max_similarity,
                cluster_id=cluster_id,
                duplicate_sources=similar_sources,
                content_freshness=content_freshness,
                redundancy_level=redundancy_level,
            )

            self.logger.debug(
                f"Deduplication analysis for {source_id}: "
                f"similarity={max_similarity:.3f}, duplicate={is_duplicate}, "
                f"freshness={content_freshness:.3f}"
            )

            return result

        except (ValueError, TypeError, KeyError, AttributeError) as e:
            self.logger.error(f"Error in deduplication analysis for {source_id}: {e}")
            return DeduplicationResult(
                is_duplicate=False, similarity_score=0.0, content_freshness=1.0
            )

    def _prepare_content(self, title: str, content: str | None) -> str:
        """Prepare content for similarity analysis."""
        if not content:
            return title.lower().strip()

        # Combine title and content, clean it up
        combined = f"{title} {content}"
        combined = combined.lower()

        # Remove common stop words and punctuation

        # Simple tokenization and filtering
        words = re.findall(r"\b\w+\b", combined)
        filtered_words = [
            word for word in words if word not in self.STOP_WORDS and len(word) > 2
        ]

        return " ".join(filtered_words)

    def _calculate_content_freshness(self, published_date: str | None) -> float:
        """Calculate content freshness score."""
        if not published_date:
            return 0.5  # Neutral freshness when date unknown

        try:
            if isinstance(published_date, str):
                # Try to parse the date
                pub_date = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
            else:
                pub_date = published_date

            now = datetime.now()
            days_diff = (now - pub_date).days

            decay_days = self.config.deduplication_settings["freshness_decay_days"]

            if days_diff <= 1:
                return 1.0  # Very fresh
            elif days_diff <= decay_days:
                # Linear decay over the specified period
                return 1.0 - (days_diff / float(decay_days))
            else:
                return 0.1  # Very old content

        except (ValueError, TypeError, AttributeError) as e:
            self.logger.exception(
                "Error calculating content freshness for published_date '%s': %s",
                published_date,
                e,
            )
            return 0.5

    async def _find_similar_content(
        self,
        source_id: str,
        content: str,
        existing_sources: list[dict[str, Any]] | None = None,
    ) -> tuple[list[str], float]:
        """Find similar content and return most similar sources."""
        if not existing_sources:
            return [], 0.0

        similar_sources = []
        max_similarity = 0.0

        for existing_source in existing_sources:
            existing_content = self._prepare_content(
                existing_source.get("title", ""), existing_source.get("content")
            )

            if not existing_content:
                continue

            similarity = self._calculate_similarity(content, existing_content)

            if similarity >= 0.3:  # Only consider somewhat similar content
                similar_sources.append(
                    existing_source.get(
                        "source_id", existing_source.get("url", "unknown")
                    )
                )
                max_similarity = max(max_similarity, similarity)

        # Return top 3 most similar sources
        return similar_sources[:3], max_similarity

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Jaccard similarity."""
        if not text1 or not text2:
            return 0.0

        # Simple word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        # Ensure union is not empty to prevent division by zero
        if not union:
            return 0.0

        jaccard_similarity = len(intersection) / len(union)

        # Also consider word overlap ratio
        overlap_ratio = len(intersection) / min(len(words1), len(words2))

        # Combine both metrics
        return (jaccard_similarity + overlap_ratio) / 2.0

    def _calculate_redundancy_level(self, similarity_score: float) -> str:
        """Calculate redundancy level based on similarity."""
        if similarity_score >= 0.9:
            return "high"
        elif similarity_score >= 0.7:
            return "medium"
        else:
            return "low"

    def _generate_cluster_id(self, similar_source_id: str) -> str:
        """Generate cluster ID for duplicate content."""
        # Use the first similar source as cluster identifier
        # Use deterministic hash for consistent cluster IDs
        hash_obj = hashlib.sha256(similar_source_id.encode())
        hash_value = int(hash_obj.hexdigest(), 16)
        return f"cluster_{hash_value % 10000}"

    async def get_cluster_info(self, cluster_id: str) -> dict[str, Any]:
        """Get information about a content cluster."""
        cluster_sources = self._similarity_clusters.get(cluster_id, [])

        return {
            "cluster_id": cluster_id,
            "source_count": len(cluster_sources),
            "sources": cluster_sources,
            "created_at": datetime.now().isoformat(),
        }

    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._content_hashes.clear()
        self._similarity_clusters.clear()
        self.logger.info("Deduplication cache cleared")
