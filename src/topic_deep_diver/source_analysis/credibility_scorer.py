"""
Credibility scoring system for source quality assessment.
"""

import time
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from ..logging_config import get_logger
from .models import AnalysisConfig, CredibilityScore, SourceQuality

logger = get_logger(__name__)


class CredibilityScorer:
    """Main credibility scoring engine."""

    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig()
        self.logger = logger

    async def score_source(
        self,
        url: str,
        title: str,
        content: str | None = None,
        published_date: str | None = None,
        author_info: dict[str, Any] | None = None,
        citation_count: int | None = None,
    ) -> CredibilityScore:
        """
        Calculate comprehensive credibility score for a source.

        Args:
            url: Source URL
            title: Source title
            content: Source content (optional)
            published_date: Publication date string
            author_info: Author information dictionary
            citation_count: Number of citations

        Returns:
            CredibilityScore with detailed breakdown
        """
        start_time = time.time()

        try:
            # Calculate individual scores
            domain_score = self._calculate_domain_authority(url)
            recency_score = self._calculate_recency_score(published_date)
            author_score = self._calculate_author_expertise(author_info)
            citation_score = self._calculate_citation_score(citation_count)
            cross_ref_score = 0.5  # Placeholder for cross-reference validation

            # Calculate weighted overall score
            weights = self.config.credibility_weights
            overall_score = (
                domain_score * weights["domain_authority"]
                + recency_score * weights["recency"]
                + author_score * weights["author_expertise"]
                + citation_score * weights["citations"]
                + cross_ref_score * weights["cross_reference"]
            )

            # Determine quality level
            quality_level = self._determine_quality_level(overall_score, domain_score)

            # Calculate confidence based on available data
            confidence = self._calculate_confidence_score(
                published_date, author_info, citation_count, content
            )

            processing_time = time.time() - start_time

            score = CredibilityScore(
                overall_score=min(overall_score, 1.0),
                domain_authority=domain_score,
                recency_score=recency_score,
                author_expertise=author_score,
                citation_count=citation_count,
                cross_reference_score=cross_ref_score,
                quality_level=quality_level,
                confidence=confidence,
                factors={
                    "domain": urlparse(url).netloc,
                    "has_publication_date": published_date is not None,
                    "has_author_info": author_info is not None,
                    "has_citations": citation_count is not None,
                    "processing_time_ms": processing_time * 1000,
                },
            )

            self.logger.debug(
                f"Credibility score for {url}: {score.overall_score:.3f} "
                f"(domain: {domain_score:.3f}, recency: {recency_score:.3f})"
            )

            return score

        except Exception as e:
            self.logger.error(f"Error scoring credibility for {url}: {e}")
            return CredibilityScore(
                overall_score=0.5,
                domain_authority=0.5,
                recency_score=0.5,
                author_expertise=0.5,
                quality_level=SourceQuality.MODERATE,
                confidence=0.1,
                factors={"error": str(e)},
            )

    def _calculate_domain_authority(self, url: str) -> float:
        """Calculate domain authority score based on TLD and known domains."""
        try:
            domain = urlparse(url).netloc.lower()

            # Check for exact domain matches first
            for domain_pattern, score in self.config.domain_authority_scores.items():
                if domain_pattern.startswith("."):
                    # TLD match
                    if domain.endswith(domain_pattern):
                        return score
                elif domain_pattern in domain:
                    return score

            # Return default score
            return self.config.domain_authority_scores.get("default", 0.5)

        except Exception:
            return 0.5

    def _parse_date(self, published_date: str) -> datetime | None:
        """Parse date string using common formats."""
        date_formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%m/%d/%Y",
            "%Y-%m-%dT%H:%M:%S",  # ISO 8601 without timezone
            "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601 with Z timezone
            "%Y-%m-%dT%H:%M:%S%z",  # ISO 8601 with timezone offset
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(published_date, fmt)
            except ValueError:
                continue

        return None

    def _calculate_recency_score(self, published_date: str | None) -> float:
        """Calculate recency score based on publication date."""
        if not published_date:
            return self.config.recency_settings["unknown_date_score"]

        try:
            # Parse various date formats
            if isinstance(published_date, str):
                pub_date = self._parse_date(published_date)
                if pub_date is None:
                    # If no format matches, use fallback score
                    self.logger.debug(
                        f"Date parsing failed for published_date '{published_date}'. "
                        f"Using fallback recency score: {self.config.recency_settings['fallback_score']}"
                    )
                    return self.config.recency_settings["fallback_score"]
            else:
                pub_date = published_date

            now = datetime.now()
            days_diff = (now - pub_date).days

            # Recency scoring curve
            if days_diff <= 1:
                return 1.0  # Very recent
            elif days_diff <= 7:
                return 0.9  # This week
            elif days_diff <= 30:
                return 0.8  # This month
            elif days_diff <= 90:
                return 0.7  # This quarter
            elif days_diff <= 365:
                return 0.6  # This year
            elif days_diff <= 730:
                return 0.4  # Last 2 years
            else:
                return 0.2  # Older than 2 years

        except (ValueError, TypeError, AttributeError) as e:
            self.logger.error(
                f"Error calculating recency score for published_date '{published_date}': {e}",
                exc_info=True,
            )
            return 0.5

    def _calculate_author_expertise(self, author_info: dict[str, Any] | None) -> float:
        """Calculate author expertise score."""
        if not author_info:
            return 0.5

        score = 0.5  # Base score
        factors = 0

        # Check for academic credentials
        if author_info.get("credentials"):
            credentials = author_info["credentials"].lower()
            if any(title in credentials for title in ["phd", "professor", "dr.", "md"]):
                score += 0.3
                factors += 1
            elif any(title in credentials for title in ["masters", "ms", "ma", "msc"]):
                score += 0.2
                factors += 1

        # Check for institutional affiliation
        if author_info.get("affiliation"):
            affiliation = author_info["affiliation"].lower()
            if any(
                inst in affiliation
                for inst in [".edu", ".ac.", "university", "college"]
            ):
                score += 0.2
                factors += 1

        # Check for publication history
        if author_info.get("publication_count", 0) > 0:
            pub_count = author_info["publication_count"]
            if pub_count > 50:
                score += 0.3
            elif pub_count > 20:
                score += 0.2
            elif pub_count > 5:
                score += 0.1
            factors += 1

        # Average the factors if any were found
        if factors > 0:
            score = score / factors

        return min(score, 1.0)

    def _calculate_citation_score(self, citation_count: int | None) -> float:
        """Calculate citation-based score."""
        if citation_count is None:
            return 0.5

        # Citation scoring curve
        if citation_count >= 100:
            return 1.0
        elif citation_count >= 50:
            return 0.9
        elif citation_count >= 20:
            return 0.8
        elif citation_count >= 10:
            return 0.7
        elif citation_count >= 5:
            return 0.6
        elif citation_count >= 1:
            return 0.5
        else:
            return 0.3

    def _determine_quality_level(
        self, overall_score: float, domain_score: float
    ) -> SourceQuality:
        """Determine quality level based on scores."""
        if overall_score >= 0.85 and domain_score >= 0.8:
            return SourceQuality.HIGH
        elif overall_score >= 0.7 and domain_score >= 0.6:
            return SourceQuality.RELIABLE
        elif overall_score >= 0.5:
            return SourceQuality.MODERATE
        elif overall_score >= 0.3:
            return SourceQuality.LOW
        else:
            return SourceQuality.UNRELIABLE

    def _calculate_confidence_score(
        self,
        published_date: str | None,
        author_info: dict[str, Any] | None,
        citation_count: int | None,
        content: str | None,
    ) -> float:
        """Calculate confidence in the credibility score."""
        confidence_factors = 0
        total_factors = 4

        if published_date:
            confidence_factors += 1
        if author_info:
            confidence_factors += 1
        if citation_count is not None:
            confidence_factors += 1
        if content and len(content) > 500:
            confidence_factors += 1

        return confidence_factors / total_factors
