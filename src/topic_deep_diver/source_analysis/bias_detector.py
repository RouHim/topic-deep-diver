"""
Bias detection system for identifying political, commercial, and sentiment bias.
"""

from ..logging_config import get_logger
from .models import AnalysisConfig, BiasAnalysis, BiasType

logger = get_logger(__name__)


class BiasDetector:
    """Main bias detection engine."""

    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig()
        self.logger = logger

        # Bias detection patterns from config
        bias_keywords = self.config.bias_keywords
        political_keywords = bias_keywords.get("political", {})
        self._political_keywords = (
            political_keywords if isinstance(political_keywords, dict) else {}
        )
        commercial_keywords = bias_keywords.get("commercial", [])
        self._commercial_indicators = (
            commercial_keywords if isinstance(commercial_keywords, list) else []
        )

    async def analyze_bias(
        self, title: str, content: str | None = None, url: str | None = None
    ) -> BiasAnalysis:
        """
        Analyze content for various types of bias.

        Args:
            title: Source title
            content: Source content (optional)
            url: Source URL (optional)

        Returns:
            BiasAnalysis with detailed breakdown
        """
        try:
            text_to_analyze = f"{title} {content or ''}".lower()

            # Detect political bias
            political_bias, political_score = self._detect_political_bias(
                text_to_analyze
            )

            # Detect commercial bias
            commercial_bias = self._detect_commercial_bias(text_to_analyze, url)

            # Analyze sentiment
            sentiment_score = self._analyze_sentiment(text_to_analyze)

            # Calculate overall bias score
            bias_score = self._calculate_overall_bias_score(
                political_score, commercial_bias, sentiment_score
            )

            # Determine primary bias type
            bias_type = self._determine_primary_bias_type(
                political_bias, commercial_bias, sentiment_score
            )

            # Calculate perspective diversity
            perspective_diversity = self._calculate_perspective_diversity(
                text_to_analyze
            )

            # Detect specific indicators
            detected_indicators = self._detect_bias_indicators(text_to_analyze)

            # Calculate confidence
            confidence = self._calculate_bias_confidence(content, detected_indicators)

            analysis = BiasAnalysis(
                bias_type=bias_type,
                bias_score=bias_score,
                political_bias=political_bias,
                commercial_bias=commercial_bias,
                sentiment_score=sentiment_score,
                perspective_diversity=perspective_diversity,
                detected_indicators=detected_indicators,
                confidence=confidence,
            )

            self.logger.debug(
                f"Bias analysis for '{title[:50]}...': {bias_type.value} "
                f"(score: {bias_score:.3f}, sentiment: {sentiment_score:.3f})"
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing bias for '{title}': {e}")
            return BiasAnalysis(bias_type=BiasType.NONE, bias_score=0.0, confidence=0.1)

    def _detect_political_bias(self, text: str) -> tuple[str | None, float]:
        """Detect political bias in text."""
        left_count = sum(1 for word in self._political_keywords["left"] if word in text)
        right_count = sum(
            1 for word in self._political_keywords["right"] if word in text
        )
        center_count = sum(
            1 for word in self._political_keywords["center"] if word in text
        )

        total_political_words = left_count + right_count + center_count

        if total_political_words == 0:
            return None, 0.0

        # Calculate bias strength
        if left_count > right_count and left_count > center_count:
            bias_direction = "left"
            bias_strength = min(left_count / max(total_political_words, 1), 1.0)
        elif right_count > left_count and right_count > center_count:
            bias_direction = "right"
            bias_strength = min(right_count / max(total_political_words, 1), 1.0)
        else:
            bias_direction = "center"
            bias_strength = min(center_count / max(total_political_words, 1), 0.5)

        return bias_direction, bias_strength

    def _detect_commercial_bias(self, text: str, url: str | None = None) -> bool:
        """Detect commercial bias indicators."""
        # Check for commercial keywords in text
        commercial_score = sum(
            1 for indicator in self._commercial_indicators if indicator in text
        )

        # Check URL for commercial patterns
        url_commercial = False
        if url:
            url_lower = url.lower()
            commercial_domains = ["amazon", "ebay", "shop", "store", "buy"]
            url_commercial = any(domain in url_lower for domain in commercial_domains)

        return commercial_score > 2 or url_commercial

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment in text (-1.0 to 1.0)."""
        # Simple sentiment analysis based on word lists
        positive_words = [
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "brilliant",
            "outstanding",
            "superb",
            "marvelous",
            "incredible",
        ]

        negative_words = [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "dreadful",
            "abysmal",
            "atrocious",
            "appalling",
            "dismal",
            "dire",
            "catastrophic",
        ]

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        total_sentiment_words = positive_count + negative_count

        if total_sentiment_words == 0:
            return 0.0

        # Calculate sentiment score
        sentiment_ratio = (positive_count - negative_count) / total_sentiment_words
        return max(-1.0, min(1.0, sentiment_ratio))

    def _calculate_overall_bias_score(
        self, political_score: float, commercial_bias: bool, sentiment_score: float
    ) -> float:
        """Calculate overall bias score."""
        score = political_score

        if commercial_bias:
            score += 0.3

        # Extreme sentiment can indicate bias
        if abs(sentiment_score) > 0.7:
            score += 0.2

        return min(score, 1.0)

    def _determine_primary_bias_type(
        self, political_bias: str | None, commercial_bias: bool, sentiment_score: float
    ) -> BiasType:
        """Determine the primary type of bias."""
        if commercial_bias:
            return BiasType.COMMERCIAL
        elif political_bias:
            return BiasType.POLITICAL
        elif abs(sentiment_score) > 0.6:
            return BiasType.SENTIMENT
        else:
            return BiasType.NONE

    def _calculate_perspective_diversity(self, text: str) -> float:
        """Calculate perspective diversity score."""
        # Count different viewpoint indicators
        viewpoint_indicators = [
            "however",
            "although",
            "nevertheless",
            "on the other hand",
            "conversely",
            "alternatively",
            "in contrast",
            "despite",
            "notwithstanding",
            "whereas",
            "while",
            "but",
            "yet",
        ]

        diversity_count = sum(
            1 for indicator in viewpoint_indicators if indicator in text
        )

        # Normalize to 0-1 scale
        return min(diversity_count / 5.0, 1.0)

    def _detect_bias_indicators(self, text: str) -> list[str]:
        """Detect specific bias indicators in text."""
        indicators = []

        # Political bias indicators
        for bias_type, keywords in self._political_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    indicators.append(f"political_{bias_type}_{keyword}")

        # Commercial bias indicators
        for indicator in self._commercial_indicators:
            if indicator in text:
                indicators.append(f"commercial_{indicator}")

        # Sentiment bias indicators
        if abs(self._analyze_sentiment(text)) > 0.5:
            indicators.append("sentiment_extreme")

        return indicators

    def _calculate_bias_confidence(
        self, content: str | None, indicators: list[str]
    ) -> float:
        """Calculate confidence in bias analysis."""
        confidence = 0.3  # Base confidence

        if content and len(content) > 200:
            confidence += 0.3

        if len(indicators) > 0:
            confidence += min(len(indicators) * 0.1, 0.4)

        return min(confidence, 1.0)
