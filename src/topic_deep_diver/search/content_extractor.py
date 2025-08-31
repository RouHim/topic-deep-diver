"""Content extractor for processing HTML to markdown conversion."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedContent:
    """Processed content with markdown and metadata."""

    original_url: str
    title: str
    markdown_content: str
    word_count: int
    reading_time_minutes: int
    language: str
    summary: str
    headings: list[str]
    links: list[str]
    images: list[str]
    metadata: dict[str, Any]


class ContentExtractor:
    """Extract and process content for research purposes."""

    def __init__(self) -> None:
        self.reading_speed_wpm = 200  # Average reading speed

    def process_extracted_content(
        self, url: str, title: str, html_content: str, text_content: str
    ) -> ProcessedContent:
        """
        Process extracted HTML content into structured markdown.

        Args:
            url: Original URL
            title: Page title
            html_content: Raw HTML content
            text_content: Extracted text content

        Returns:
            ProcessedContent with markdown and analysis
        """
        logger.info(f"Processing content from: {url}")

        # Convert to markdown
        markdown_content = self._html_to_markdown(html_content)

        # Analyze content
        word_count = len(text_content.split())
        reading_time = max(1, word_count // self.reading_speed_wpm)

        # Extract structural elements
        headings = self._extract_headings(html_content)
        links = self._extract_links(html_content, url)
        images = self._extract_images(html_content, url)

        # Generate summary
        summary = self._generate_summary(text_content)

        # Detect language (simple heuristic)
        language = self._detect_language(text_content)

        # Compile metadata
        metadata = {
            "processed_at": datetime.now().isoformat(),
            "domain": urlparse(url).netloc,
            "content_length": len(html_content),
            "text_length": len(text_content),
            "heading_count": len(headings),
            "link_count": len(links),
            "image_count": len(images),
            "estimated_quality": self._assess_content_quality(text_content, headings),
        }

        logger.info(
            f"Content processed: {word_count} words, {reading_time}min read, "
            f"{len(headings)} headings, {metadata['estimated_quality']:.2f} quality"
        )

        return ProcessedContent(
            original_url=url,
            title=title,
            markdown_content=markdown_content,
            word_count=word_count,
            reading_time_minutes=reading_time,
            language=language,
            summary=summary,
            headings=headings,
            links=links,
            images=images,
            metadata=metadata,
        )

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to markdown format."""

        # Clean HTML first
        html = self._clean_html(html)

        # Convert common HTML elements to markdown
        text = html

        # Headers
        text = re.sub(
            r"<h1[^>]*>(.*?)</h1>", r"# \1\n", text, flags=re.IGNORECASE | re.DOTALL
        )
        text = re.sub(
            r"<h2[^>]*>(.*?)</h2>", r"## \1\n", text, flags=re.IGNORECASE | re.DOTALL
        )
        text = re.sub(
            r"<h3[^>]*>(.*?)</h3>", r"### \1\n", text, flags=re.IGNORECASE | re.DOTALL
        )
        text = re.sub(
            r"<h4[^>]*>(.*?)</h4>", r"#### \1\n", text, flags=re.IGNORECASE | re.DOTALL
        )
        text = re.sub(
            r"<h5[^>]*>(.*?)</h5>", r"##### \1\n", text, flags=re.IGNORECASE | re.DOTALL
        )
        text = re.sub(
            r"<h6[^>]*>(.*?)</h6>",
            r"###### \1\n",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Bold and italic
        text = re.sub(
            r"<(?:b|strong)[^>]*>(.*?)</(?:b|strong)>",
            r"**\1**",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(
            r"<(?:i|em)[^>]*>(.*?)</(?:i|em)>",
            r"*\1*",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Links
        text = re.sub(
            r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>',
            r"[\2](\1)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Images
        text = re.sub(
            r'<img[^>]*src=["\']([^"\']*)["\'][^>]*alt=["\']([^"\']*)["\'][^>]*/?>',
            r"![\2](\1)",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r'<img[^>]*src=["\']([^"\']*)["\'][^>]*/?>',
            r"![](\1)",
            text,
            flags=re.IGNORECASE,
        )

        # Lists
        text = re.sub(
            r"<li[^>]*>(.*?)</li>", r"- \1\n", text, flags=re.IGNORECASE | re.DOTALL
        )

        # Paragraphs
        text = re.sub(
            r"<p[^>]*>(.*?)</p>", r"\1\n\n", text, flags=re.IGNORECASE | re.DOTALL
        )

        # Line breaks
        text = re.sub(r"<br[^>]*>", "\n", text, flags=re.IGNORECASE)

        # Remove remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Clean up whitespace
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)  # Multiple newlines
        text = re.sub(r" +", " ", text)  # Multiple spaces
        text = text.strip()

        return text

    def _clean_html(self, html: str) -> str:
        """Clean HTML by removing unwanted elements."""

        # Remove script and style elements
        html = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )

        # Remove comments
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

        # Remove common unwanted elements
        unwanted_tags = ["nav", "header", "footer", "aside", "advertisement", "sidebar"]
        for tag in unwanted_tags:
            html = re.sub(
                f"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE
            )

        return html

    def _extract_headings(self, html: str) -> list[str]:
        """Extract all headings from HTML."""
        headings = []

        for level in range(1, 7):  # h1 to h6
            pattern = f"<h{level}[^>]*>(.*?)</h{level}>"
            matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Clean heading text
                heading = re.sub(r"<[^>]+>", "", match).strip()
                if heading:
                    headings.append(heading)

        return headings

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Extract all links from HTML."""
        links = []

        pattern = r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>'
        matches = re.findall(pattern, html, re.IGNORECASE)

        for link in matches:
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, link)

            # Filter out unwanted links
            if self._is_valid_link(absolute_url):
                links.append(absolute_url)

        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)

        return unique_links[:50]  # Limit to 50 links

    def _extract_images(self, html: str, base_url: str) -> list[str]:
        """Extract all images from HTML."""
        images = []

        pattern = r'<img[^>]*src=["\']([^"\']*)["\'][^>]*>'
        matches = re.findall(pattern, html, re.IGNORECASE)

        for img in matches:
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, img)

            # Filter out unwanted images (icons, etc.)
            if self._is_valid_image(absolute_url):
                images.append(absolute_url)

        return images[:20]  # Limit to 20 images

    def _generate_summary(self, text: str) -> str:
        """Generate a summary of the text content."""

        # Simple extractive summary - take first paragraph or first few sentences
        sentences = text.split(". ")

        if len(sentences) > 3:
            # Take first 3 sentences
            summary = ". ".join(sentences[:3]) + "."
        else:
            # Take first 200 characters
            summary = text[:200]
            if len(text) > 200:
                summary += "..."

        return summary.strip()

    def _detect_language(self, text: str) -> str:
        """Simple language detection based on common words."""

        # Very basic language detection
        english_words = [
            "the",
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
        ]

        text_lower = text.lower()
        english_count = sum(1 for word in english_words if word in text_lower)

        # If we find English words, assume English, otherwise "unknown"
        return "en" if english_count >= 3 else "unknown"

    def _assess_content_quality(self, text: str, headings: list[str]) -> float:
        """Assess content quality on a scale of 0-1."""

        score = 0.0

        # Length factor (longer content typically better, up to a point)
        word_count = len(text.split())
        if word_count > 100:
            score += 0.3
        if word_count > 500:
            score += 0.2

        # Structure factor (presence of headings)
        if len(headings) > 0:
            score += 0.2
        if len(headings) > 3:
            score += 0.1

        # Content depth factor (sentences and paragraphs)
        sentence_count = len(text.split("."))
        if sentence_count > 10:
            score += 0.1

        # Readability factor (not too many very long sentences)
        avg_sentence_length = word_count / max(sentence_count, 1)
        if 10 <= avg_sentence_length <= 25:  # Good readable length
            score += 0.1

        return min(score, 1.0)

    def _is_valid_link(self, url: str) -> bool:
        """Check if a link is valid and useful."""

        # Filter out common unwanted links
        unwanted_patterns = [
            r"#",  # Internal anchors
            r"javascript:",  # JavaScript links
            r"mailto:",  # Email links
            r"tel:",  # Phone links
            r"\.pdf$",  # PDF files
            r"\.jpg$|\.png$|\.gif$|\.jpeg$",  # Image files
        ]

        for pattern in unwanted_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False

        return True

    def _is_valid_image(self, url: str) -> bool:
        """Check if an image URL is valid and useful."""

        # Filter out small icons and unwanted images
        unwanted_patterns = [
            r"icon",
            r"logo",
            r"avatar",
            r"thumbnail",
            r"\.svg$",  # SVG icons
            r"data:",  # Data URLs
        ]

        for pattern in unwanted_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False

        return True
