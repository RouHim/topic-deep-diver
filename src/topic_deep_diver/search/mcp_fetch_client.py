"""MCP fetch client for content extraction from URLs."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedContent:
    """Content extracted from a URL."""

    url: str
    title: str
    content: str
    text_content: str
    metadata: dict[str, Any]
    extraction_time: float
    success: bool
    error_message: str | None = None


class MCPFetchClient:
    """Client for extracting content from URLs using MCP fetch server."""

    def __init__(self, timeout: int = 30, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries

    async def extract_content(self, url: str) -> ExtractedContent:
        """
        Extract content from a URL.

        This is a simplified implementation that would normally use
        an MCP fetch server. For now, we'll implement basic HTTP fetching.

        Args:
            url: URL to extract content from

        Returns:
            ExtractedContent with extracted text and metadata
        """
        extraction_start = asyncio.get_event_loop().time()

        try:
            if httpx is None:
                raise ImportError(
                    "httpx is not installed. Please install it to use content extraction."
                )

            logger.info(f"Extracting content from: {url}")

            # Configure HTTP client
            async with httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "User-Agent": "TopicDeepDiver/1.0 (Research Agent)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            ) as client:

                response = await client.get(url)
                response.raise_for_status()

                # Basic content extraction
                html_content = response.text
                title = self._extract_title(html_content)
                text_content = self._extract_text(html_content)

                extraction_time = asyncio.get_event_loop().time() - extraction_start

                # Prepare metadata
                metadata = {
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type", ""),
                    "content_length": len(html_content),
                    "encoding": response.encoding,
                    "extraction_method": "basic_http",
                    "domain": urlparse(url).netloc,
                }

                logger.info(
                    f"Content extracted from {url}: {len(text_content)} chars "
                    f"in {extraction_time:.2f}s"
                )

                return ExtractedContent(
                    url=url,
                    title=title,
                    content=html_content,
                    text_content=text_content,
                    metadata=metadata,
                    extraction_time=extraction_time,
                    success=True,
                )

        except ImportError:
            logger.error("httpx not available for content extraction")
            return self._create_error_result(
                url, "httpx dependency not available", extraction_start
            )

        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {e}")
            return self._create_error_result(url, str(e), extraction_start)

    async def extract_batch(self, urls: list[str]) -> list[ExtractedContent]:
        """
        Extract content from multiple URLs concurrently.

        Args:
            urls: List of URLs to extract content from

        Returns:
            List of ExtractedContent results
        """
        logger.info(f"Extracting content from {len(urls)} URLs")

        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)

        async def extract_with_semaphore(url: str) -> ExtractedContent:
            async with semaphore:
                return await self.extract_content(url)

        results = await asyncio.gather(
            *[extract_with_semaphore(url) for url in urls], return_exceptions=True
        )

        # Handle any exceptions that occurred
        extracted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception extracting {urls[i]}: {result}")
                extracted_results.append(
                    self._create_error_result(urls[i], str(result), 0)
                )
            elif isinstance(result, ExtractedContent):
                # result is ExtractedContent
                extracted_results.append(result)

        successful_extractions = sum(1 for r in extracted_results if r.success)
        logger.info(
            f"Batch extraction completed: {successful_extractions}/{len(urls)} successful"
        )

        return extracted_results

    def _extract_title(self, html: str) -> str:
        """Extract title from HTML content."""
        import re

        # Try to find title tag
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()

        # Try to find h1 tag as fallback
        h1_match = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
        if h1_match:
            return h1_match.group(1).strip()

        return "No title found"

    def _extract_text(self, html: str) -> str:
        """Extract text content from HTML."""
        import re

        # Remove script and style elements
        html = re.sub(
            r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE
        )

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", html)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # Limit text length to prevent memory issues
        if len(text) > 50000:  # 50KB limit
            text = text[:50000] + "... [content truncated]"

        return text

    def _create_error_result(
        self, url: str, error_message: str, start_time: float
    ) -> ExtractedContent:
        """Create an ExtractedContent object for failed extractions."""
        extraction_time = asyncio.get_event_loop().time() - start_time

        return ExtractedContent(
            url=url,
            title="Extraction Failed",
            content="",
            text_content="",
            metadata={"error": error_message},
            extraction_time=extraction_time,
            success=False,
            error_message=error_message,
        )
