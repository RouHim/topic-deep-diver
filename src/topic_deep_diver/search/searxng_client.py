"""SearXNG client for web search integration."""

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Individual search result from SearXNG."""

    title: str
    url: str
    content: str
    score: float
    source_engine: str
    published_date: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class SearchResponse:
    """Response from SearXNG search containing multiple results."""

    query: str
    results: list[SearchResult]
    total_results: int
    search_time: float
    engines_used: list[str]


class SearXNGClient:
    """Async client for SearXNG search engine."""

    def __init__(
        self,
        base_url: str = "https://search.himmelstein.info",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        # Configure HTTP client with proper headers
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "User-Agent": "TopicDeepDiver/1.0 (Research Agent)",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

    async def search(
        self,
        query: str,
        categories: list[str] | None = None,
        engines: list[str] | None = None,
        time_range: str | None = None,
        safesearch: int = 1,
        page_no: int = 1,
        results_count: int = 20,
    ) -> SearchResponse:
        """
        Perform search using SearXNG.

        Args:
            query: Search query string
            categories: Search categories (general, images, news, etc.)
            engines: Specific engines to use
            time_range: Time filter (day, week, month, year)
            safesearch: Safe search level (0=off, 1=moderate, 2=strict)
            page_no: Page number for pagination
            results_count: Number of results per page

        Returns:
            SearchResponse with results and metadata
        """
        search_start = asyncio.get_event_loop().time()

        # Build search parameters
        params: dict[str, str | int] = {
            "q": query,
            "format": "json",
            # SearXNG API expects 'safesearch' as a string
            "safesearch": str(safesearch),
        }

        if categories:
            params["categories"] = ",".join(categories)

        if engines:
            params["engines"] = ",".join(engines)

        if time_range:
            params["time_range"] = time_range

        logger.info(f"Searching SearXNG for: '{query}' with params: {params}")

        # Perform search with retries
        for attempt in range(self.max_retries):
            try:
                response = await self.client.get(
                    f"{self.base_url}/search", params=params
                )
                response.raise_for_status()

                search_data = response.json()
                search_time = asyncio.get_event_loop().time() - search_start

                return self._parse_search_response(query, search_data, search_time)

            except httpx.HTTPError as e:
                logger.warning(f"SearXNG search attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)  # Exponential backoff

        raise RuntimeError("Max retries exceeded for SearXNG search")

    def _parse_search_response(
        self, query: str, data: dict[str, Any], search_time: float
    ) -> SearchResponse:
        """Parse raw SearXNG JSON response into SearchResponse object."""

        results = []
        engines_used = set()

        # Parse individual results
        for item in data.get("results", []):
            # Extract basic fields
            title = item.get("title", "").strip()
            url = item.get("url", "")
            content = item.get("content", "").strip()

            # Skip invalid results
            if not title or not url:
                continue

            # Extract metadata
            score = item.get("score", 0.0)
            engine = item.get("engine", "unknown")
            engines_used.add(engine)

            # Handle published date
            published_date = item.get("publishedDate")
            if published_date and not isinstance(published_date, str):
                published_date = str(published_date)

            # Create result object
            result = SearchResult(
                title=title,
                url=url,
                content=content,
                score=score,
                source_engine=engine,
                published_date=published_date,
                metadata={
                    "parsed_url": item.get("parsed_url", {}),
                    "template": item.get("template"),
                    "positions": item.get("positions", []),
                    "category": item.get("category"),
                },
            )
            results.append(result)

        # Sort results by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        total_results = data.get("number_of_results", len(results))

        logger.info(
            f"SearXNG search completed: {len(results)} results in {search_time:.2f}s "
            f"using engines: {list(engines_used)}"
        )

        return SearchResponse(
            query=query,
            results=results,
            total_results=total_results,
            search_time=search_time,
            engines_used=list(engines_used),
        )

    async def search_news(
        self, query: str, time_range: str = "week", results_count: int = 15
    ) -> SearchResponse:
        """Search for news articles specifically."""
        return await self.search(
            query=query,
            categories=["news"],
            time_range=time_range,
            results_count=results_count,
        )

    async def search_academic(
        self, query: str, results_count: int = 10
    ) -> SearchResponse:
        """Search for academic/scholarly content."""
        # Use engines that focus on academic content
        academic_engines = ["google scholar", "semantic scholar", "crossref"]
        return await self.search(
            query=query, engines=academic_engines, results_count=results_count
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self) -> "SearXNGClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
