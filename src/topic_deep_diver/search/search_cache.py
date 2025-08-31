"""In-memory caching for search results and extracted content."""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with data and metadata."""

    key: str
    data: Any
    created_at: float
    access_count: int
    last_accessed: float
    ttl_seconds: int | None = None

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class SearchCache:
    """In-memory LRU cache for search results and content."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,  # 1 hour
        cleanup_interval: int = 300,  # 5 minutes
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        # Use OrderedDict for LRU implementation
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Cache statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Background cleanup task
        self._cleanup_task: asyncio.Task | None = None
        self._start_cleanup_task()

    def _generate_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate cache key from arguments."""

        # Create a deterministic string from arguments
        key_data = {"args": args, "kwargs": sorted(kwargs.items()) if kwargs else []}

        # Convert to JSON and hash
        json_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""

        if key not in self._cache:
            self._misses += 1
            logger.debug(f"Cache miss for key: {key[:8]}...")
            return None

        entry = self._cache[key]

        # Check if expired
        if entry.is_expired():
            logger.debug(f"Cache entry expired for key: {key[:8]}...")
            del self._cache[key]
            self._misses += 1
            return None

        # Update access stats and move to end (most recently used)
        entry.update_access()
        self._cache.move_to_end(key)
        self._hits += 1

        logger.debug(f"Cache hit for key: {key[:8]}...")
        return entry.data

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""

        current_time = time.time()
        ttl_seconds = ttl if ttl is not None else self.default_ttl

        # Create cache entry
        entry = CacheEntry(
            key=key,
            data=value,
            created_at=current_time,
            access_count=0,
            last_accessed=current_time,
            ttl_seconds=ttl_seconds,
        )

        # Add to cache
        self._cache[key] = entry
        self._cache.move_to_end(key)

        # Check if we need to evict old entries
        await self._evict_if_needed()

        logger.debug(f"Cache set for key: {key[:8]}... (TTL: {ttl_seconds}s)")

    async def search_results_key(
        self, query: str, search_type: str = "web", **search_params: Any
    ) -> str:
        """Generate cache key for search results."""
        return self._generate_key("search", query, search_type, **search_params)

    async def content_key(self, url: str) -> str:
        """Generate cache key for extracted content."""
        return self._generate_key("content", url)

    async def get_search_results(
        self, query: str, search_type: str = "web", **search_params: Any
    ) -> Any | None:
        """Get cached search results."""
        key = await self.search_results_key(query, search_type, **search_params)
        return await self.get(key)

    async def set_search_results(
        self,
        query: str,
        results: Any,
        search_type: str = "web",
        ttl: int | None = None,
        **search_params: Any,
    ) -> None:
        """Cache search results."""
        key = await self.search_results_key(query, search_type, **search_params)
        await self.set(key, results, ttl)

    async def get_content(self, url: str) -> Any | None:
        """Get cached extracted content."""
        key = await self.content_key(url)
        return await self.get(key)

    async def set_content(self, url: str, content: Any, ttl: int | None = None) -> None:
        """Cache extracted content."""
        key = await self.content_key(url)
        # Use longer TTL for content (6 hours)
        content_ttl = ttl if ttl is not None else 21600
        await self.set(key, content, content_ttl)

    async def invalidate(self, key: str) -> bool:
        """Remove entry from cache."""
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Cache invalidated for key: {key[:8]}...")
            return True
        return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Remove all entries matching a pattern."""
        keys_to_remove = [key for key in self._cache.keys() if pattern in key]

        for key in keys_to_remove:
            del self._cache[key]

        logger.info(
            f"Cache invalidated {len(keys_to_remove)} entries matching: {pattern}"
        )
        return len(keys_to_remove)

    async def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared: {count} entries removed")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "utilization": len(self._cache) / self.max_size,
        }

    async def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""

        while len(self._cache) > self.max_size:
            # Remove least recently used (first item)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._evictions += 1
            logger.debug(f"Cache evicted LRU entry: {oldest_key[:8]}...")

    def _start_cleanup_task(self) -> None:
        """Start background cleanup task."""

        async def cleanup_expired() -> None:
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self._cleanup_expired_entries()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_expired())

    async def _cleanup_expired_entries(self) -> None:
        """Remove expired entries from cache."""

        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug(f"Cache cleanup: removed {len(expired_keys)} expired entries")

    async def close(self) -> None:
        """Close cache and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        await self.clear()
        logger.info("Search cache closed")

    async def __aenter__(self) -> "SearchCache":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
