"""
Core MCP server implementation for Topic Deep Diver.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Literal, cast

# UTC compatibility for Python <3.11
try:
    from datetime import UTC
except ImportError:
    from datetime import timezone

    UTC = timezone.utc

from mcp.server import FastMCP
from pydantic import BaseModel, Field

from .config import get_config
from .logging_config import get_logger


class ResearchProgress(BaseModel):
    """Model for research progress tracking."""

    session_id: str = Field(description="Unique identifier for the research session")
    status: str = Field(description="Current status of the research")
    progress: float = Field(description="Progress percentage (0.0 to 1.0)")
    stage: str = Field(description="Current research stage")
    estimated_completion: str = Field(description="Estimated completion time")


class ResearchResult(BaseModel):
    """Model for comprehensive research results."""

    session_id: str = Field(description="Unique identifier for the research session")
    topic: str = Field(description="Research topic")
    scope: str = Field(description="Research scope (quick, comprehensive, academic)")
    executive_summary: str = Field(description="Executive summary of findings")
    key_findings: list[str] = Field(description="List of key research findings")
    sources: list[dict[str, Any]] = Field(description="List of source information")
    confidence_score: float = Field(description="Confidence score (0.0 to 1.0)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ExportResult(BaseModel):
    """Model for research export results."""

    session_id: str = Field(description="Research session identifier")
    format: str = Field(description="Export format")
    resource_links: list[str] = Field(description="Links to exported resources")
    size: str = Field(description="Estimated file size")
    expires_at: str = Field(description="Resource expiration timestamp")


class DeepResearchServer:
    """Main MCP server class for deep research functionality."""

    def __init__(self) -> None:
        self.logger = get_logger("server")
        self.config = get_config()

        # Session storage for tracking research progress.
        # Structure: {session_id: {session_data}}
        # - session_id (str): Unique identifier for each research session.
        # - session_data (dict): Arbitrary data associated with the session, including progress, results, etc.
        # Expiration: Sessions are not automatically expired; cleanup must be handled elsewhere if needed.
        # Thread safety: Access to session data should be protected using the corresponding lock in _session_locks.
        self._sessions: dict[str, dict[str, Any]] = {}

        # Per-session locks for thread-safe access to session data.
        # Structure: {session_id: asyncio.Lock}
        # - session_id (str): Unique identifier for each research session.
        # - asyncio.Lock: Used to ensure that concurrent access to session data is safe.
        # Thread safety: Always acquire the lock before reading or writing to _sessions[session_id].
        self._session_locks: dict[str, asyncio.Lock] = {}

        # Create FastMCP instance with structured content
        self.mcp = FastMCP(
            name=self.config.server.name,
            instructions="Advanced deep research MCP server providing comprehensive topic analysis and synthesis.",
        )
        self._setup_tools()

        self.logger.info("DeepResearchServer initialized with structured output support and session management")

    async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for the session."""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    async def _create_session(self, topic: str, scope: str) -> str:
        """Create a new research session."""
        session_id = str(uuid.uuid4())

        # Validate scope
        valid_scopes = ["quick", "comprehensive", "academic"]
        if scope not in valid_scopes:
            raise ValueError(f"Invalid scope '{scope}'. Must be one of: {valid_scopes}")

        session_data = {
            "session_id": session_id,
            "topic": topic,
            "scope": scope,
            "status": "pending",
            "stage": "initializing",
            "progress": 0.0,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "expires_at": datetime.now(UTC) + timedelta(hours=24),
            "executive_summary": None,
            "key_findings": [],
            "sources": [],
            "confidence_score": 0.0,
            "metadata": {
                "created_by": "topic_deep_diver",
                "version": "0.1.0",
                "scope_config": self._get_scope_config(scope),
            },
            "error_message": None,
            "retry_count": 0,
        }

        self._sessions[session_id] = session_data
        self.logger.info(f"Created session {session_id} for topic: {topic} (scope: {scope})")
        return session_id

    def _get_scope_config(self, scope: str) -> dict[str, Any]:
        """Get configuration for different research scopes."""
        scope_configs = {
            "quick": {
                "max_sources": 10,
                "timeout_minutes": 2,
                "stages": ["planning", "searching", "synthesizing"],
                "depth": "surface",
            },
            "comprehensive": {
                "max_sources": 30,
                "timeout_minutes": 5,
                "stages": ["planning", "searching", "analyzing", "synthesizing"],
                "depth": "detailed",
            },
            "academic": {
                "max_sources": 50,
                "timeout_minutes": 10,
                "stages": [
                    "planning",
                    "searching",
                    "analyzing",
                    "synthesizing",
                    "validating",
                ],
                "depth": "scholarly",
            },
        }
        return scope_configs.get(scope, scope_configs["comprehensive"])

    async def _update_session(self, session_id: str, **kwargs: Any) -> None:
        """Update session data with thread safety."""
        lock = await self._get_session_lock(session_id)

        async with lock:
            if session_id not in self._sessions:
                return

            session_data = self._sessions[session_id]

            # Update fields
            for key, value in kwargs.items():
                if key in session_data:
                    session_data[key] = value

            session_data["updated_at"] = datetime.now(UTC)
            self.logger.debug(f"Updated session {session_id}: {kwargs}")

    async def _conduct_research(self, session_id: str) -> dict[str, Any]:
        """Conduct the actual research for a session."""
        session_data = self._sessions.get(session_id)
        if not session_data:
            raise ValueError(f"Session {session_id} not found")

        scope = session_data["scope"]

        try:
            # Update status to in_progress
            await self._update_session(session_id, status="in_progress", stage="planning")

            # Execute research pipeline based on scope
            if scope == "quick":
                await self._execute_quick_research(session_id)
            elif scope == "academic":
                await self._execute_academic_research(session_id)
            else:  # comprehensive
                await self._execute_comprehensive_research(session_id)

            # Mark as completed
            await self._update_session(session_id, status="completed", stage="completed", progress=1.0)
            self.logger.info(f"Research completed for session {session_id}")

            return self._sessions[session_id]

        except Exception as e:
            self.logger.error(f"Research failed for session {session_id}: {e}")
            await self._update_session(session_id, status="failed", stage="failed", error_message=str(e))
            raise

    async def _execute_quick_research(self, session_id: str) -> None:
        """Execute quick research pipeline (2 minutes, 10 sources)."""
        topic = self._sessions[session_id]["topic"]

        # Stage 1: Planning (20% progress)
        await self._update_session(session_id, stage="planning", progress=0.1)
        keywords = await self._generate_search_keywords(topic, max_keywords=5)
        await asyncio.sleep(0.5)  # Simulate planning time

        # Stage 2: Searching (60% progress)
        await self._update_session(session_id, stage="searching", progress=0.3)
        sources = await self._conduct_web_search(topic, keywords, max_sources=10)
        await asyncio.sleep(1.0)  # Simulate search time

        # Stage 3: Synthesizing (100% progress)
        await self._update_session(session_id, stage="synthesizing", progress=0.8)
        result = await self._synthesize_findings(topic, sources, depth="surface")

        # Update session with results
        await self._update_session(
            session_id,
            executive_summary=result["executive_summary"],
            key_findings=result["key_findings"],
            sources=sources,
            confidence_score=result["confidence_score"],
            progress=1.0,
        )

    async def _execute_comprehensive_research(self, session_id: str) -> None:
        """Execute comprehensive research pipeline (5 minutes, 30 sources)."""
        topic = self._sessions[session_id]["topic"]

        # Stage 1: Planning (15% progress)
        await self._update_session(session_id, stage="planning", progress=0.05)
        keywords = await self._generate_search_keywords(topic, max_keywords=10)
        await asyncio.sleep(0.8)  # Simulate planning time

        # Stage 2: Searching (50% progress)
        await self._update_session(session_id, stage="searching", progress=0.2)
        sources = await self._conduct_comprehensive_search(topic, keywords, max_sources=30)
        await asyncio.sleep(2.0)  # Simulate search time

        # Stage 3: Analyzing (80% progress)
        await self._update_session(session_id, stage="analyzing", progress=0.6)
        analyzed_sources = await self._analyze_sources(sources)
        await asyncio.sleep(1.0)  # Simulate analysis time

        # Stage 4: Synthesizing (100% progress)
        await self._update_session(session_id, stage="synthesizing", progress=0.9)
        result = await self._synthesize_findings(topic, analyzed_sources, depth="detailed")

        # Update session with results
        await self._update_session(
            session_id,
            executive_summary=result["executive_summary"],
            key_findings=result["key_findings"],
            sources=analyzed_sources,
            confidence_score=result["confidence_score"],
            progress=1.0,
        )

    async def _execute_academic_research(self, session_id: str) -> None:
        """Execute academic research pipeline (10 minutes, 50+ sources)."""
        topic = self._sessions[session_id]["topic"]

        # Stage 1: Planning (10% progress)
        await self._update_session(session_id, stage="planning", progress=0.02)
        keywords = await self._generate_academic_keywords(topic, max_keywords=15)
        await asyncio.sleep(1.0)  # Simulate planning time

        # Stage 2: Searching (40% progress)
        await self._update_session(session_id, stage="searching", progress=0.15)
        sources = await self._conduct_academic_search(topic, keywords, max_sources=50)
        await asyncio.sleep(3.0)  # Simulate academic search time

        # Stage 3: Analyzing (70% progress)
        await self._update_session(session_id, stage="analyzing", progress=0.5)
        analyzed_sources = await self._analyze_academic_sources(sources)
        await asyncio.sleep(2.0)  # Simulate analysis time

        # Stage 4: Synthesizing (90% progress)
        await self._update_session(session_id, stage="synthesizing", progress=0.8)
        result = await self._synthesize_academic_findings(topic, analyzed_sources)
        await asyncio.sleep(1.0)  # Simulate synthesis time

        # Update session with results
        await self._update_session(
            session_id,
            executive_summary=result["executive_summary"],
            key_findings=result["key_findings"],
            sources=analyzed_sources,
            confidence_score=result["confidence_score"],
            progress=1.0,
        )

    async def _generate_search_keywords(self, topic: str, max_keywords: int = 10) -> list[str]:
        """Generate search keywords from topic."""
        # TODO: Implement actual keyword generation using NLP
        # For now, create basic keywords from topic
        keywords = [topic]

        # Add some variations
        words = topic.lower().split()
        if len(words) > 1:
            keywords.extend(words)

        # Add common research terms
        keywords.extend(
            [
                f"{topic} overview",
                f"{topic} analysis",
                f"what is {topic}",
                f"{topic} research",
            ]
        )

        return keywords[:max_keywords]

    async def _generate_academic_keywords(self, topic: str, max_keywords: int = 15) -> list[str]:
        """Generate academic-focused keywords."""
        keywords = await self._generate_search_keywords(topic, max_keywords // 2)

        # Add academic-specific terms
        academic_terms = [
            f"{topic} study",
            f"{topic} literature review",
            f"{topic} scholarly articles",
            f"{topic} peer reviewed",
            f"{topic} methodology",
            f"{topic} theoretical framework",
            f"{topic} empirical research",
        ]

        keywords.extend(academic_terms)
        return keywords[:max_keywords]

    async def _conduct_web_search(self, topic: str, keywords: list[str], max_sources: int = 10) -> list[dict[str, Any]]:
        """Conduct web search for sources."""
        # TODO: Implement actual web search integration (Issue #3)
        # For now, generate mock sources
        sources = []

        for i in range(min(max_sources, len(keywords) * 2)):
            keyword = keywords[i % len(keywords)]
            sources.append(
                {
                    "title": f"Source about {keyword}",
                    "url": f"https://example.com/source-{i + 1}",
                    "type": "web",
                    "summary": f"This source discusses {keyword} in the context of {topic}.",
                    "credibility_score": 0.7 + (i % 3) * 0.1,
                    "date_published": "2024-01-01",
                    "source_quality": "reliable" if i % 2 == 0 else "moderate",
                    "content_length": 1000 + i * 100,
                }
            )

        return sources[:max_sources]

    async def _conduct_comprehensive_search(
        self, topic: str, keywords: list[str], max_sources: int = 30
    ) -> list[dict[str, Any]]:
        """Conduct comprehensive search across multiple sources."""
        # TODO: Implement actual comprehensive search (Issue #3)
        sources = await self._conduct_web_search(topic, keywords, max_sources // 2)

        # Add mock news sources
        for i in range(max_sources // 4):
            sources.append(
                {
                    "title": f"News: Recent developments in {topic}",
                    "url": f"https://news.com/article-{i + 1}",
                    "type": "news",
                    "summary": f"Latest news about {topic}",
                    "credibility_score": 0.8,
                    "date_published": "2024-12-01",
                    "source_quality": "high",
                    "content_length": 800,
                }
            )

        # Add mock blog sources
        for i in range(max_sources // 4):
            sources.append(
                {
                    "title": f"Expert Opinion: {topic} Analysis",
                    "url": f"https://blog.com/post-{i + 1}",
                    "type": "blog",
                    "summary": f"Expert analysis of {topic}",
                    "credibility_score": 0.6,
                    "date_published": "2024-11-01",
                    "source_quality": "moderate",
                    "content_length": 1200,
                }
            )

        return sources[:max_sources]

    async def _conduct_academic_search(self, topic: str, keywords: list[str], max_sources: int = 50) -> list[dict[str, Any]]:
        """Conduct academic search across scholarly databases."""
        # TODO: Implement actual academic search (Issue #3)
        sources = await self._conduct_comprehensive_search(topic, keywords, max_sources // 2)

        # Add mock academic sources
        for i in range(max_sources // 2):
            sources.append(
                {
                    "title": f"Scholarly Article: {topic} Research Study {i + 1}",
                    "url": f"https://scholar.com/paper-{i + 1}",
                    "type": "academic",
                    "summary": f"Peer-reviewed research on {topic}",
                    "credibility_score": 0.9,
                    "date_published": "2023-06-01",
                    "source_quality": "high",
                    "content_length": 5000,
                    "citations": 25 + i * 5,
                    "peer_reviewed": True,
                    "journal": f"Journal of {topic} Studies",
                    "authors": ["Dr. Smith", "Dr. Johnson"],
                    "doi": f"10.1000/journal.{i + 1}",
                }
            )

        return sources[:max_sources]

    async def _analyze_sources(self, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Analyze source credibility and extract key information."""
        analyzed_sources = []

        for source in sources:
            # TODO: Implement actual source analysis (bias detection, credibility scoring)
            analyzed_source = source.copy()

            # Add analysis metadata
            analyzed_source.update(
                {
                    "analysis_completed": True,
                    "bias_score": 0.2 + (hash(source["url"]) % 30) / 100,  # Mock bias score
                    "relevance_score": 0.7 + (hash(source["title"]) % 30) / 100,  # Mock relevance
                    "key_points": [
                        f"Key insight 1 from {source['title'][:30]}...",
                        f"Key insight 2 from {source['title'][:30]}...",
                    ],
                    "topics_covered": source["title"].split()[:3],
                }
            )

            analyzed_sources.append(analyzed_source)

        return analyzed_sources

    async def _analyze_academic_sources(self, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Analyze academic sources with scholarly metrics."""
        analyzed_sources = await self._analyze_sources(sources)

        for source in analyzed_sources:
            if source.get("type") == "academic":
                # Add academic-specific analysis
                source.update(
                    {
                        "impact_factor": 2.5 + (hash(source["url"]) % 20) / 10,  # Mock impact factor
                        "h_index": 15 + (hash(source["title"]) % 35),  # Mock h-index
                        "methodology_quality": ("high" if source.get("citations", 0) > 20 else "moderate"),
                        "research_type": "empirical",  # Mock research type
                    }
                )

        return analyzed_sources

    async def _synthesize_findings(
        self, topic: str, sources: list[dict[str, Any]], depth: str = "detailed"
    ) -> dict[str, Any]:
        """Synthesize research findings into coherent summary."""
        # TODO: Implement actual synthesis using NLP and AI

        high_quality_sources = [s for s in sources if s.get("credibility_score", 0) > 0.7]
        total_sources = len(sources)
        quality_ratio = len(high_quality_sources) / total_sources if total_sources > 0 else 0

        if depth == "surface":
            summary = (
                f"Quick analysis of {topic} based on {total_sources} sources reveals key insights. "
                f"Research shows diverse perspectives on this topic with {quality_ratio:.1%} high-quality sources."
            )

            findings = [
                f"Primary insight: {topic} is a significant area of study",
                f"Source diversity: {total_sources} sources across multiple types",
                f"Quality assessment: {len(high_quality_sources)} high-credibility sources",
            ]
            confidence = min(0.9, 0.5 + quality_ratio * 0.4)

        elif depth == "detailed":
            summary = (
                f"Comprehensive research on {topic} synthesized from {total_sources} diverse sources "
                f"provides detailed insights. Analysis of {len(high_quality_sources)} high-quality sources "
                f"reveals multiple perspectives and evidence-based conclusions."
            )

            findings = [
                f"Comprehensive overview: {topic} encompasses multiple dimensions",
                f"Evidence base: {total_sources} sources with {quality_ratio:.1%} high reliability",
                f"Research depth: Detailed analysis across {len({s.get('type', 'unknown') for s in sources})} source types",
                f"Quality assurance: {len(high_quality_sources)} sources exceed credibility threshold",
                "Coverage assessment: Analysis spans recent and historical perspectives",
            ]
            confidence = min(0.95, 0.6 + quality_ratio * 0.35)

        else:  # scholarly
            summary = (
                f"Scholarly analysis of {topic} draws from {total_sources} rigorously evaluated sources, "
                f"including {len([s for s in sources if s.get('type') == 'academic'])} peer-reviewed publications. "
                "This research synthesis provides evidence-based insights with high methodological rigor."
            )

            academic_sources = [s for s in sources if s.get("type") == "academic"]
            citation_count = sum(s.get("citations", 0) for s in academic_sources)

            findings = [
                f"Scholarly foundation: {len(academic_sources)} peer-reviewed sources",
                f"Citation impact: {citation_count} total citations across academic sources",
                "Methodological rigor: Sources evaluated for research quality and bias",
                "Evidence synthesis: Cross-validation of findings across multiple studies",
                "Knowledge gaps: Areas identified for future research",
                f"Theoretical frameworks: Multiple approaches to understanding {topic}",
            ]
            confidence = min(0.98, 0.7 + quality_ratio * 0.28)

        return {
            "executive_summary": summary,
            "key_findings": findings,
            "confidence_score": confidence,
            "synthesis_metadata": {
                "total_sources": total_sources,
                "high_quality_sources": len(high_quality_sources),
                "quality_ratio": quality_ratio,
                "synthesis_depth": depth,
                "synthesis_date": datetime.now(UTC).isoformat(),
            },
        }

    async def _synthesize_academic_findings(self, topic: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
        """Synthesize academic findings with scholarly rigor."""
        result = await self._synthesize_findings(topic, sources, depth="scholarly")

        # Add academic-specific metadata
        academic_sources = [s for s in sources if s.get("type") == "academic"]

        result["synthesis_metadata"].update(
            {
                "academic_sources": len(academic_sources),
                "total_citations": sum(s.get("citations", 0) for s in academic_sources),
                "peer_reviewed_ratio": (
                    len([s for s in academic_sources if s.get("peer_reviewed")]) / len(academic_sources)
                    if academic_sources
                    else 0
                ),
                "average_impact_factor": (
                    sum(s.get("impact_factor", 0) for s in academic_sources) / len(academic_sources)
                    if academic_sources
                    else 0
                ),
            }
        )

        return result

    def _setup_tools(self) -> None:
        """Register MCP tools with structured output schemas."""

        @self.mcp.tool()
        async def deep_research(topic: str, scope: str = "comprehensive") -> ResearchResult:
            """
            Conduct comprehensive deep research on a given topic.

            Args:
                topic: The research topic to investigate
                scope: Research scope - options: quick, comprehensive, academic

            Returns:
                Comprehensive research results with structured findings
            """
            self.logger.info(f"Starting deep research on topic: {topic} (scope: {scope})")

            try:
                # Create session and conduct research
                session_id = await self._create_session(topic, scope)
                session_data = await self._conduct_research(session_id)

                # Convert session data to ResearchResult
                result = ResearchResult(
                    session_id=session_data["session_id"],
                    topic=session_data["topic"],
                    scope=session_data["scope"],
                    executive_summary=session_data["executive_summary"] or "Research completed successfully",
                    key_findings=session_data["key_findings"],
                    sources=session_data["sources"],
                    confidence_score=session_data["confidence_score"],
                    metadata={
                        **session_data["metadata"],
                        "research_started": session_data["created_at"].isoformat() + "Z",
                        "research_completed": session_data["updated_at"].isoformat() + "Z",
                        "system_version": "0.1.0",
                        "mcp_protocol": self.config.mcp.protocol_version,
                        "status": session_data["status"],
                        "stage": session_data["stage"],
                        "progress": session_data["progress"],
                    },
                )

                self.logger.info(f"Research completed for session: {session_id}")
                return result

            except Exception as e:
                self.logger.error(f"Research failed: {e}")
                # Return error result
                session_id = str(uuid.uuid4())
                return ResearchResult(
                    session_id=session_id,
                    topic=topic,
                    scope=scope,
                    executive_summary=f"Research failed due to error: {str(e)}",
                    key_findings=[
                        "Research initialization failed",
                        f"Error: {str(e)}",
                        "Please try again or contact support",
                    ],
                    sources=[],
                    confidence_score=0.0,
                    metadata={
                        "error": str(e),
                        "research_started": datetime.now(UTC).isoformat() + "Z",
                        "system_version": "0.1.0",
                        "mcp_protocol": self.config.mcp.protocol_version,
                        "status": "failed",
                    },
                )

        @self.mcp.tool()
        async def research_status(session_id: str) -> ResearchProgress:
            """
            Get the current status of an ongoing research session.

            Args:
                session_id: Unique identifier for the research session

            Returns:
                Current research progress and status information
            """
            self.logger.info(f"Checking status for session: {session_id}")

            try:
                # Get session data
                session_data = self._sessions.get(session_id)

                if not session_data:
                    return ResearchProgress(
                        session_id=session_id,
                        status="not_found",
                        progress=0.0,
                        stage="unknown",
                        estimated_completion="Session not found",
                    )

                # Check if session is expired
                if session_data["expires_at"] < datetime.now(UTC):
                    return ResearchProgress(
                        session_id=session_id,
                        status="expired",
                        progress=session_data.get("progress", 0.0),
                        stage=session_data.get("stage", "unknown"),
                        estimated_completion="Session expired",
                    )

                # Calculate estimated completion time
                estimated_completion = await self._estimate_completion_time(session_id)

                return ResearchProgress(
                    session_id=session_id,
                    status=session_data["status"],
                    progress=session_data["progress"],
                    stage=session_data["stage"],
                    estimated_completion=estimated_completion,
                )

            except Exception as e:
                self.logger.error(f"Error checking status for session {session_id}: {e}")
                return ResearchProgress(
                    session_id=session_id,
                    status="error",
                    progress=0.0,
                    stage="error",
                    estimated_completion=f"Error: {str(e)}",
                )

        @self.mcp.tool()
        async def export_research(session_id: str, format: str = "markdown") -> ExportResult:
            """
            Export research results in the specified format with resource links.

            Args:
                session_id: Unique identifier for the research session
                format: Export format - options: markdown, pdf, json, html, txt

            Returns:
                Export results with resource links for downloadable content
            """
            self.logger.info(f"Exporting research for session: {session_id} as {format}")

            try:
                # Validate format
                supported_formats = ["markdown", "pdf", "json", "html", "txt"]
                if format not in supported_formats:
                    raise ValueError(f"Unsupported format '{format}'. Supported: {supported_formats}")

                # Get session data
                session_data = self._sessions.get(session_id)
                if not session_data:
                    raise ValueError(f"Session {session_id} not found")

                if session_data["status"] != "completed":
                    raise ValueError(f"Session {session_id} is not completed. Status: {session_data['status']}")

                # Generate export content
                export_content = await self._generate_export_content(session_data, format)

                # Calculate estimated file size
                content_size = len(export_content.encode("utf-8"))
                if content_size < 1024:
                    size = f"{content_size} B"
                elif content_size < 1024 * 1024:
                    size = f"{content_size / 1024:.1f} KB"
                else:
                    size = f"{content_size / (1024 * 1024):.1f} MB"

                # Generate resource URIs following MCP 2025-06-18 patterns
                base_uri = f"research://{session_id}"
                expires_at = (datetime.now(UTC) + timedelta(days=30)).isoformat() + "Z"

                return ExportResult(
                    session_id=session_id,
                    format=format,
                    resource_links=[
                        f"{base_uri}/report.{format}",  # Main research report
                        f"{base_uri}/sources.json",  # Source bibliography
                        f"{base_uri}/metadata.json",  # Research metadata
                        f"{base_uri}/raw_data.json",  # Raw research data
                        f"{base_uri}/citations.bib",  # Bibliography in BibTeX format
                    ],
                    size=size,
                    expires_at=expires_at,
                )

            except Exception as e:
                self.logger.error(f"Export failed for session {session_id}: {e}")
                # Return error result
                return ExportResult(
                    session_id=session_id,
                    format=format,
                    resource_links=[],
                    size="0 B",
                    expires_at=(datetime.now(UTC) + timedelta(days=1)).isoformat() + "Z",
                )

        @self.mcp.tool()
        async def get_research_resource(session_id: str, resource_type: str = "report") -> dict[str, Any]:
            """
            Retrieve a specific research resource by session ID and type.

            Args:
                session_id: Unique identifier for the research session
                resource_type: Type of resource - options: report, sources,
                    metadata, raw_data, citations

            Returns:
                Resource content with metadata and links
            """
            self.logger.info(f"Retrieving {resource_type} resource for session: {session_id}")

            # Generate resource URI
            resource_uri = f"research://{session_id}/{resource_type}"

            # TODO: Implement actual resource retrieval
            # For now, return structured resource information
            return {
                "session_id": session_id,
                "resource_type": resource_type,
                "resource_uri": resource_uri,
                "content_available": True,
                "last_updated": datetime.now(UTC).isoformat() + "Z",
                "size_bytes": 1024 * 50,  # 50KB placeholder
                "mime_type": self._get_mime_type_for_resource(resource_type),
                "description": self._get_resource_description(resource_type),
            }

    async def _estimate_completion_time(self, session_id: str) -> str:
        """Estimate time remaining for session completion."""
        session_data = self._sessions.get(session_id)
        if not session_data:
            return "Unknown"

        scope_config = session_data["metadata"].get("scope_config", {})
        total_timeout = scope_config.get("timeout_minutes", 5)
        current_progress = session_data["progress"]

        if current_progress <= 0:
            return f"{total_timeout} minutes"

        elapsed = datetime.now(UTC) - session_data["created_at"]
        elapsed_minutes = elapsed.total_seconds() / 60

        if current_progress >= 1.0:
            return "Completed"

        if session_data["status"] == "failed":
            return "Failed"

        estimated_total = elapsed_minutes / current_progress
        remaining = max(0, estimated_total - elapsed_minutes)

        if remaining < 1:
            return "< 1 minute"
        elif remaining < 60:
            return f"{int(remaining)} minutes"
        else:
            hours = int(remaining / 60)
            minutes = int(remaining % 60)
            return f"{hours}h {minutes}m"

    async def _generate_export_content(self, session_data: dict[str, Any], format: str) -> str:
        """Generate export content in specified format."""
        if format == "markdown":
            return await self._generate_markdown(session_data)
        elif format == "pdf":
            return await self._generate_pdf_placeholder(session_data)
        elif format == "json":
            return await self._generate_json(session_data)
        elif format == "html":
            return await self._generate_html(session_data)
        elif format == "txt":
            return await self._generate_text(session_data)
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def _generate_markdown(self, session_data: dict[str, Any]) -> str:
        """Generate markdown format research report."""
        content = f"""# Research Report: {session_data['topic']}

## Executive Summary

{session_data.get('executive_summary', 'Research summary not available.')}

## Research Metadata

- **Session ID**: {session_data['session_id']}
- **Research Scope**: {session_data['scope'].title()}
- **Status**: {session_data['status'].title()}
- **Confidence Score**: {session_data['confidence_score']:.2%}
- **Created**: {session_data['created_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}
- **Last Updated**: {session_data['updated_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}

## Key Findings

"""
        for i, finding in enumerate(session_data.get("key_findings", []), 1):
            content += f"{i}. {finding}\n"

        content += f"\n## Sources ({len(session_data.get('sources', []))} total)\n\n"

        for i, source in enumerate(session_data.get("sources", []), 1):
            content += f"### Source {i}: {source.get('title', 'Untitled')}\n\n"
            content += f"- **URL**: {source.get('url', 'N/A')}\n"
            content += f"- **Type**: {source.get('type', 'Unknown').title()}\n"
            content += f"- **Credibility Score**: {source.get('credibility_score', 0):.2f}\n"
            content += f"- **Published**: {source.get('date_published', 'Unknown')}\n"

            if source.get("summary"):
                content += f"- **Summary**: {source['summary']}\n"

            content += "\n"

        content += f"""## Report Information

This research report was generated by Topic Deep Diver, an automated deep research system.

- **Generated**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}
- **System Version**: 0.1.0
- **Research Method**: Automated multi-source analysis
- **Quality Assurance**: Credibility scoring and bias detection applied

---

*This report is valid until {(datetime.now(UTC) + timedelta(days=30)).strftime('%Y-%m-%d')}*
"""
        return content

    async def _generate_json(self, session_data: dict[str, Any]) -> str:
        """Generate JSON format research data."""
        data = {
            "session_info": {
                "session_id": session_data["session_id"],
                "topic": session_data["topic"],
                "scope": session_data["scope"],
                "status": session_data["status"],
                "stage": session_data["stage"],
                "progress": session_data["progress"],
                "confidence_score": session_data["confidence_score"],
                "created_at": session_data["created_at"].isoformat() + "Z",
                "updated_at": session_data["updated_at"].isoformat() + "Z",
                "expires_at": session_data["expires_at"].isoformat() + "Z",
            },
            "research_results": {
                "executive_summary": session_data.get("executive_summary"),
                "key_findings": session_data.get("key_findings", []),
                "sources": session_data.get("sources", []),
                "metadata": session_data.get("metadata", {}),
            },
            "export_info": {
                "generated_at": datetime.now(UTC).isoformat() + "Z",
                "format": "json",
                "system_version": "0.1.0",
            },
        }

        if session_data.get("error_message"):
            data["error_info"] = {
                "error_message": session_data["error_message"],
                "retry_count": session_data.get("retry_count", 0),
            }

        return json.dumps(data, indent=2, default=str)

    async def _generate_html(self, session_data: dict[str, Any]) -> str:
        """Generate HTML format research report."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report: {session_data['topic']}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        .header {{
            border-bottom: 2px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .metadata {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .sources {{
            margin-top: 30px;
        }}
        .source {{
            border-left: 4px solid #007acc;
            padding-left: 15px;
            margin: 20px 0;
        }}
        .confidence-score {{
            background: #e7f3ff;
            padding: 10px;
            border-radius: 5px;
            display: inline-block;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Research Report: {session_data['topic']}</h1>
        <div class="confidence-score">
            <strong>Confidence Score: {session_data['confidence_score']:.2%}</strong>
        </div>
    </div>

    <div class="metadata">
        <h2>Research Information</h2>
        <ul>
            <li><strong>Session ID:</strong> {session_data['session_id']}</li>
            <li><strong>Research Scope:</strong> {session_data['scope'].title()}</li>
            <li><strong>Status:</strong> {session_data['status'].title()}</li>
            <li><strong>Created:</strong> {session_data['created_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}</li>
            <li><strong>Last Updated:</strong> {session_data['updated_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}</li>
        </ul>
    </div>

    <h2>Executive Summary</h2>
    <p>{session_data.get('executive_summary', 'Research summary not available.')}</p>

    <h2>Key Findings</h2>
    <ol>
"""
        for finding in session_data.get("key_findings", []):
            html_content += f"        <li>{finding}</li>\n"

        html_content += f"""    </ol>

    <div class="sources">
        <h2>Sources ({len(session_data.get('sources', []))} total)</h2>
"""

        for i, source in enumerate(session_data.get("sources", []), 1):
            html_content += f"""        <div class="source">
            <h3>Source {i}: {source.get('title', 'Untitled')}</h3>
            <p><strong>URL:</strong> <a href="{source.get('url', '#')}">{source.get('url', 'N/A')}</a></p>
            <p><strong>Type:</strong> {source.get('type', 'Unknown').title()}</p>
            <p><strong>Credibility Score:</strong> {source.get('credibility_score', 0):.2f}</p>
            <p><strong>Published:</strong> {source.get('date_published', 'Unknown')}</p>
"""
            if source.get("summary"):
                html_content += f"            <p><strong>Summary:</strong> {source['summary']}</p>\n"

            html_content += "        </div>\n"

        html_content += f"""    </div>

    <div class="footer">
        <p><strong>Report Generated:</strong> {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        <p><strong>System:</strong> Topic Deep Diver v0.1.0</p>
        <p><strong>Method:</strong> Automated multi-source analysis with credibility scoring</p>
        <p><em>This report is valid until {(datetime.now(UTC) + timedelta(days=30)).strftime('%Y-%m-%d')}</em></p>
    </div>
</body>
</html>"""
        return html_content

    async def _generate_text(self, session_data: dict[str, Any]) -> str:
        """Generate plain text format research report."""
        content = f"""RESEARCH REPORT: {session_data['topic'].upper()}
{'=' * (len(session_data['topic']) + 17)}

EXECUTIVE SUMMARY
-----------------
{session_data.get('executive_summary', 'Research summary not available.')}

RESEARCH INFORMATION
-------------------
Session ID: {session_data['session_id']}
Research Scope: {session_data['scope'].title()}
Status: {session_data['status'].title()}
Confidence Score: {session_data['confidence_score']:.2%}
Created: {session_data['created_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}
Last Updated: {session_data['updated_at'].strftime('%Y-%m-%d %H:%M:%S UTC')}

KEY FINDINGS
-----------
"""
        for i, finding in enumerate(session_data.get("key_findings", []), 1):
            content += f"{i}. {finding}\n"

        content += f"\nSOURCES ({len(session_data.get('sources', []))} total)\n"
        content += "-" * 40 + "\n\n"

        for i, source in enumerate(session_data.get("sources", []), 1):
            content += f"Source {i}: {source.get('title', 'Untitled')}\n"
            content += f"URL: {source.get('url', 'N/A')}\n"
            content += f"Type: {source.get('type', 'Unknown').title()}\n"
            content += f"Credibility Score: {source.get('credibility_score', 0):.2f}\n"
            content += f"Published: {source.get('date_published', 'Unknown')}\n"

            if source.get("summary"):
                content += f"Summary: {source['summary']}\n"

            content += "\n" + "-" * 40 + "\n\n"

        content += f"""REPORT INFORMATION
------------------
Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}
System Version: 0.1.0
Research Method: Automated multi-source analysis
Quality Assurance: Credibility scoring and bias detection applied

This report is valid until {(datetime.now(UTC) + timedelta(days=30)).strftime('%Y-%m-%d')}
"""
        return content

    async def _generate_pdf_placeholder(self, session_data: dict[str, Any]) -> str:
        """Generate PDF format placeholder (returns markdown for PDF conversion)."""
        markdown_content = await self._generate_markdown(session_data)

        # For now, return markdown with PDF-specific metadata
        # In a production system, this would use libraries like reportlab or weasyprint
        pdf_metadata = f"""<!-- PDF Metadata
Title: Research Report - {session_data['topic']}
Author: Topic Deep Diver System
Subject: Automated Research Analysis
Creator: Topic Deep Diver v0.1.0
Producer: MCP Research Server
Keywords: research, analysis, {session_data['scope']}
Creation Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}
-->

{markdown_content}

<!-- PDF Generation Notes:
This content is optimized for PDF conversion using tools like pandoc, weasyprint, or reportlab.
Recommended command: pandoc input.md -o output.pdf --pdf-engine=wkhtmltopdf
-->"""
        return pdf_metadata

    def _get_mime_type_for_resource(self, resource_type: str) -> str:
        """Get MIME type for a resource type."""
        mime_types = {
            "report": "text/markdown",
            "sources": "application/json",
            "metadata": "application/json",
            "raw_data": "application/json",
            "citations": "application/x-bibtex",
        }
        return mime_types.get(resource_type, "application/octet-stream")

    def _get_resource_description(self, resource_type: str) -> str:
        """Get human-readable description for a resource type."""
        descriptions = {
            "report": "Main research report with analysis and findings",
            "sources": "Comprehensive bibliography and source information",
            "metadata": "Research session metadata and configuration",
            "raw_data": "Raw research data and intermediate results",
            "citations": "Bibliography in BibTeX format for academic use",
        }
        return descriptions.get(resource_type, "Research resource")

    def run_sync(self) -> None:
        """Start the MCP server synchronously."""
        self.logger.info("Topic Deep Diver MCP Server starting...")
        self.logger.info(f"MCP Protocol Version: {self.config.mcp.protocol_version}")
        self.logger.info(f"Transport: {self.config.mcp.transport}")
        self.logger.info("Structured output support: ENABLED")

        # Run the FastMCP server with specified transport
        # The FastMCP framework automatically handles protocol version headers
        transport = cast(Literal["stdio", "sse", "streamable-http"], self.config.mcp.transport)
        self.mcp.run(transport=transport)

    async def run(self) -> None:
        """Start the MCP server asynchronously."""
        self.logger.info("Topic Deep Diver MCP Server starting...")
        self.logger.info(f"MCP Protocol Version: {self.config.mcp.protocol_version}")
        self.logger.info(f"Transport: {self.config.mcp.transport}")
        self.logger.info("Structured output support: ENABLED")

        # FastMCP async runner - use sync version for compatibility
        transport = cast(Literal["stdio", "sse", "streamable-http"], self.config.mcp.transport)
        self.mcp.run(transport=transport)
