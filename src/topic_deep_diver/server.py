"""
Core MCP server implementation for Topic Deep Diver.
"""

import uuid
from typing import Any, Literal, cast

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
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


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

        # Create FastMCP instance with structured content
        self.mcp = FastMCP(
            name=self.config.server.name,
            instructions=(
                "Advanced deep research MCP server providing comprehensive "
                "topic analysis and synthesis."
            ),
        )
        self._setup_tools()

        self.logger.info(
            "DeepResearchServer initialized with structured output support"
        )

    def _setup_tools(self) -> None:
        """Register MCP tools with structured output schemas."""

        @self.mcp.tool()
        async def deep_research(
            topic: str, scope: str = "comprehensive"
        ) -> ResearchResult:
            """
            Conduct comprehensive deep research on a given topic.

            Args:
                topic: The research topic to investigate
                scope: Research scope - options: quick, comprehensive, academic

            Returns:
                Comprehensive research results with structured findings
            """
            self.logger.info(
                f"Starting deep research on topic: {topic} (scope: {scope})"
            )

            # Generate unique session ID
            session_id = str(uuid.uuid4())

            # TODO: Implement actual research logic
            # For now, return structured placeholder data
            result = ResearchResult(
                session_id=session_id,
                topic=topic,
                scope=scope,
                executive_summary=(
                    f"Comprehensive research on '{topic}' reveals significant "
                    "opportunities for analysis. This automated deep research "
                    "system is currently in development and will provide "
                    "extensive multi-source analysis."
                ),
                key_findings=[
                    "Research framework successfully initialized",
                    "Topic validation completed",
                    "Search strategy formulated",
                    "Implementation in progress",
                ],
                sources=[
                    {
                        "title": "System Initialization",
                        "url": "internal://system",
                        "type": "system",
                        "credibility_score": 1.0,
                        "summary": "System successfully initialized for research",
                    }
                ],
                confidence_score=0.85,
                metadata={
                    "research_started": "2025-08-26T13:59:34Z",
                    "system_version": "0.1.0",
                    "mcp_protocol": self.config.mcp.protocol_version,
                },
            )

            self.logger.info(f"Research completed for session: {session_id}")
            return result

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

            # TODO: Implement actual status tracking with persistent storage
            return ResearchProgress(
                session_id=session_id,
                status="in_progress",
                progress=0.65,
                stage="source_analysis",
                estimated_completion="2 minutes",
            )

        @self.mcp.tool()
        async def export_research(
            session_id: str, format: str = "markdown"
        ) -> ExportResult:
            """
            Export research results in the specified format with resource links.

            Args:
                session_id: Unique identifier for the research session
                format: Export format - options: markdown, pdf, json, html

            Returns:
                Export results with resource links for downloadable content
            """
            self.logger.info(
                f"Exporting research for session: {session_id} as {format}"
            )

            # Generate resource URIs following MCP 2025-06-18 patterns
            base_uri = f"research://{session_id}"

            # TODO: Implement actual export functionality with file generation
            # For now, return structured resource links
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
                size="2.5MB",
                expires_at="2025-09-25T11:46:14Z",
            )

        # Add a new tool for accessing research resources
        @self.mcp.tool()
        async def get_research_resource(
            session_id: str, resource_type: str = "report"
        ) -> dict[str, Any]:
            """
            Retrieve a specific research resource by session ID and type.

            Args:
                session_id: Unique identifier for the research session
                resource_type: Type of resource - options: report, sources,
                    metadata, raw_data, citations

            Returns:
                Resource content with metadata and links
            """
            self.logger.info(
                f"Retrieving {resource_type} resource for session: {session_id}"
            )

            # Generate resource URI
            resource_uri = f"research://{session_id}/{resource_type}"

            # TODO: Implement actual resource retrieval
            # For now, return structured resource information
            return {
                "session_id": session_id,
                "resource_type": resource_type,
                "resource_uri": resource_uri,
                "content_available": True,
                "last_updated": "2025-08-26T14:00:00Z",
                "size_bytes": 1024 * 50,  # 50KB placeholder
                "mime_type": self._get_mime_type_for_resource(resource_type),
                "description": self._get_resource_description(resource_type),
            }

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
        transport = cast(
            Literal["stdio", "sse", "streamable-http"], self.config.mcp.transport
        )
        self.mcp.run(transport=transport)

    async def run(self) -> None:
        """Start the MCP server asynchronously."""
        self.logger.info("Topic Deep Diver MCP Server starting...")
        self.logger.info(f"MCP Protocol Version: {self.config.mcp.protocol_version}")
        self.logger.info(f"Transport: {self.config.mcp.transport}")
        self.logger.info("Structured output support: ENABLED")

        # For async context, use the appropriate async runner based on transport
        if self.config.mcp.transport == "stdio":
            await self.mcp.run_stdio_async()
        elif self.config.mcp.transport == "sse":
            await self.mcp.run_sse_async()
        elif self.config.mcp.transport == "streamable-http":
            await self.mcp.run_streamable_http_async()
        else:
            # Fall back to sync run for unknown transports
            transport = cast(
                Literal["stdio", "sse", "streamable-http"], self.config.mcp.transport
            )
            self.mcp.run(transport=transport)
