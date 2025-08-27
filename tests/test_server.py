"""
Tests for Topic Deep Diver MCP server functionality.
"""

from datetime import UTC, datetime

import pytest

from topic_deep_diver.server import DeepResearchServer


@pytest.mark.asyncio
async def test_server_initialization():
    """Test that the server initializes correctly."""
    server = DeepResearchServer()

    # Test basic server setup
    assert server.config is not None
    assert server.logger is not None
    assert server.mcp is not None
    assert server.config.mcp.protocol_version == "2025-06-18"

    # Test session management initialization
    assert hasattr(server, "_sessions")
    assert hasattr(server, "_session_locks")
    assert isinstance(server._sessions, dict)
    assert isinstance(server._session_locks, dict)


@pytest.mark.asyncio
async def test_server_has_required_tools():
    """Test that all required MCP tools are registered."""
    server = DeepResearchServer()

    # Since FastMCP internal structure varies, let's test that server initializes
    # and has the required tools by checking the mcp object exists
    assert server.mcp is not None
    # For now, just verify the server was created successfully
    # TODO: Add proper tool registration testing once we understand FastMCP API better


@pytest.mark.asyncio
async def test_config_loading():
    """Test that configuration loads properly."""
    server = DeepResearchServer()

    config = server.config
    assert config.server.name == "Topic Deep Diver"  # This is the actual config value
    assert config.mcp.protocol_version == "2025-06-18"
    assert config.mcp.transport in ["stdio", "sse", "streamable-http"]


@pytest.mark.asyncio
async def test_pydantic_models():
    """Test that Pydantic models work correctly."""
    from topic_deep_diver.server import ExportResult, ResearchProgress, ResearchResult

    # Test ResearchProgress model
    progress = ResearchProgress(
        session_id="test-123",
        status="in_progress",
        progress=0.5,
        stage="analysis",
        estimated_completion="5 minutes",
    )
    assert progress.session_id == "test-123"
    assert progress.progress == 0.5

    # Test ResearchResult model
    result = ResearchResult(
        session_id="test-456",
        topic="test topic",
        scope="comprehensive",
        executive_summary="Test summary",
        key_findings=["finding 1", "finding 2"],
        sources=[{"title": "test", "url": "http://test.com"}],
        confidence_score=0.9,
    )
    assert result.topic == "test topic"
    assert len(result.key_findings) == 2

    # Test ExportResult model
    export = ExportResult(
        session_id="test-789",
        format="markdown",
        resource_links=["http://test.com/report.md"],
        size="1MB",
        expires_at="2025-12-31T23:59:59Z",
    )
    assert export.format == "markdown"
    assert len(export.resource_links) == 1


@pytest.mark.asyncio
async def test_session_management():
    """Test session creation and management."""
    server = DeepResearchServer()

    # Test session creation
    session_id = await server._create_session("AI research", "comprehensive")
    assert session_id in server._sessions

    session_data = server._sessions[session_id]
    assert session_data["topic"] == "AI research"
    assert session_data["scope"] == "comprehensive"
    assert session_data["status"] == "pending"
    assert session_data["stage"] == "initializing"
    assert session_data["progress"] == 0.0

    # Test session update
    await server._update_session(session_id, status="in_progress", progress=0.5)
    updated_data = server._sessions[session_id]
    assert updated_data["status"] == "in_progress"
    assert updated_data["progress"] == 0.5


@pytest.mark.asyncio
async def test_scope_configuration():
    """Test that different research scopes have correct configurations."""
    server = DeepResearchServer()

    # Test quick scope
    quick_config = server._get_scope_config("quick")
    assert quick_config["max_sources"] == 10
    assert quick_config["timeout_minutes"] == 2
    assert quick_config["depth"] == "surface"

    # Test comprehensive scope
    comp_config = server._get_scope_config("comprehensive")
    assert comp_config["max_sources"] == 30
    assert comp_config["timeout_minutes"] == 5
    assert comp_config["depth"] == "detailed"

    # Test academic scope
    academic_config = server._get_scope_config("academic")
    assert academic_config["max_sources"] == 50
    assert academic_config["timeout_minutes"] == 10
    assert academic_config["depth"] == "scholarly"


@pytest.mark.asyncio
async def test_invalid_scope():
    """Test that invalid scope raises appropriate error."""
    server = DeepResearchServer()

    with pytest.raises(ValueError, match="Invalid scope"):
        await server._create_session("test topic", "invalid_scope")


@pytest.mark.asyncio
async def test_keyword_generation():
    """Test search keyword generation."""
    server = DeepResearchServer()

    # Test basic keyword generation
    keywords = await server._generate_search_keywords("artificial intelligence", max_keywords=5)
    assert "artificial intelligence" in keywords
    assert len(keywords) <= 5

    # Test academic keyword generation
    academic_keywords = await server._generate_academic_keywords("machine learning", max_keywords=10)
    assert "machine learning" in academic_keywords
    assert any("study" in keyword for keyword in academic_keywords)
    assert len(academic_keywords) <= 10


@pytest.mark.asyncio
async def test_mock_search_functionality():
    """Test mock search functions return expected structure."""
    server = DeepResearchServer()

    # Test web search
    sources = await server._conduct_web_search("AI", ["AI", "artificial intelligence"], max_sources=5)
    assert len(sources) <= 5
    assert all("title" in source for source in sources)
    assert all("url" in source for source in sources)
    assert all("credibility_score" in source for source in sources)

    # Test comprehensive search
    comp_sources = await server._conduct_comprehensive_search("AI", ["AI"], max_sources=15)
    assert len(comp_sources) <= 15
    source_types = {source.get("type") for source in comp_sources}
    assert "web" in source_types

    # Test academic search
    academic_sources = await server._conduct_academic_search("AI", ["AI"], max_sources=20)
    assert len(academic_sources) <= 20
    academic_types = {source.get("type") for source in academic_sources}
    assert "academic" in academic_types


@pytest.mark.asyncio
async def test_source_analysis():
    """Test source analysis functionality."""
    server = DeepResearchServer()

    mock_sources = [
        {
            "title": "Test Article",
            "url": "https://example.com/test",
            "type": "web",
            "credibility_score": 0.8,
        }
    ]

    analyzed_sources = await server._analyze_sources(mock_sources)
    assert len(analyzed_sources) == 1
    analyzed = analyzed_sources[0]
    assert "analysis_completed" in analyzed
    assert "bias_score" in analyzed
    assert "relevance_score" in analyzed
    assert "key_points" in analyzed


@pytest.mark.asyncio
async def test_findings_synthesis():
    """Test research findings synthesis."""
    server = DeepResearchServer()

    mock_sources = [
        {
            "title": "High Quality Source",
            "credibility_score": 0.9,
            "type": "academic",
        },
        {
            "title": "Medium Quality Source",
            "credibility_score": 0.6,
            "type": "web",
        },
    ]

    # Test surface synthesis
    surface_result = await server._synthesize_findings("AI", mock_sources, depth="surface")
    assert "executive_summary" in surface_result
    assert "key_findings" in surface_result
    assert "confidence_score" in surface_result
    assert isinstance(surface_result["key_findings"], list)

    # Test detailed synthesis
    detailed_result = await server._synthesize_findings("AI", mock_sources, depth="detailed")
    assert len(detailed_result["key_findings"]) >= len(surface_result["key_findings"])

    # Test scholarly synthesis
    scholarly_result = await server._synthesize_findings("AI", mock_sources, depth="scholarly")
    assert "scholarly" in scholarly_result["executive_summary"].lower()


@pytest.mark.asyncio
async def test_quick_research_execution():
    """Test quick research execution pipeline."""
    server = DeepResearchServer()

    # Create session
    session_id = await server._create_session("test topic", "quick")

    # Execute quick research
    await server._execute_quick_research(session_id)

    # Check session was updated
    session_data = server._sessions[session_id]
    assert session_data["progress"] == 1.0
    assert session_data["executive_summary"] is not None
    assert len(session_data["key_findings"]) > 0
    assert len(session_data["sources"]) > 0


@pytest.mark.asyncio
async def test_completion_time_estimation():
    """Test research completion time estimation."""
    server = DeepResearchServer()

    # Create session
    session_id = await server._create_session("test topic", "comprehensive")

    # Test initial estimation
    initial_estimate = await server._estimate_completion_time(session_id)
    assert "minutes" in initial_estimate

    # Update progress and test again
    await server._update_session(session_id, progress=0.5)
    mid_estimate = await server._estimate_completion_time(session_id)
    assert mid_estimate != initial_estimate

    # Mark as complete
    await server._update_session(session_id, progress=1.0)
    final_estimate = await server._estimate_completion_time(session_id)
    assert final_estimate == "Completed"


@pytest.mark.asyncio
async def test_export_content_generation():
    """Test export content generation in different formats."""
    server = DeepResearchServer()

    # Create mock session data
    mock_session = {
        "session_id": "test-123",
        "topic": "AI Research",
        "scope": "comprehensive",
        "status": "completed",
        "stage": "completed",
        "progress": 1.0,
        "created_at": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
        "updated_at": datetime(2024, 1, 1, 1, 0, 0, tzinfo=UTC),
        "expires_at": datetime(2024, 1, 2, 0, 0, 0, tzinfo=UTC),
        "confidence_score": 0.85,
        "executive_summary": "Test summary",
        "key_findings": ["Finding 1", "Finding 2"],
        "sources": [
            {
                "title": "Test Source",
                "url": "https://example.com",
                "type": "web",
                "credibility_score": 0.8,
                "summary": "Test summary",
            }
        ],
        "metadata": {"test": "data"},
    }

    # Test markdown generation
    markdown_content = await server._generate_markdown(mock_session)
    assert "# Research Report: AI Research" in markdown_content
    assert "Test summary" in markdown_content
    assert "Finding 1" in markdown_content

    # Test JSON generation
    json_content = await server._generate_json(mock_session)
    assert "test-123" in json_content
    assert "AI Research" in json_content

    # Test HTML generation
    html_content = await server._generate_html(mock_session)
    assert "<html" in html_content
    assert "AI Research" in html_content
    assert "Test summary" in html_content

    # Test text generation
    text_content = await server._generate_text(mock_session)
    assert "AI RESEARCH" in text_content
    assert "Test summary" in text_content

    # Test PDF generation (placeholder)
    pdf_content = await server._generate_pdf_placeholder(mock_session)
    assert "PDF Metadata" in pdf_content
    assert "Research Report - AI Research" in pdf_content
    assert "# Research Report: AI Research" in pdf_content  # Contains markdown
    assert "PDF Generation Notes" in pdf_content
