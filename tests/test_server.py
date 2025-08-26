"""
Basic test to verify MCP server functionality.
"""

import asyncio
import pytest
from unittest.mock import Mock

from topic_deep_diver.server import DeepResearchServer


@pytest.mark.asyncio
async def test_server_initialization():
    """Test that the server initializes correctly."""
    server = DeepResearchServer()
    
    assert server.mcp_version == "2025-06-18"
    assert server.oauth_metadata["resource_indicators_required"] is True
    assert server.mcp is not None


@pytest.mark.asyncio 
async def test_deep_research_tool():
    """Test the deep_research tool basic functionality."""
    server = DeepResearchServer()
    
    # Get the tool function
    tools = server.mcp._tools
    assert "deep_research" in tools
    
    # Test with mock call - we'd need to properly mock the tool call in a real test
    # For now, just verify the tool is registered


@pytest.mark.asyncio
async def test_research_status_tool():
    """Test the research_status tool basic functionality."""
    server = DeepResearchServer()
    
    # Get the tool function
    tools = server.mcp._tools
    assert "research_status" in tools


@pytest.mark.asyncio
async def test_export_research_tool():
    """Test the export_research tool basic functionality."""
    server = DeepResearchServer()
    
    # Get the tool function
    tools = server.mcp._tools
    assert "export_research" in tools