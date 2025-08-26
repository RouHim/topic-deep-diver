"""
Main entry point for the Topic Deep Diver MCP server.
"""

import asyncio
from typing import Any

from .server import DeepResearchServer
from .logging_config import setup_logging, get_logger


def setup_app_logging() -> None:
    """Set up application logging configuration."""
    setup_logging()


async def main() -> None:
    """Main entry point for the MCP server."""
    setup_app_logging()
    logger = get_logger("main")
    
    try:
        server = DeepResearchServer()
        logger.info("Starting Topic Deep Diver MCP Server...")
        await server.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


def main_sync() -> None:
    """Synchronous entry point for the MCP server."""
    setup_app_logging()
    logger = get_logger("main")
    
    try:
        server = DeepResearchServer()
        logger.info("Starting Topic Deep Diver MCP Server...")
        server.run_sync()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    # Use sync version to avoid asyncio issues
    main_sync()