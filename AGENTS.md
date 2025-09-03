# Topic Deep Diver - Project Agents & Knowledge Base

## Pre-Commit Quality Gates (MANDATORY)

**CRITICAL: Pre-commit hooks are now ENFORCED!**

### Setup (One-time)
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Or if using uv
uv add --dev pre-commit
uv run pre-commit install
```

### What Happens Automatically
- **Before each commit**: All quality checks run automatically
- **If checks fail**: Commit is blocked until issues are fixed
- **Auto-fixes**: Ruff and Black will auto-fix what they can

### Manual Quality Checks (if needed)
```bash
# Quick check (recommended)
uv run pytest tests/ && uv run ruff check . && uv run black --check . && uv run mypy src/

# Individual checks
uv run pytest tests/ -v          # Tests
uv run ruff check --fix .       # Lint (auto-fix)
uv run black .                  # Format (auto-fix)
uv run mypy src/                # Types
```

**If any fail, fix issues before committing.** CI will block your PR otherwise.

## Build Commands & Development

```bash
# Setup
uv sync --dev

# Run server
uv run python -m topic_deep_diver

# Run tests
uv run pytest tests/ -v
uv run pytest tests/test_file.py::test_function

# Quality checks (same as above)
uv run ruff check --fix .    # Lint + auto-fix
uv run black .               # Format
uv run mypy src/             # Types
```

## CI Pipeline

- **Triggers**: Push/PR to any branch
- **Gates**: pytest, ruff, black, mypy (all must pass)
- **Environment**: Python 3.11+, Ubuntu 24.04
- **Debug locally**: Run same commands as pre-commit checks

**CI Failure = PR blocked** - Fix locally first!

## Local CI Testing with act

Test the GitHub Actions pipeline locally before pushing to avoid CI failures:

### Prerequisites
```bash
# Install act (GitHub Actions runner)
curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Or using package manager
# Ubuntu/Debian: sudo apt install act
# macOS: brew install act
```

### Test Commands
```bash
# List available workflows
act --list

# Test push event (recommended for most cases)
act push

# Test specific job only
act push --job test

# Test pull request event
act pull_request

# Run with verbose output for debugging
act push --verbose

# Run with timeout to avoid hanging
timeout 300 act push --job test --verbose

# Use specific Docker image (if needed)
act push --container-architecture linux/amd64
```

### Common act Commands
```bash
# Test only the test job
act push --job test

# Use specific event file
act push --eventpath .github/workflows/event.json

# Skip certain steps
act push --skip-checkout

# Use different container image
act push --container-architecture linux/amd64

# Debug with more output
act push --verbose --log-prefix
```

### Troubleshooting act Issues
```bash
# Check Docker is running
docker --version
docker ps

# Clean up act cache
rm -rf ~/.cache/act

# Use different container image
act push --container-architecture linux/amd64

# Check workflow syntax
act push --list
```

### When to Use act
- **Before pushing** to catch CI issues locally
- **When CI fails** to reproduce the issue
- **When testing workflow changes** before committing
- **For faster iteration** than waiting for GitHub Actions

### act vs GitHub Actions
- **act**: Runs locally with Docker, faster feedback
- **GitHub Actions**: Official CI, runs on GitHub's infrastructure
- **Use act first**, then verify with GitHub Actions for final validation

## Code Style Guidelines

- **Language**: Python 3.11+, async/await for I/O
- **Imports**: Absolute imports, group stdlib/3rd-party/local
- **Formatting**: Black (88 char), ruff linting, type hints required
- **Naming**: snake_case functions/vars, PascalCase classes, UPPER_CASE constants
- **Error handling**: Specific exceptions, structured logging
- **MCP compliance**: 2025-06-18 spec, FastMCP, structured outputs
- **Testing**: pytest with asyncio, descriptive names, mock APIs
- **Documentation**: Docstrings with Args/Returns

## Project Structure

```
topic-deep-diver/
├── src/topic_deep_diver/      # Main package (src/ prefix)
│   ├── server.py              # MCP server
│   ├── config.py              # Configuration
│   ├── logging_config.py      # Logging setup
│   └── main.py                # Entry point
├── tests/                     # Test suite
├── config/                    # Config files
└── pyproject.toml             # Project metadata & deps
```

## MCP Tools API

**Core Tools:**
1. `deep_research(topic, scope)` - Main research orchestrator
2. `research_status(session_id)` - Progress monitoring
3. `export_research(session_id, format)` - Results export

## Configuration

- Config: `config/config.yaml` (copy from example)
- Env var: `CONFIG_PATH` for custom location
- Dependencies: Redis caching, optional API keys

## Project Overview

**Topic Deep Diver** is a fully automated deep research MCP server providing comprehensive topic analysis using multiple search engines and AI-powered synthesis. Designed to rival commercial solutions like Perplexity's Deep Research.

**Core Pipeline:** User Query → Research Planner → Multi-Search Engine → Content Processor → Knowledge Synthesizer → Structured Report

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- MCP Server Setup with 2025-06-18 SDK
- OAuth Resource Server classification
- Core tools: deep_research, research_status, export_research
- Basic SearXNG + Fetch MCP integration

### Phase 2: Intelligence Layer (Weeks 3-4)
- Query Processing Engine with topic decomposition
- Source Analysis (credibility, bias detection, deduplication)
- Information synthesis and citation tracking

### Phase 3: Advanced Features (Weeks 5-6)
- Academic integration (Google Scholar, PubMed)
- Real-time processing and dynamic strategy adjustment
- Quality assurance and fact-checking

### Phase 4: Optimization & Security (Weeks 7-8)
- RFC 8707 compliance and secure token handling
- Performance optimization and caching
- Comprehensive testing and documentation

## Key Technologies

**Search Integration:**
- SearXNG, Brave Search, Google/Bing APIs
- Academic: Google Scholar, Semantic Scholar, PubMed
- Specialized: arXiv, SSRN, patent databases

**Content Processing:**
- Fetch MCP server, Firecrawl, Jina Reader
- Source credibility scoring (0-100 scale)
- Bias detection and content deduplication

**MCP 2025-06-18 Features:**
- Structured tool output with JSON reports
- OAuth Resource Server authentication
- Resource links for research artifacts
- Enhanced security best practices

## Success Metrics

- **Completion time**: <4 minutes for comprehensive topics
- **Source diversity**: 10+ unique credible sources per report
- **Accuracy**: >90% fact verification
- **Quality**: >85% topic coverage, >80/100 credibility scores

## References

- [MCP 2025-06-18 Specification](https://modelcontextprotocol.io/specification/2025-06-18/)
- [Perplexity Deep Research](https://www.deepestresearch.com/)
- [Academic Search Engines](https://paperguide.ai/blog/academic-search-engines/)