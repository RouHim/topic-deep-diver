# Topic Deep Diver 🛰️

A fully automated deep research MCP (Model Context Protocol) server that provides comprehensive topic analysis using multiple search engines and AI-powered synthesis. Built with MCP 2025-06-18 specification compliance.

## Overview

Topic Deep Diver rivals commercial solutions like Perplexity's Deep Research while maintaining full automation and extensive online search capabilities. The system performs autonomous multi-step research, evaluates sources, and synthesizes findings into comprehensive reports—all without user intervention.

## Key Features

### 🤖 Fully Automated Research
- Zero user intervention during research process
- Autonomous decision-making for search strategies
- Intelligent stopping criteria based on information saturation
- Complete end-to-end research pipeline

### 🔍 Comprehensive Search Integration
- **Multi-Tier Search Strategy**: Web, academic, and specialized databases
- **15+ Search Engines**: SearXNG, Google Scholar, PubMed, arXiv, and more
- **Real-Time Content Extraction**: HTML to markdown conversion with metadata
- **Source Diversity Optimization**: Ensures balanced perspective coverage

### 🧠 AI-Powered Intelligence
- **Query Decomposition**: Breaks complex topics into structured sub-questions
- **Source Credibility Scoring**: 0-100 scale with bias detection
- **Information Synthesis**: Multi-source aggregation with citation tracking
- **Gap Identification**: Automatically identifies and fills knowledge gaps

### 🔒 MCP 2025-06-18 Compliance
- **Structured Tool Output**: JSON-formatted research reports
- **OAuth Resource Server**: Secure authentication with Resource Indicators (RFC 8707)
- **Resource Links**: Efficient handling of large research artifacts
- **Enhanced Security**: Follows latest MCP security best practices

## MCP Tools

The server exposes three core tools for MCP clients (Claude, OpenCode, Cline):

### `deep_research(topic: str, scope: str = "comprehensive")`
Main research orchestrator that performs autonomous deep research on any topic.

**Parameters:**
- `topic` (required): The research topic or question
- `scope` (optional): Research depth - "quick", "comprehensive", or "academic"

**Returns:** Structured research report with findings, sources, and citations

### `research_status(session_id: str)`
Monitor the progress of ongoing research sessions.

**Parameters:**
- `session_id` (required): Unique identifier for the research session

**Returns:** Real-time progress updates and current research stage

### `export_research(session_id: str, format: str = "markdown")`
Export research findings in various formats.

**Parameters:**
- `session_id` (required): Research session to export
- `format` (optional): Output format - "markdown", "pdf", "json", "html"

**Returns:** Resource link to the exported research report

## Research Pipeline

```
User Query → Research Planner → Multi-Search Engine → Content Processor → Knowledge Synthesizer → Structured Report
```

### 1. Research Planner Agent
- Analyzes topic complexity and scope
- Generates comprehensive search strategy
- Creates research taxonomy and keywords
- Determines stopping criteria automatically

### 2. Multi-Search Orchestrator
- Parallel execution across multiple search engines
- Dynamic search refinement based on results
- Source diversity optimization
- Real-time result quality assessment

### 3. Content Analysis Engine
- Automatic source credibility scoring
- Content deduplication and clustering
- Bias detection and perspective analysis
- Information freshness validation

### 4. Knowledge Synthesizer
- Cross-source fact verification
- Narrative structure generation
- Citation tracking and management
- Gap identification and resolution

## Architecture

### Search Integration Strategy

**Primary Search Engines (General Web):**
- SearXNG instances (privacy-focused)
- Brave Search API (privacy-respecting)
- Google/Bing Search APIs (comprehensive coverage)

**Academic Search Engines:**
- Google Scholar API (200M+ scholarly articles)
- PubMed API (30M+ medical citations)
- arXiv API (STEM preprints)
- Crossref API (scholarly metadata)

**Content Extraction Tools:**
- Fetch MCP server integration
- Firecrawl for JavaScript-heavy sites
- Jina Reader for clean text extraction

### Autonomous Decision-Making

**Query Decomposition Algorithm:**
- NLP-based key concept extraction
- Question type identification (factual, analytical, comparative)
- Research taxonomy generation
- Sub-question prioritization by importance

**Source Credibility Scoring:**
- Domain authority assessment
- Publication recency and relevance
- Author expertise verification
- Citation count and impact analysis
- Cross-reference validation

**Completion Criteria:**
- Information saturation detection
- Confidence threshold achievement (≥85% coverage)
- Maximum time/resource limits
- All major perspectives captured

## Installation

### Requirements
- Python 3.11+
- MCP 2025-06-18 SDK
- Redis (for caching)
- External API keys (optional, for enhanced search)

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/RouHim/topic-deep-diver.git
cd topic-deep-diver
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure the server:**
```bash
cp config/config.example.yaml config/config.yaml
# Edit configuration with your API keys and preferences
```

4. **Start the MCP server:**
```bash
python -m topic_deep_diver
```

### MCP Client Configuration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "topic-deep-diver": {
      "command": "python",
      "args": ["-m", "topic_deep_diver"],
      "env": {
        "CONFIG_PATH": "/path/to/config.yaml"
      }
    }
  }
}
```

## Development

### Project Structure
```
topic-deep-diver/
├── topic_deep_diver/          # Main package
│   ├── server/                # MCP server implementation
│   ├── research/              # Research pipeline components
│   ├── search/                # Search engine integrations
│   ├── analysis/              # Source analysis and scoring
│   ├── synthesis/             # Information synthesis
│   └── utils/                 # Utilities and helpers
├── tests/                     # Test suite
├── docs/                      # Documentation
├── config/                    # Configuration files
└── AGENTS.md                  # Project knowledge base
```

### Development Setup

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install development dependencies:**
```bash
pip install -r requirements-dev.txt
```

3. **Run tests:**
```bash
pytest tests/ -v
```

4. **Run with debugging:**
```bash
python -m topic_deep_diver --debug
```

## Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [x] MCP Server Setup with 2025-06-18 specification
- [ ] Core MCP Tools Implementation
- [ ] Basic Search Integration

### Phase 2: Intelligence Layer (Weeks 3-4)
- [ ] Query Processing Engine
- [ ] Source Analysis Engine
- [ ] Information Synthesis Engine

### Phase 3: Advanced Features (Weeks 5-6)
- [ ] Academic Search Integration
- [ ] Real-time Processing
- [ ] Quality Assurance Systems

### Phase 4: Optimization & Security (Weeks 7-8)
- [ ] Security Implementation (OAuth, RFC 8707)
- [ ] Performance Optimization
- [ ] Testing & Documentation

## Performance Targets

- **Research Completion Time**: <4 minutes for comprehensive topics
- **Source Diversity**: Minimum 10 unique, credible sources per report
- **Accuracy Rate**: >90% fact verification across sources
- **Information Coverage**: >85% coverage of major topic aspects

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Process
1. Check the [Issues](https://github.com/RouHim/topic-deep-diver/issues) for current tasks
2. Read [AGENTS.md](AGENTS.md) for project knowledge and context
3. Follow the established architecture and coding standards
4. Ensure all tests pass before submitting PRs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/) for the foundational specification
- [Perplexity AI](https://www.perplexity.ai/) for inspiration on deep research capabilities
- The open-source MCP community for tools and integrations

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issues](https://github.com/RouHim/topic-deep-diver/issues)
- 💬 [Discussions](https://github.com/RouHim/topic-deep-diver/discussions)
- 📧 Contact: [Create an issue](https://github.com/RouHim/topic-deep-diver/issues/new)

---

**Status**: 🚧 In Development | **Version**: 0.1.0-alpha | **MCP Spec**: 2025-06-18