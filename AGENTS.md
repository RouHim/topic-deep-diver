# Topic Deep Diver - Project Agents & Knowledge Base

## Project Overview

**Topic Deep Diver** is a fully automated deep research MCP (Model Context Protocol) server that provides comprehensive topic analysis using multiple search engines and AI-powered synthesis. The system is designed to rival commercial solutions like Perplexity's Deep Research while maintaining full automation and extensive online search capabilities.

## Core Requirements

### Functional Requirements
- **Fully Automated**: Zero user intervention during research process
- **MCP 2025-06-18 Compliance**: Latest specification support with enhanced security
- **Comprehensive Search**: Multi-tier search across web, academic, and specialized databases
- **AI-Powered Synthesis**: Intelligent information aggregation and narrative generation
- **Structured Output**: JSON-formatted research reports with citations and metadata

### Technical Requirements
- **Protocol**: Model Context Protocol (MCP) 2025-06-18 specification
- **Language**: Python (recommended for AI/ML ecosystem)
- **Architecture**: Multi-agent orchestration with autonomous decision-making
- **Security**: OAuth Resource Server, Resource Indicators (RFC 8707)
- **Integration**: Leverage existing MCP tools (fetch, search engines)

## Research Architecture

### Core Pipeline
```
User Query → Research Planner → Multi-Search Engine → Content Processor → Knowledge Synthesizer → Structured Report
```

### Key Components

#### 1. Research Planner Agent
- Analyzes topic complexity and scope
- Generates comprehensive search strategy
- Creates research taxonomy and keywords
- Determines stopping criteria automatically

#### 2. Multi-Search Orchestrator
- Parallel execution across multiple search engines
- Dynamic search refinement based on results
- Source diversity optimization
- Real-time result quality assessment

#### 3. Content Analysis Engine
- Automatic source credibility scoring
- Content deduplication and clustering
- Bias detection and perspective analysis
- Information freshness validation

#### 4. Knowledge Synthesizer
- Cross-source fact verification
- Narrative structure generation
- Citation tracking and management
- Gap identification and resolution

## MCP 2025-06-18 Features Integration

### New Features to Implement
- **Structured Tool Output**: Return JSON-structured research reports with `structuredContent`
- **OAuth Resource Server**: Implement proper authentication for secure API access
- **Resource Links**: Return URIs to research artifacts instead of inlining everything
- **Enhanced Security**: Follow new security best practices and Resource Indicators (RFC 8707)
- **Protocol Version Headers**: Include `MCP-Protocol-Version` header in all HTTP requests

### Removed Features (Not Needed)
- ~~Elicitation support~~ (Not needed for fully automated system)
- ~~Interactive user feedback loops~~ (Fully autonomous operation)
- ~~JSON-RPC batching~~ (Removed from spec)

## Search Integration Strategy

### Multi-Tier Search Architecture

#### 1. Primary Search Engines (General Web)
- SearXNG instances (privacy-focused, multiple engines)
- Brave Search API (privacy-respecting, fresh results)
- Google Search API (comprehensive coverage)
- Bing Search API (Microsoft ecosystem)

#### 2. Academic Search Engines
- Google Scholar API (200M+ scholarly articles)
- Semantic Scholar (AI-enhanced academic search)
- PubMed API (30M+ medical citations)
- Crossref API (metadata for scholarly content)
- CORE API (open access academic papers)

#### 3. Specialized Data Sources
- arXiv API (preprints in STEM fields)
- SSRN (social sciences research)
- RePEc (economics research)
- Patent databases (USPTO, EPO)
- News APIs (Reuters, AP, specialized outlets)

#### 4. Content Extraction Tools
- Fetch MCP server (web content extraction)
- Firecrawl (JavaScript-heavy sites)
- Jina Reader (clean text extraction)
- PDF extraction tools

## Autonomous Decision-Making Algorithms

### 1. Query Decomposition Algorithm
```
INPUT: Complex research topic
PROCESS:
  - Extract key concepts using NLP
  - Identify question types (factual, analytical, comparative)
  - Generate search taxonomy
  - Prioritize sub-questions by importance
OUTPUT: Structured research plan with ranked queries
```

### 2. Search Strategy Selection
```
For each sub-question:
  - Analyze question type and domain
  - Select appropriate search engines
  - Determine search parameters
  - Estimate information need
```

### 3. Source Credibility Scoring
```
FACTORS:
  - Domain authority (academic institutions, government, established media)
  - Publication recency and relevance
  - Citation count and impact factor
  - Author expertise and credentials
  - Cross-reference validation
SCORING: 0-100 credibility score
```

### 4. Information Synthesis Logic
```
PROCESS:
  - Cluster similar information from multiple sources
  - Identify consensus vs. conflicting viewpoints
  - Weight information by source credibility
  - Generate coherent narrative structure
  - Track citation chains and evidence strength
```

### 5. Completion Criteria
```
STOP RESEARCH WHEN:
  - Information saturation reached (diminishing returns)
  - Confidence threshold met (≥85% coverage)
  - Maximum time/resource limits hit
  - All major perspectives captured
```

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. **MCP Server Setup**
   - Initialize Python project with MCP 2025-06-18 SDK
   - Implement OAuth Resource Server classification
   - Set up structured tool output capabilities
   - Create resource link support

2. **Core Tools Implementation**
   ```python
   # MCP Tools to expose
   @mcp_tool
   def deep_research(topic: str, scope: str = "comprehensive") -> StructuredContent
   
   @mcp_tool  
   def research_status(session_id: str) -> ResearchProgress
   
   @mcp_tool
   def export_research(session_id: str, format: str = "markdown") -> ResourceLink
   ```

3. **Basic Search Integration**
   - Integrate with SearXNG for web search
   - Connect to existing fetch MCP server
   - Implement basic content extraction

### Phase 2: Intelligence Layer (Weeks 3-4)
1. **Query Processing Engine**
   - Topic decomposition algorithms
   - Search strategy generation
   - Priority-based query planning

2. **Source Analysis**
   - Credibility scoring system
   - Bias detection algorithms
   - Content deduplication

3. **Information Synthesis**
   - Multi-source aggregation
   - Narrative generation
   - Citation tracking

### Phase 3: Advanced Features (Weeks 5-6)
1. **Academic Integration**
   - Google Scholar API integration
   - PubMed and academic database connections
   - Citation network analysis

2. **Real-time Processing**
   - Parallel search execution
   - Progressive result streaming
   - Dynamic strategy adjustment

3. **Quality Assurance**
   - Fact-checking protocols
   - Source verification
   - Information completeness validation

### Phase 4: Optimization & Security (Weeks 7-8)
1. **Security Implementation**
   - Resource Indicators (RFC 8707) compliance
   - Protected resource metadata
   - Secure token handling

2. **Performance Optimization**
   - Caching strategies
   - Rate limiting
   - Error handling and recovery

3. **Testing & Documentation**
   - Comprehensive test coverage
   - API documentation
   - Usage examples

## MCP Server Architecture

### Core Tools
1. `deep_research(topic, scope)` - Main research orchestrator
2. `research_status(session_id)` - Progress monitoring  
3. `export_research(session_id, format)` - Results export

### Example Implementation Structure
```python
# MCP 2025-06-18 compliance
class DeepResearchServer:
    def __init__(self):
        self.mcp_version = "2025-06-18"
        self.oauth_metadata = {
            "authorization_endpoint": "https://auth.topicdeepDiver.com/oauth2/authorize",
            "resource_indicators_required": True
        }
    
    async def deep_research(self, topic: str) -> StructuredContent:
        # Return structured JSON with research findings
        return StructuredContent(
            type="research_report",
            data={
                "executive_summary": "...",
                "key_findings": [...],
                "sources": [...],
                "methodology": "...",
                "confidence_score": 0.87
            },
            resource_links=[
                ResourceLink(uri="https://research-cache.com/session123/full-report.pdf"),
                ResourceLink(uri="https://research-cache.com/session123/sources.json")
            ]
        )
```

## Research Insights from Analysis

### Key Findings from MCP Research
- MCP 2025-06-18 specification includes major security enhancements
- Structured tool output and resource links are critical new features
- OAuth Resource Server classification requires proper implementation
- Elicitation feature removed from scope (user interaction not needed)

### Deep Research System Analysis
- Perplexity's Test Time Compute (TTC) framework is the gold standard
- Multi-step iterative reasoning is essential for quality
- Source credibility and bias detection are critical differentiators
- Academic integration provides significant value for research quality

### Search Engine Integration Research
- 15+ search engines and academic databases identified
- Multi-tier approach (web, academic, specialized) provides comprehensive coverage
- Content extraction tools are essential for processing diverse sources
- API rate limiting and error handling are critical for reliability

## Success Metrics

### Technical Metrics
- Research completion time: Target <4 minutes for comprehensive topics
- Source diversity: Minimum 10 unique, credible sources per report
- Accuracy rate: >90% fact verification across sources
- Citation coverage: Complete source attribution and linking

### Quality Metrics
- Information completeness: >85% coverage of major topic aspects
- Source credibility: Average credibility score >80/100
- Synthesis quality: Coherent narrative with balanced perspectives
- User satisfaction: Comprehensive, actionable research reports

## Next Steps

1. **Repository Setup**: Create project structure and development environment
2. **Issue Creation**: Break down roadmap into trackable GitHub issues
3. **MVP Development**: Start with Phase 1 foundation implementation
4. **Continuous Integration**: Set up testing and deployment pipelines
5. **Documentation**: Maintain comprehensive API and usage documentation

## References

### MCP Specification
- [MCP 2025-06-18 Specification](https://modelcontextprotocol.io/specification/2025-06-18/)
- [MCP Security Best Practices](https://auth0.com/blog/mcp-specs-update-all-about-auth/)
- [Microsoft MCP SDK Updates](https://devblogs.microsoft.com/dotnet/mcp-csharp-sdk-2025-06-18-update/)

### Deep Research Systems
- [Perplexity Deep Research Analysis](https://www.deepestresearch.com/2025/03/perplexity-deep-research-expert-guide.html)
- [Academic Research Agents](https://github.com/GiovaneIwamoto/deep-research)
- [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/)

### Search Engine Integration
- [Academic Search Engines Guide](https://paperguide.ai/blog/academic-search-engines/)
- [MCP Servers Collection](https://mcpservers.org/)
- [Web Crawling MCP Servers](https://medium.com/@gkrajoriya3/stop-writing-scrapers-5-best-web-crawling-mcp-servers-with-examples-28d42766c3ff)