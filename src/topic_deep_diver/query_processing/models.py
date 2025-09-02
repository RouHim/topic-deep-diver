"""
Data models for query processing components.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """Types of questions that can be identified."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    EXPLANATORY = "explanatory"
    PREDICTIVE = "predictive"


class ResearchScope(str, Enum):
    """Research scope options."""

    QUICK = "quick"
    COMPREHENSIVE = "comprehensive"
    ACADEMIC = "academic"
    CUSTOM = "custom"


class SearchEngine(str, Enum):
    """Available search engines."""

    SEARXNG = "searxng"
    GOOGLE_SCHOLAR = "google_scholar"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PUBMED = "pubmed"
    CROSSREF = "crossref"
    ARXIV = "arxiv"


class SubQuestion(BaseModel):
    """A decomposed sub-question from the main topic."""

    question: str = Field(..., description="The sub-question text")
    question_type: QuestionType = Field(..., description="Type of question")
    importance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Importance score (0-1)"
    )
    keywords: list[str] = Field(
        default_factory=list, description="Key terms for search"
    )
    domain: str | None = Field(None, description="Specific domain or field")
    dependencies: list[str] = Field(
        default_factory=list, description="IDs of dependent questions"
    )


class SearchStrategy(BaseModel):
    """Search strategy for a specific query."""

    query: str = Field(..., description="Search query")
    engines: list[SearchEngine] = Field(..., description="Search engines to use")
    scope: str = Field(..., description="Search scope (broad, focused, academic)")
    max_results: int = Field(..., ge=1, description="Maximum results to retrieve")
    time_limit: int = Field(..., ge=1, description="Time limit in seconds")
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Additional search filters"
    )


class QueryAnalysis(BaseModel):
    """Complete analysis of a research topic."""

    original_topic: str = Field(..., description="Original research topic")
    key_concepts: list[str] = Field(..., description="Extracted key concepts")
    question_type: QuestionType = Field(..., description="Overall question type")
    domains: list[str] = Field(..., description="Relevant domains/fields")
    complexity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Topic complexity (0-1)"
    )
    estimated_sources: int = Field(..., ge=1, description="Estimated sources needed")
    sub_questions: list[SubQuestion] = Field(
        ..., description="Decomposed sub-questions"
    )


class QueryPlan(BaseModel):
    """Complete research plan for a topic."""

    topic: str = Field(..., description="Research topic")
    scope: ResearchScope = Field(..., description="Research scope")
    analysis: QueryAnalysis = Field(..., description="Topic analysis")
    strategies: list[SearchStrategy] = Field(..., description="Search strategies")
    total_estimated_time: int = Field(
        ..., ge=1, description="Total estimated time in seconds"
    )
    total_estimated_sources: int = Field(
        ..., ge=1, description="Total estimated sources"
    )
    priority_order: list[str] = Field(
        ..., description="Execution order of sub-questions"
    )
    created_at: float = Field(..., description="Plan creation timestamp")


class ScopeConfig(BaseModel):
    """Configuration for different research scopes."""

    name: str = Field(..., description="Scope name")
    max_sources: int = Field(..., ge=1, description="Maximum sources to retrieve")
    time_limit: int = Field(..., ge=1, description="Time limit in seconds")
    engines: list[SearchEngine] = Field(..., description="Allowed search engines")
    depth_preference: str = Field(..., description="Breadth vs depth preference")


class TaxonomyNode(BaseModel):
    """Node in the research taxonomy."""

    term: str = Field(..., description="Taxonomy term")
    parent: str | None = Field(None, description="Parent term")
    children: list[str] = Field(default_factory=list, description="Child terms")
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Relevance to topic"
    )
    sources: list[str] = Field(default_factory=list, description="Related sources")
