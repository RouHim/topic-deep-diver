from typing import List

from pydantic import BaseModel, Field


class SearchStep(BaseModel):
    """A single search query to be executed."""

    query: str = Field(description="The search query to execute.")
    engine: str = Field(default="searxng", description="The search engine to use.")
    source_type: str = Field(
        default="web", description="The type of source (e.g., web, academic, news)."
    )


class SubQuestion(BaseModel):
    """A sub-question to be researched as part of a larger topic."""

    question: str = Field(description="The sub-question to research.")
    search_steps: List[SearchStep] = Field(
        description="A list of search steps to answer the sub-question."
    )


class ResearchPlan(BaseModel):
    """A complete research plan for a given topic."""

    topic: str = Field(description="The original research topic.")
    scope: str = Field(
        description="The scope of the research (e.g., quick, comprehensive)."
    )
    sub_questions: List[SubQuestion] = Field(
        description="A list of sub-questions to research."
    )
