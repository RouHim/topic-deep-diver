"""
NLP processing utilities for query analysis and concept extraction.
"""

from typing import Any

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from ..logging_config import get_logger
from .models import QuestionType

logger = get_logger(__name__)


class NLPProcessor:
    """Handles natural language processing tasks for query analysis."""

    def __init__(self) -> None:
        self.nlp: Any = None  # spaCy Language model
        self.lemmatizer = WordNetLemmatizer()
        self._ensure_nltk_data()

    def _ensure_nltk_data(self) -> None:
        """Ensure required NLTK data is downloaded."""
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("corpora/stopwords")
            nltk.data.find("corpora/wordnet")
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)

    def initialize_spacy(self) -> Any:
        """Initialize spaCy model if not already loaded."""
        if self.nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. Using NLTK-only mode."
                )
                self.nlp = None  # Will use NLTK fallbacks
        return self.nlp

    def extract_key_concepts(self, text: str) -> list[str]:
        """
        Extract key concepts from text using NLP techniques.

        Args:
            text: Input text to analyze

        Returns:
            List of key concepts/phrases
        """
        if not self.nlp:
            self.initialize_spacy()

        if self.nlp:
            # Use spaCy if available
            doc = self.nlp(text)

            # Extract noun phrases and named entities
            concepts = []

            # Get noun chunks (potential concepts)
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 4:  # Limit to reasonable phrase length
                    concepts.append(chunk.text.lower().strip())

            # Get named entities
            for ent in doc.ents:
                if ent.label_ in [
                    "PERSON",
                    "ORG",
                    "GPE",
                    "PRODUCT",
                    "EVENT",
                    "WORK_OF_ART",
                ]:
                    concepts.append(ent.text.lower().strip())
        else:
            # Fallback to NLTK-based extraction
            concepts = self._extract_concepts_nltk(text)

        # Remove duplicates and filter
        concepts = list(set(concepts))
        concepts = [c for c in concepts if len(c.split()) >= 1 and len(c) > 2]

        # Sort by relevance (simple heuristic: longer phrases first, then alphabetically)
        concepts.sort(key=lambda x: (-len(x.split()), x))

        return concepts[:20]  # Limit to top 20 concepts

    def _extract_concepts_nltk(self, text: str) -> list[str]:
        """
        Extract concepts using NLTK when spaCy is not available.

        Args:
            text: Input text

        Returns:
            List of potential concepts
        """
        # Simple noun phrase extraction using NLTK
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha() and len(token) > 2]

        # Get most frequent nouns (simple heuristic)
        from collections import Counter

        word_freq = Counter(tokens)

        # Return most common words as concepts
        concepts = [word for word, _ in word_freq.most_common(20)]
        return concepts

    def identify_question_type(self, question: str) -> QuestionType:
        """
        Identify the type of question based on linguistic patterns.

        Args:
            question: Question text to analyze

        Returns:
            QuestionType enum value
        """
        question_lower = question.lower()

        # Question type patterns
        if any(
            word in question_lower for word in ["what", "who", "when", "where", "which"]
        ):
            return QuestionType.FACTUAL
        elif any(word in question_lower for word in ["why", "how", "explain"]):
            return QuestionType.EXPLANATORY
        elif any(
            word in question_lower for word in ["compare", "difference", "versus", "vs"]
        ):
            return QuestionType.COMPARATIVE
        elif any(word in question_lower for word in ["analyze", "evaluate", "assess"]):
            return QuestionType.ANALYTICAL
        elif any(
            word in question_lower for word in ["will", "future", "predict", "trend"]
        ):
            return QuestionType.PREDICTIVE
        else:
            # Default to analytical for complex questions
            return QuestionType.ANALYTICAL

    def extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """
        Extract important keywords from text.

        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return

        Returns:
            List of keywords
        """
        # Tokenize and clean
        tokens = word_tokenize(text.lower())

        # Remove stopwords and non-alphabetic tokens
        stop_words = set(stopwords.words("english"))
        tokens = [
            token for token in tokens if token.isalpha() and token not in stop_words
        ]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Count frequency
        from collections import Counter

        word_freq = Counter(tokens)

        # Get most common words
        keywords = [word for word, _ in word_freq.most_common(max_keywords)]

        return keywords

    def calculate_complexity_score(self, text: str) -> float:
        """
        Calculate complexity score for a topic based on linguistic features.

        Args:
            text: Topic text to analyze

        Returns:
            Complexity score between 0.0 and 1.0
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        # Features for complexity calculation
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        unique_words = len(set(words))
        total_words = len(words)
        lexical_diversity = unique_words / total_words if total_words > 0 else 0

        # Complex words (words with 3+ syllables - simplified heuristic)
        complex_words = sum(1 for word in words if len(word) > 6)

        # Calculate score based on multiple factors
        sentence_factor = min(avg_sentence_length / 20.0, 1.0)  # Normalize to 0-1
        diversity_factor = lexical_diversity
        complexity_factor = complex_words / total_words if total_words > 0 else 0

        # Weighted average
        score = sentence_factor * 0.4 + diversity_factor * 0.4 + complexity_factor * 0.2

        return min(max(score, 0.0), 1.0)

    def split_into_sub_questions(self, topic: str) -> list[str]:
        """
        Split a complex topic into potential sub-questions.

        Args:
            topic: Research topic

        Returns:
            List of potential sub-questions
        """
        sentences = sent_tokenize(topic)

        # Look for question-like sentences
        questions = []
        for sentence in sentences:
            if "?" in sentence or any(
                sentence.lower().startswith(word)
                for word in ["what", "how", "why", "when", "where", "who"]
            ):
                questions.append(sentence.strip())

        # If no explicit questions, generate based on key concepts
        if not questions:
            concepts = self.extract_key_concepts(topic)
            for concept in concepts[:5]:  # Limit to top 5 concepts
                questions.append(f"What is {concept}?")
                questions.append(f"How does {concept} work?")

        return questions[:10]  # Limit to 10 sub-questions
