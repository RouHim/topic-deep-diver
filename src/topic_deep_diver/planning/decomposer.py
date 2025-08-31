from typing import List

import spacy

from .models import SearchStep, SubQuestion

class TopicDecomposer:
    """
    Decomposes a topic into a set of sub-questions using NLP.
    """

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # The model isn't installed. In a production setup, the model should be pre-installed.
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Please install it by running: python -m spacy download en_core_web_sm"
            )

    def decompose(self, topic: str) -> List[SubQuestion]:
        """
        Decomposes the topic into sub-questions.
        For now, this is a simple implementation.
        """
        doc = self.nlp(topic)
        
        # A simple approach: create a sub-question for each noun chunk
        sub_questions = []
        for chunk in doc.noun_chunks:
            sub_question = SubQuestion(
                question=f"What is the role of {chunk.text} in {topic}?",
                search_steps=[
                    SearchStep(query=f"{chunk.text} in {topic}"),
                    SearchStep(query=f"define {chunk.text}"),
                ]
            )
            sub_questions.append(sub_question)

        # Add a general overview question if no noun chunks were found
        if not sub_questions:
            sub_questions.append(
                SubQuestion(
                    question=f"What is a general overview of {topic}?",
                    search_steps=[
                        SearchStep(query=f"{topic} overview"),
                        SearchStep(query=f"introduction to {topic}"),
                    ]
                )
            )
            
        return sub_questions
