from .decomposer import TopicDecomposer
from .models import ResearchPlan


class QueryPlanner:
    """
    Generates a research plan for a given topic by decomposing it into sub-questions.
    """

    def __init__(self):
        self.decomposer = TopicDecomposer()

    def generate_plan(self, topic: str, scope: str) -> ResearchPlan:
        """
        Generates a research plan for a given topic and scope.
        """
        sub_questions = self.decomposer.decompose(topic)

        plan = ResearchPlan(
            topic=topic,
            scope=scope,
            sub_questions=sub_questions,
        )
        return plan
