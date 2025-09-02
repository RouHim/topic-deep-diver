"""
Research taxonomy generation and management.
"""


from ..logging_config import get_logger
from .models import QueryAnalysis, TaxonomyNode

logger = get_logger(__name__)


class TaxonomyGenerator:
    """Generates and manages research taxonomies for topics."""

    def __init__(self):
        # Pre-defined domain hierarchies
        self.domain_hierarchies = {
            "technology": {
                "children": ["software", "hardware", "ai", "internet", "mobile"],
                "software": ["programming", "databases", "web", "security", "cloud"],
                "ai": ["machine_learning", "nlp", "computer_vision", "robotics"],
                "hardware": ["computers", "networks", "embedded", "iot"]
            },
            "science": {
                "children": ["physics", "chemistry", "biology", "mathematics", "earth_science"],
                "biology": ["genetics", "ecology", "neuroscience", "microbiology"],
                "physics": ["quantum", "relativity", "thermodynamics", "electromagnetism"]
            },
            "health": {
                "children": ["medicine", "nutrition", "mental_health", "public_health"],
                "medicine": ["cardiology", "oncology", "neurology", "pediatrics"]
            },
            "business": {
                "children": ["finance", "marketing", "management", "entrepreneurship"],
                "finance": ["investing", "banking", "cryptocurrency", "economics"]
            },
            "social": {
                "children": ["politics", "education", "culture", "sociology"],
                "politics": ["government", "policy", "international_relations"]
            }
        }

    def generate_taxonomy(self, analysis: QueryAnalysis) -> list[TaxonomyNode]:
        """
        Generate a research taxonomy based on query analysis.

        Args:
            analysis: Query analysis results

        Returns:
            List of taxonomy nodes
        """
        nodes = []

        # Create nodes for key concepts
        for concept in analysis.key_concepts:
            node = TaxonomyNode(
                term=concept,
                relevance_score=self._calculate_concept_relevance(concept, analysis),
                sources=[]
            )
            nodes.append(node)

        # Add domain-specific nodes
        for domain in analysis.domains:
            if domain in self.domain_hierarchies:
                domain_nodes = self._expand_domain_hierarchy(domain)
                nodes.extend(domain_nodes)

        # Establish parent-child relationships
        self._build_relationships(nodes)

        # Sort by relevance
        nodes.sort(key=lambda x: x.relevance_score, reverse=True)

        return nodes

    def _calculate_concept_relevance(self, concept: str, analysis: QueryAnalysis) -> float:
        """
        Calculate relevance score for a concept.

        Args:
            concept: Concept to score
            analysis: Query analysis

        Returns:
            Relevance score (0.0 to 1.0)
        """
        # Simple relevance calculation based on frequency and position
        topic_lower = analysis.original_topic.lower()
        concept_lower = concept.lower()

        # Exact matches get higher score
        if concept_lower in topic_lower:
            base_score = 0.8
        else:
            base_score = 0.4

        # Adjust based on concept length (longer concepts are more specific)
        length_bonus = min(len(concept.split()) * 0.1, 0.3)

        # Adjust based on question type
        type_multiplier = {
            "factual": 1.0,
            "analytical": 1.1,
            "comparative": 1.2,
            "explanatory": 1.1,
            "predictive": 1.0
        }.get(analysis.question_type.value, 1.0)

        score = (base_score + length_bonus) * type_multiplier
        return min(score, 1.0)

    def _expand_domain_hierarchy(self, domain: str) -> list[TaxonomyNode]:
        """
        Expand a domain into its hierarchical components.

        Args:
            domain: Domain name

        Returns:
            List of taxonomy nodes for the domain
        """
        nodes = []
        hierarchy = self.domain_hierarchies.get(domain, {})

        # Add main domain node
        nodes.append(TaxonomyNode(
            term=domain,
            relevance_score=0.9,
            sources=[]
        ))

        # Add child domains
        for child in hierarchy.get("children", []):
            nodes.append(TaxonomyNode(
                term=child,
                parent=domain,
                relevance_score=0.7,
                sources=[]
            ))

            # Add grandchild domains if they exist
            if child in hierarchy:
                for grandchild in hierarchy[child]:
                    nodes.append(TaxonomyNode(
                        term=grandchild,
                        parent=child,
                        relevance_score=0.5,
                        sources=[]
                    ))

        return nodes

    def _build_relationships(self, nodes: list[TaxonomyNode]) -> None:
        """
        Build parent-child relationships between taxonomy nodes.

        Args:
            nodes: List of taxonomy nodes to process
        """
        node_dict = {node.term: node for node in nodes}

        for node in nodes:
            if node.parent and node.parent in node_dict:
                parent_node = node_dict[node.parent]
                if node.term not in parent_node.children:
                    parent_node.children.append(node.term)

    def identify_relevant_domains(self, topic: str) -> list[str]:
        """
        Identify relevant domains for a research topic.

        Args:
            topic: Research topic

        Returns:
            List of relevant domain names
        """
        topic_lower = topic.lower()
        relevant_domains = []

        domain_keywords = {
            "technology": ["software", "hardware", "computer", "digital", "tech", "ai", "machine learning", "internet"],
            "science": ["research", "scientific", "physics", "chemistry", "biology", "mathematics", "theory"],
            "health": ["medical", "health", "disease", "treatment", "medicine", "clinical", "patient"],
            "business": ["company", "market", "finance", "economy", "business", "corporate", "industry"],
            "social": ["society", "social", "political", "education", "culture", "community", "policy"]
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                relevant_domains.append(domain)

        return relevant_domains if relevant_domains else ["general"]

    def get_domain_specific_terms(self, domain: str) -> list[str]:
        """
        Get domain-specific terminology.

        Args:
            domain: Domain name

        Returns:
            List of domain-specific terms
        """
        hierarchy = self.domain_hierarchies.get(domain, {})
        terms = [domain]

        for child in hierarchy.get("children", []):
            terms.append(child)
            if child in hierarchy:
                terms.extend(hierarchy[child])

        return terms
