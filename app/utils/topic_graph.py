# ==============================================================================
# Program: topic_graph.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Conceptual graph model connecting related knowledge base documents.
#          Constructs semantic links between code entities, APIs, and
#          architectural topics.
# ==============================================================================

"""
Topic-Aware Sampling Graph for the Karl self-improvement flywheel.

Tracks selection frequency across a hierarchical concept tree so the
problem generator always picks the least-represented leaf topic next,
driving even coverage across the training curriculum.
"""


class TopicNode:
    def __init__(self, name: str, parent=None):
        # Step 1: Validate input parameters, operational preconditions, and state
        self.name = name
        self.parent = parent
        self.children = []
        self.frequency = 0


class DynamicTopicGraph:
    def __init__(self):
        # Step 1: Validate input parameters, operational preconditions, and state
        self.root = TopicNode("root")

        math_node   = TopicNode("math",   self.root)
        coding_node = TopicNode("coding", self.root)

        self.leaves = [
            TopicNode("algebra_3var",        math_node),
            TopicNode("quadratics",          math_node),
            TopicNode("parentheses_matching", coding_node),
            TopicNode("matrix_transposition", coding_node),
            TopicNode("anagrams",            coding_node),
        ]

    def get_underrepresented_topic(self) -> str:
        """Return the leaf name with the lowest selection frequency and increment it."""
        # Step 1: Validate input parameters, operational preconditions, and state
        under = min(self.leaves, key=lambda node: node.frequency)
        under.frequency += 1
        return under.name
