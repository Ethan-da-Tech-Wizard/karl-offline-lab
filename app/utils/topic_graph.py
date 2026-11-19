# ==============================================================================
# Program: topic_graph.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Small fixed topic tree (math / coding, five leaf topics) used to
#          balance which training problem the self-improvement flywheel
#          generates next. It is a frequency sampler, not a knowledge graph.
# ==============================================================================

"""
Topic-Aware Sampling Graph for the Karl self-improvement flywheel.

Tracks selection frequency across a hierarchical concept tree so the
problem generator always picks the least-represented leaf topic next,
driving even coverage across the training curriculum.
"""


class TopicNode:
    """One node in the topic tree, tracking how often it has been selected."""
    def __init__(self, name: str, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.frequency = 0


class DynamicTopicGraph:
    """Fixed two-level topic tree (math, coding) used to balance generated training problems.
    """
    def __init__(self):
        """Build the root, the two categories and five leaf topics."""
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
        under = min(self.leaves, key=lambda node: node.frequency)
        under.frequency += 1
        return under.name
