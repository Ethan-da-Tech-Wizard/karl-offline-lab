# Program: karl-offline-lab - topic_graph.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Keeps a small tree of training topics and always hands back the least-used leaf topic so coverage stays even.

"""
Topic-Aware Sampling Graph for the Karl self-improvement flywheel.

Tracks selection frequency across a hierarchical concept tree so the
problem generator always picks the least-represented leaf topic next,
driving even coverage across the training curriculum.
"""


# ── Topic tree node ──────────────────────────────────────────────────────
class TopicNode:
    """A single topic in the tree, with a parent link, child list and a usage counter."""

    def __init__(self, name: str, parent=None):
        """Create a node with a name, optional parent, no children and a zero usage count."""
        self.name = name
        self.parent = parent
        self.children = []
        self.frequency = 0


# ── Topic graph ──────────────────────────────────────────────────────────
class DynamicTopicGraph:
    """Builds a fixed two-level topic tree (math and coding) and tracks leaf selection counts."""

    def __init__(self):
        """Build the topic tree: a root, two categories and five leaf topics."""
        # Root of the tree; the category nodes below hang off it.
        self.root = TopicNode("root")

        math_node   = TopicNode("math",   self.root)
        coding_node = TopicNode("coding", self.root)

        # Only leaf topics are ever selected for problem generation.
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
