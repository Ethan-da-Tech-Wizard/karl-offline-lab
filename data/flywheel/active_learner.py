# ==============================================================================
# Program: active_learner.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Active learning selector prioritizing high-value training samples.
#          Identifies low-confidence model responses and flags them for
#          synthetic refinement.
# ==============================================================================

import os
import json

# Stats live next to this file so they travel with the flywheel data folder.
STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats.json")

class ActiveLearner:
    """Tracks recent pass/fail results per topic and decides which topics are worth generating data for.

    History is a rolling window of the last 20 booleans per topic, persisted as JSON.
    """
    def __init__(self, stats_path: str = STATS_PATH):
        # Step 1: Validate input parameters, operational preconditions, and state
        """Load any previously saved history from ``stats_path``."""
        self.stats_path = stats_path
        self.topic_history = {} # topic -> list of bool
        self.load()

    def load(self):
        # Step 1: Validate input parameters, operational preconditions, and state
        """Read history from disk; a missing or corrupt file just means starting empty."""
        if os.path.exists(self.stats_path):
            try:
                # Execute guarded operations and establish recovery boundaries
                with open(self.stats_path, "r", encoding="utf-8") as f:
                    self.topic_history = json.load(f)
            except Exception:
                self.topic_history = {}

    def save(self):
        # Step 1: Validate input parameters, operational preconditions, and state
        """Write history atomically: dump to a .tmp file, then rename over the real file.

        Save errors are printed and swallowed so a disk problem never stops a run.
        """
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        try:
            # Execute guarded operations and establish recovery boundaries
            temp_path = self.stats_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.topic_history, f, indent=2)
            # Rename is atomic on the same filesystem, so readers never see a half-written file.
            os.rename(temp_path, self.stats_path)
        except Exception as e:
            print(f"Error saving ActiveLearner stats: {e}")

    def record_result(self, topic: str, passed: bool):
        # Step 1: Validate input parameters, operational preconditions, and state
        """Append one pass/fail outcome for ``topic``, keep only the latest 20, and save."""
        history = self.topic_history.setdefault(topic, [])
        history.append(passed)
        # Rolling window: old results age out so the stat tracks current ability.
        if len(history) > 20:
            history.pop(0)
        self.save()

    def should_generate(self, topic: str) -> bool:
        # Step 1: Validate input parameters, operational preconditions, and state
        """Return True if more training samples for ``topic`` are worthwhile.

        With fewer than 5 results there is no baseline, so always generate. After
        that, skip topics that are already mastered (>=95% pass) or hopeless (<=5%).
        """
        history = self.topic_history.get(topic, [])
        if len(history) < 5:
            return True # Need more samples to establish baseline
        accuracy = sum(1 for x in history if x) / len(history)
        
        # Zone of Proximal Development: keep generating for any topic whose accuracy is
        # strictly between 5% and 95%. Throttle at 95%+ (already mastered) or 5%- (too
        # hard for the current parameters); the cut-offs are deliberately wide.
        if accuracy >= 0.95 or accuracy <= 0.05:
            return False
        return True
