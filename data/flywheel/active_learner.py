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

STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats.json")

class ActiveLearner:
    def __init__(self, stats_path: str = STATS_PATH):
        # Step 1: Validate input parameters, operational preconditions, and state
        self.stats_path = stats_path
        self.topic_history = {} # topic -> list of bool
        self.load()

    def load(self):
        # Step 1: Validate input parameters, operational preconditions, and state
        if os.path.exists(self.stats_path):
            try:
                # Execute guarded operations and establish recovery boundaries
                with open(self.stats_path, "r", encoding="utf-8") as f:
                    self.topic_history = json.load(f)
            except Exception:
                self.topic_history = {}

    def save(self):
        # Step 1: Validate input parameters, operational preconditions, and state
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        try:
            # Execute guarded operations and establish recovery boundaries
            temp_path = self.stats_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.topic_history, f, indent=2)
            os.rename(temp_path, self.stats_path)
        except Exception as e:
            print(f"Error saving ActiveLearner stats: {e}")

    def record_result(self, topic: str, passed: bool):
        # Step 1: Validate input parameters, operational preconditions, and state
        history = self.topic_history.setdefault(topic, [])
        history.append(passed)
        if len(history) > 20:
            history.pop(0)
        self.save()

    def should_generate(self, topic: str) -> bool:
        # Step 1: Validate input parameters, operational preconditions, and state
        history = self.topic_history.get(topic, [])
        if len(history) < 5:
            return True # Need more samples to establish baseline
        accuracy = sum(1 for x in history if x) / len(history)
        
        # Zone of Proximal Development: Focus where accuracy is between 30% and 70%
        # Throttle generation on 95%+ (already mastered) or 5%- (too hard for current parameters)
        if accuracy >= 0.95 or accuracy <= 0.05:
            return False
        return True
