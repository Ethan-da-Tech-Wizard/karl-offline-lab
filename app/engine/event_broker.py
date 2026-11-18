# ==============================================================================
# Program: event_broker.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Pub/sub message broker decoupling backend subsystems from UI components.
#          Enables loose coupling and thread-safe notification dispatching
#          across workers and workspaces.
# ==============================================================================

import threading
from typing import Callable, Dict, Set

class EventBroker:
    """Thread-safe singleton pub/sub bus for in-process telemetry events."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        """Create the subscriber registry.

        Use get_instance(); direct construction after the singleton exists raises
        RuntimeError.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        if EventBroker._instance is not None:
            raise RuntimeError("EventBroker is a singleton, use get_instance()")
        self._subscribers: Dict[str, Set[Callable[[dict], None]]] = {}
        self._registry_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "EventBroker":
        """Return the process-wide EventBroker instance."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with cls._lock:
            if cls._instance is None:
                cls._instance = EventBroker()
            return cls._instance

    def subscribe(self, topic: str, callback: Callable[[dict], None]) -> None:
        """Register callback for topic payloads.

        The callback receives the published dict payload. Duplicate
        subscriptions are collapsed by set semantics.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._registry_lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = set()
            self._subscribers[topic].add(callback)

    def unsubscribe(self, topic: str, callback: Callable[[dict], None]) -> None:
        """Remove callback from a topic if it is currently subscribed."""
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._registry_lock:
            if topic in self._subscribers:
                self._subscribers[topic].discard(callback)
                if not self._subscribers[topic]:
                    del self._subscribers[topic]

    def publish(self, topic: str, data: dict) -> None:
        """Publish a payload to all current subscribers of topic.

        Subscriber exceptions are suppressed so one failing callback cannot
        break token streaming or telemetry publication.
        """
        # Step 1: Validate input parameters, operational preconditions, and state
        with self._registry_lock:
            subscribers = list(self._subscribers.get(topic, []))
        for callback in subscribers:
            # Iterate over items, evaluate loop invariant, and update progress
            try:
                # Execute guarded operations and establish recovery boundaries
                callback(data)
            except Exception:
                pass  # Suppress individual subscriber failures to protect broker loop
