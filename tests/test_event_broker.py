# Program: karl-offline-lab - test_event_broker.py
# Programmer: Ethan-da-Tech-Wizard (written with Claude, an AI assistant)
# Last Modified: 2026-11-18
# Purpose: Tests that the event broker delivers every message when many threads publish at once.
"""Thread-safety test for app.engine.event_broker.EventBroker."""

import os
import sys
import threading

# Make the repository root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine.event_broker import EventBroker

def test_event_broker_thread_safety():
    """Scenario: 10 threads each publish 100 events to one topic. Expected: the subscriber receives all 1000."""
    broker = EventBroker.get_instance()
    
    received_messages = []
    received_lock = threading.Lock()
    
    def on_event(data):
        """Record each received payload under a lock."""
        with received_lock:
            received_messages.append(data)
            
    broker.subscribe("test:thread_safety", on_event)
    
    threads = []
    num_threads = 10
    messages_per_thread = 100
    
    def publisher(thread_id):
        """Publish a fixed number of numbered events from one thread."""
        for i in range(messages_per_thread):
            broker.publish("test:thread_safety", {"thread_id": thread_id, "index": i})
            
    for t_id in range(num_threads):
        t = threading.Thread(target=publisher, args=(t_id,))
        threads.append(t)
        # Start publishers concurrently to stress the broker
        t.start()
        
    for t in threads:
        t.join()
        
    # Total messages received should be exactly 1000
    assert len(received_messages) == num_threads * messages_per_thread
    
    # Clean up subscriber
    broker.unsubscribe("test:thread_safety", on_event)
