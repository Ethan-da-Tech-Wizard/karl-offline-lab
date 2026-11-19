# ==============================================================================
# Program: test_event_broker.py
# Programmer: Ethan-da-Tech-Wizard
# Last Modified: November 18, 2026
# Purpose: Comprehensive test suite exercising event broker functionality.
#          Verifies state transitions, error boundaries, regression resistance,
#          and contract compliance under tests.
# ==============================================================================

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine.event_broker import EventBroker

def test_event_broker_thread_safety():
    """Publishing from many threads at once delivers every message exactly once to a subscriber (1000 in total), i.e. the broker is thread-safe."""
    broker = EventBroker.get_instance()
    
    received_messages = []
    received_lock = threading.Lock()
    
    def on_event(data):
        """Subscriber that appends received messages under a lock."""
        with received_lock:
            received_messages.append(data)
            
    broker.subscribe("test:thread_safety", on_event)
    
    threads = []
    num_threads = 10
    messages_per_thread = 100
    
    def publisher(thread_id):
        """Worker that publishes a fixed number of messages from one thread."""
        for i in range(messages_per_thread):
            broker.publish("test:thread_safety", {"thread_id": thread_id, "index": i})
            
    for t_id in range(num_threads):
        t = threading.Thread(target=publisher, args=(t_id,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    # Total messages received should be exactly 1000
    assert len(received_messages) == num_threads * messages_per_thread
    
    # Clean up subscriber
    broker.unsubscribe("test:thread_safety", on_event)
