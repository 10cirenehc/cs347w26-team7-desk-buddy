"""
Lightweight pub/sub event bus for Desk Buddy.

Allows components (AlertEngine, FocusSessionManager, SmartCoaster)
to emit events that consumers (LCDController, main loop) can subscribe to.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Any

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events that can be emitted."""
    ALERT_TRIGGERED = "alert_triggered"
    FOCUS_STARTED = "focus_started"
    FOCUS_COMPLETED = "focus_completed"
    FOCUS_SUGGESTION = "focus_suggestion"
    BREAK_STARTED = "break_started"
    SIP_DETECTED = "sip_detected"
    HYDRATION_REMINDER = "hydration_reminder"
    HYDRATION_GOAL_REACHED = "hydration_goal_reached"


@dataclass
class Event:
    """An event emitted on the bus."""
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """
    Synchronous pub/sub event bus.

    Usage:
        bus = EventBus()
        bus.subscribe(EventType.ALERT_TRIGGERED, my_callback)
        bus.emit(Event(type=EventType.ALERT_TRIGGERED, data={"message": "Stand up!"}))
    """

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Register a callback for an event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Remove a callback for an event type."""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass

    def emit(self, event: Event) -> None:
        """Emit an event, calling all registered callbacks synchronously."""
        with self._lock:
            callbacks = list(self._subscribers.get(event.type, []))
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.type}: {e}")
