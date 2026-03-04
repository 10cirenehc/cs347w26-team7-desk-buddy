"""
Simulated hydration tracker for Desk Buddy.

Provides the same interface that the real SmartCoaster will use,
so LCD and voice intents work against a stable API now.
When the real coaster merges, swap this out for SmartCoaster.
"""

import logging
import time
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .events import EventBus

logger = logging.getLogger(__name__)


class HydrationTracker:
    """
    Tracks daily water intake with a stable interface.

    Currently simulated (manual logging via voice/LCD).
    Will be replaced by SmartCoaster with real load cell data.

    Usage:
        tracker = HydrationTracker(goal_ml=2000, event_bus=bus)
        tracker.add_intake(250)
        status = tracker.get_hydration_status()
    """

    def __init__(self, goal_ml: float = 2000, event_bus: Optional['EventBus'] = None):
        self.intake_ml: float = 0.0
        self.goal_ml: float = goal_ml
        self.last_sip_time: Optional[float] = None
        self.cup_name: Optional[str] = None
        self.event_bus = event_bus
        self._goal_reached_notified = False

    def add_intake(self, ml: float) -> None:
        """Log water intake in mL."""
        self.intake_ml += ml
        self.last_sip_time = time.time()
        logger.info(f"Water intake: +{ml} mL (total: {self.intake_ml:.0f}/{self.goal_ml:.0f} mL)")

        if self.event_bus:
            from .events import Event, EventType
            self.event_bus.emit(Event(
                type=EventType.SIP_DETECTED,
                data=self.get_hydration_status(),
            ))

            # Check if goal reached
            if self.intake_ml >= self.goal_ml and not self._goal_reached_notified:
                self._goal_reached_notified = True
                self.event_bus.emit(Event(
                    type=EventType.HYDRATION_GOAL_REACHED,
                    data=self.get_hydration_status(),
                ))

    def set_goal(self, ml: float) -> None:
        """Set daily water goal in mL."""
        self.goal_ml = ml
        self._goal_reached_notified = self.intake_ml >= self.goal_ml
        logger.info(f"Water goal set to {ml} mL")

    def reset_daily(self) -> None:
        """Reset intake for a new day."""
        self.intake_ml = 0.0
        self.last_sip_time = None
        self._goal_reached_notified = False

    def get_hydration_status(self) -> Dict[str, Any]:
        """
        Get current hydration status.

        Returns dict with same interface the real SmartCoaster will provide.
        """
        return {
            "intake_ml": self.intake_ml,
            "goal_ml": self.goal_ml,
            "percent": (self.intake_ml / self.goal_ml * 100) if self.goal_ml > 0 else 0,
            "last_sip_time": self.last_sip_time,
            "cup_name": self.cup_name,
        }
