"""
LCD display module for Desk Buddy.

Provides touchscreen interface for posture/focus status,
hydration tracking, timer control, and alert notifications.
"""

from .lcd_controller import LCDController, LCDState

__all__ = [
    "LCDController",
    "LCDState",
]
