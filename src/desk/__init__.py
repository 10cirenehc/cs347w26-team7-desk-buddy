"""
Desk control module for BLE standing desk integration.

Provides async wrapper for UPLIFT/Jiecang desk control via BLE.
"""

from .desk_client import DeskClient, DeskState, DeskPosition

__all__ = [
    "DeskClient",
    "DeskState",
    "DeskPosition",
]
