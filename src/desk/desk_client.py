"""
Async BLE desk control client.

Provides a high-level interface for controlling UPLIFT/Jiecang standing desks
via Bluetooth Low Energy. Supports:
- Moving to presets (sit/stand)
- Nudge movements (brief up/down as gentle reminder)
- Height queries
- Connection management

The actual BLE communication is handled by the sitstand repo's desk_control.py.
This module provides an async wrapper and higher-level commands.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Awaitable
import time

logger = logging.getLogger(__name__)


class DeskState(Enum):
    """Current desk state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    MOVING = "moving"
    ERROR = "error"


class DeskPosition(Enum):
    """Desk position presets."""
    SIT = "sit"
    STAND = "stand"
    UNKNOWN = "unknown"


@dataclass
class DeskStatus:
    """Current desk status."""
    state: DeskState
    position: DeskPosition
    height_cm: Optional[float] = None
    last_command: Optional[str] = None
    last_command_time: Optional[float] = None
    error_message: Optional[str] = None


class DeskClient:
    """
    Async BLE desk control client.

    Wraps the sitstand desk_control.py for async operation and provides
    high-level commands for the alert engine.

    Usage:
        desk = DeskClient()
        await desk.connect()

        # Move to preset
        await desk.stand()
        await desk.sit()

        # Gentle reminder nudge
        await desk.nudge_up(duration_ms=500)

        await desk.disconnect()
    """

    # Sitting height threshold in cm (below this = sitting)
    SIT_THRESHOLD_CM = 75.0
    # Standing height threshold in cm (above this = standing)
    STAND_THRESHOLD_CM = 100.0

    def __init__(
        self,
        sitstand_path: Optional[str] = None,
        auto_reconnect: bool = True,
        enabled: bool = True,
    ):
        """
        Initialize desk client.

        Args:
            sitstand_path: Path to sitstand repo (for importing desk_control)
            auto_reconnect: Whether to auto-reconnect on connection loss
            enabled: Whether desk control is enabled (False = no-op mode)
        """
        self.sitstand_path = sitstand_path
        self.auto_reconnect = auto_reconnect
        self.enabled = enabled

        self._state = DeskState.DISCONNECTED
        self._position = DeskPosition.UNKNOWN
        self._height_cm: Optional[float] = None
        self._last_command: Optional[str] = None
        self._last_command_time: Optional[float] = None
        self._error_message: Optional[str] = None

        # BLE client (lazily initialized)
        self._ble_client = None
        self._desk_control = None
        self._connected = False

        # Movement lock to prevent concurrent commands
        self._move_lock = asyncio.Lock()

        # Callbacks
        self._on_state_change: Optional[Callable[[DeskStatus], Awaitable[None]]] = None

    async def connect(self) -> bool:
        """
        Connect to the desk via BLE.

        Returns:
            True if connected successfully
        """
        if not self.enabled:
            logger.info("Desk control disabled, skipping connect")
            self._state = DeskState.CONNECTED  # Fake connected for no-op mode
            return True

        if self._connected:
            return True

        self._state = DeskState.CONNECTING
        await self._notify_state_change()

        try:
            # Try to import desk_control from sitstand repo
            if self._desk_control is None:
                self._desk_control = await self._load_desk_control()

            if self._desk_control is None:
                logger.warning("Could not load desk_control module, running in simulation mode")
                self._connected = True
                self._state = DeskState.CONNECTED
                await self._notify_state_change()
                return True

            # Attempt BLE connection
            # The actual connection is handled by desk_control's functions
            # which connect on-demand
            self._connected = True
            self._state = DeskState.CONNECTED
            await self._notify_state_change()
            logger.info("Desk client connected")
            return True

        except Exception as e:
            self._error_message = str(e)
            self._state = DeskState.ERROR
            await self._notify_state_change()
            logger.error(f"Failed to connect to desk: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the desk."""
        self._connected = False
        self._state = DeskState.DISCONNECTED
        await self._notify_state_change()
        logger.info("Desk client disconnected")

    async def _load_desk_control(self):
        """Load the desk_control module from sitstand repo."""
        import sys
        from pathlib import Path

        # Try multiple possible paths
        possible_paths = [
            self.sitstand_path,
            "../sitstand",
            "../../sitstand",
            Path.home() / "sitstand",
        ]

        for path in possible_paths:
            if path is None:
                continue
            path = Path(path)
            if (path / "desk_control.py").exists():
                sys.path.insert(0, str(path))
                try:
                    import desk_control
                    logger.info(f"Loaded desk_control from {path}")
                    return desk_control
                except ImportError as e:
                    logger.warning(f"Failed to import desk_control from {path}: {e}")
                    sys.path.remove(str(path))

        return None

    async def sit(self) -> bool:
        """
        Move desk to sitting position.

        Returns:
            True if command executed successfully
        """
        return await self._move_to_preset("sit")

    async def stand(self) -> bool:
        """
        Move desk to standing position.

        Returns:
            True if command executed successfully
        """
        return await self._move_to_preset("stand")

    async def _move_to_preset(self, preset: str) -> bool:
        """Move desk to a preset position."""
        if not self.enabled:
            logger.info(f"Desk control disabled, skipping {preset}")
            self._position = DeskPosition.SIT if preset == "sit" else DeskPosition.STAND
            return True

        async with self._move_lock:
            self._last_command = preset
            self._last_command_time = time.time()
            self._state = DeskState.MOVING
            await self._notify_state_change()

            try:
                if self._desk_control is not None:
                    # Run the blocking BLE command in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        self._desk_control.move_to_preset,
                        preset
                    )
                    success = result == 0
                else:
                    # Simulation mode
                    await asyncio.sleep(2.0)  # Simulate movement time
                    success = True

                if success:
                    self._position = DeskPosition.SIT if preset == "sit" else DeskPosition.STAND
                    logger.info(f"Desk moved to {preset} position")
                else:
                    logger.warning(f"Failed to move desk to {preset}")

                self._state = DeskState.CONNECTED
                await self._notify_state_change()
                return success

            except Exception as e:
                self._error_message = str(e)
                self._state = DeskState.ERROR
                await self._notify_state_change()
                logger.error(f"Error moving desk: {e}")
                return False

    async def nudge_up(self, duration_ms: int = 500) -> bool:
        """
        Brief upward movement as gentle reminder.

        Args:
            duration_ms: Duration of movement in milliseconds

        Returns:
            True if command executed successfully
        """
        return await self._nudge("up", duration_ms)

    async def nudge_down(self, duration_ms: int = 500) -> bool:
        """
        Brief downward movement.

        Args:
            duration_ms: Duration of movement in milliseconds

        Returns:
            True if command executed successfully
        """
        return await self._nudge("down", duration_ms)

    async def _nudge(self, direction: str, duration_ms: int) -> bool:
        """Execute a nudge movement."""
        if not self.enabled:
            logger.info(f"Desk control disabled, skipping nudge {direction}")
            return True

        async with self._move_lock:
            self._last_command = f"nudge_{direction}"
            self._last_command_time = time.time()
            self._state = DeskState.MOVING
            await self._notify_state_change()

            try:
                if self._desk_control is not None:
                    loop = asyncio.get_event_loop()

                    # Start movement
                    await loop.run_in_executor(
                        None,
                        self._desk_control.move_to_preset,
                        direction
                    )

                    # Wait for duration
                    await asyncio.sleep(duration_ms / 1000.0)

                    # Stop movement
                    await loop.run_in_executor(
                        None,
                        self._desk_control.move_to_preset,
                        "stop"
                    )
                else:
                    # Simulation mode
                    await asyncio.sleep(duration_ms / 1000.0)

                logger.info(f"Desk nudged {direction} for {duration_ms}ms")
                self._state = DeskState.CONNECTED
                await self._notify_state_change()
                return True

            except Exception as e:
                self._error_message = str(e)
                self._state = DeskState.ERROR
                await self._notify_state_change()
                logger.error(f"Error during nudge: {e}")
                return False

    async def stop(self) -> bool:
        """Stop any ongoing movement."""
        if not self.enabled or self._desk_control is None:
            return True

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._desk_control.move_to_preset,
                "stop"
            )
            self._state = DeskState.CONNECTED
            await self._notify_state_change()
            return True
        except Exception as e:
            logger.error(f"Error stopping desk: {e}")
            return False

    async def get_height(self) -> Optional[float]:
        """
        Get current desk height in cm.

        Returns:
            Height in cm, or None if not available
        """
        # Height reading requires desk_control support
        # For now, return cached value or estimate from position
        if self._height_cm is not None:
            return self._height_cm

        if self._position == DeskPosition.SIT:
            return self.SIT_THRESHOLD_CM - 5
        elif self._position == DeskPosition.STAND:
            return self.STAND_THRESHOLD_CM + 10
        return None

    def get_status(self) -> DeskStatus:
        """Get current desk status."""
        return DeskStatus(
            state=self._state,
            position=self._position,
            height_cm=self._height_cm,
            last_command=self._last_command,
            last_command_time=self._last_command_time,
            error_message=self._error_message,
        )

    def on_state_change(self, callback: Callable[[DeskStatus], Awaitable[None]]) -> None:
        """Register callback for state changes."""
        self._on_state_change = callback

    async def _notify_state_change(self) -> None:
        """Notify registered callback of state change."""
        if self._on_state_change:
            try:
                await self._on_state_change(self.get_status())
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")

    @property
    def is_connected(self) -> bool:
        return self._connected or not self.enabled

    @property
    def position(self) -> DeskPosition:
        return self._position

    @property
    def state(self) -> DeskState:
        return self._state
