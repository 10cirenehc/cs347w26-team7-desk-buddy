"""
Async BLE desk control client.

Provides a high-level interface for controlling UPLIFT/Jiecang standing desks
via Bluetooth Low Energy. Supports:
- Moving to presets (sit/stand)
- Nudge movements (brief up/down as gentle reminder)
- Persistent height monitoring (updates on button press or BLE command)
- Connection management with auto-reconnect

The actual BLE communication is handled by the sitstand repo's desk_control.py.
This module maintains a persistent BLE connection and subscribes to height
notifications so that physical button presses update the tracked height.
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

    Maintains a persistent BLE connection with height notification subscription.
    Physical button presses and BLE commands both trigger height updates.

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

    # BLE height parsing constants (from sitstand/desk_server.py)
    HEIGHT_SCALE_FACTOR = 100.7874
    HEIGHT_BASE_OFFSET_MM = 650.09

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

        # Persistent BLE connection for height monitoring and commands
        self._ble_client = None
        self._ble_config = None
        self._desk_control = None
        self._connected = False

        # Movement lock to prevent concurrent commands
        self._move_lock = asyncio.Lock()

        # Callbacks
        self._on_state_change: Optional[Callable[[DeskStatus], Awaitable[None]]] = None

    async def connect(self) -> bool:
        """
        Connect to the desk via BLE.

        Establishes a persistent BLE connection, subscribes to height
        notifications, and performs an initial height query.

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

            # Establish persistent BLE connection with height monitoring
            try:
                await self._start_height_monitor()
            except Exception as e:
                logger.warning(f"Persistent BLE connection failed, commands will use on-demand connections: {e}")

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
        # Close persistent BLE connection
        if self._ble_client is not None:
            try:
                if self._ble_client.is_connected:
                    await self._ble_client.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting BLE client: {e}")
            self._ble_client = None
            self._ble_config = None

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

    # ----- Persistent BLE height monitor -----

    async def _start_height_monitor(self) -> None:
        """
        Establish persistent BLE connection and subscribe to height notifications.

        The desk sends height notifications whenever it moves — whether from
        BLE commands or physical button presses. By keeping a connection open
        and subscribed, we get real-time height updates from any source.

        Also performs an initial height query by sending a brief 'up' then 'stop'.
        """
        try:
            from bleak import BleakClient
        except ImportError:
            logger.warning("bleak not installed, cannot establish persistent connection")
            return

        addr = await self._desk_control.get_desk_address()
        if not addr:
            logger.warning("Could not find desk address for height monitor")
            return

        self._ble_client = BleakClient(
            addr,
            timeout=20.0,
            disconnected_callback=self._on_ble_disconnect,
        )
        await self._ble_client.connect()

        if not self._ble_client.is_connected:
            logger.warning("BLE connection failed")
            self._ble_client = None
            return

        # Detect desk config
        self._ble_config = self._desk_control.get_cached_config()
        if not self._ble_config:
            self._ble_config = await self._desk_control.detect_desk_config(self._ble_client)
        if not self._ble_config:
            from bleak.uuids import normalize_uuid_16
            self._ble_config = self._desk_control.DESK_CONFIGS[normalize_uuid_16(0xFF00)]

        # Subscribe to height notifications
        await self._ble_client.start_notify(
            self._ble_config.output_char_uuid,
            self._handle_height_notification,
        )
        logger.info("Subscribed to desk height notifications")

        # Initial height query: brief 'up' to trigger a notification, then 'stop'
        try:
            await self._desk_control.send_command(self._ble_client, self._ble_config, "up")
            await asyncio.sleep(0.3)
            await self._desk_control.send_command(self._ble_client, self._ble_config, "stop")
        except Exception as e:
            logger.debug(f"Initial height query failed (non-fatal): {e}")

    def _handle_height_notification(self, sender, data) -> None:
        """
        Handle BLE height notification from the desk.

        Protocol: f2 f2 01 03 SS HH HH checksum 7e
        """
        if len(data) >= 8 and data[0] == 0xF2 and data[1] == 0xF2:
            if data[2] == 0x01:  # Height notification
                raw_value = int.from_bytes(data[5:7], byteorder='big')
                height_mm = (raw_value / self.HEIGHT_SCALE_FACTOR) + self.HEIGHT_BASE_OFFSET_MM
                self._height_cm = height_mm / 10.0
                self._update_position_from_height()

    def _on_ble_disconnect(self, client) -> None:
        """Handle unexpected BLE disconnection."""
        logger.warning("BLE desk connection lost")
        self._ble_client = None
        self._ble_config = None

        if self.auto_reconnect and self._connected:
            logger.info("Will attempt BLE reconnect on next command")

    async def _ensure_ble_connection(self) -> bool:
        """
        Ensure persistent BLE connection is active, reconnecting if needed.

        Returns:
            True if connection is available.
        """
        if self._ble_client is not None and self._ble_client.is_connected:
            return True

        if not self.auto_reconnect or self._desk_control is None:
            return False

        logger.info("Reconnecting persistent BLE connection...")
        try:
            await self._start_height_monitor()
            return self._ble_client is not None and self._ble_client.is_connected
        except Exception as e:
            logger.warning(f"BLE reconnect failed: {e}")
            return False

    async def _send_command_direct(self, command: str) -> bool:
        """
        Send command through the persistent BLE connection.

        Returns:
            True if command sent successfully.
        """
        if not await self._ensure_ble_connection():
            return False

        return await self._desk_control.send_command(
            self._ble_client, self._ble_config, command
        )

    # ----- Position helpers -----

    def _update_position_from_height(self) -> None:
        """Derive position (sit/stand/unknown) from current height."""
        if self._height_cm is None:
            return
        if self._height_cm < self.SIT_THRESHOLD_CM:
            self._position = DeskPosition.SIT
        elif self._height_cm > self.STAND_THRESHOLD_CM:
            self._position = DeskPosition.STAND
        else:
            self._position = DeskPosition.UNKNOWN

    # ----- Public commands -----

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
        """
        Move desk to a preset position (non-blocking).

        Launches a background task that sends directional commands ("up"/"down")
        every 200ms while monitoring height via BLE notifications. Stops once
        the target height is reached or timeout expires.
        """
        if not self.enabled:
            logger.info(f"Desk control disabled, skipping {preset}")
            self._position = DeskPosition.SIT if preset == "sit" else DeskPosition.STAND
            return True

        self._last_command = preset
        self._last_command_time = time.time()
        self._state = DeskState.MOVING
        await self._notify_state_change()

        if await self._ensure_ble_connection():
            asyncio.create_task(self._move_until_height(preset))
            return True
        elif self._desk_control is not None:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._desk_control.main, preset
            )
            self._state = DeskState.CONNECTED
            await self._notify_state_change()
            return result == 0
        else:
            # Simulation mode
            self._position = DeskPosition.SIT if preset == "sit" else DeskPosition.STAND
            self._state = DeskState.CONNECTED
            await self._notify_state_change()
            return True

    async def _move_until_height(self, preset: str, timeout_seconds: float = 15.0) -> None:
        """
        Background task: wake the desk once, then send directional keep-alive
        packets (up/down) every 200ms while monitoring height. The Jiecang
        protocol stops the motor if it doesn't receive a command within ~1s,
        so periodic resends are required. Raw GATT writes bypass the heavy
        wake-per-call overhead in send_command.
        """
        dc = self._desk_control
        config = self._ble_config
        client = self._ble_client
        input_uuid = config.input_char_uuid

        direction = "up" if preset == "stand" else "down"
        target_cm = self.STAND_THRESHOLD_CM if preset == "stand" else self.SIT_THRESHOLD_CM
        direction_packet = dc.COMMANDS[direction]
        stop_packet = dc.COMMANDS["stop"]

        try:
            async with self._move_lock:
                # Wake once
                await dc.send_wake_sequence(client, input_uuid)

                end_time = time.time() + timeout_seconds
                while time.time() < end_time:
                    if self._height_cm is not None:
                        if preset == "stand" and self._height_cm >= target_cm:
                            logger.info(f"Reached standing height: {self._height_cm:.1f}cm")
                            break
                        if preset == "sit" and self._height_cm <= target_cm:
                            logger.info(f"Reached sitting height: {self._height_cm:.1f}cm")
                            break

                    # Keep-alive: resend direction packet directly
                    await client.write_gatt_char(input_uuid, direction_packet, response=False)
                    await asyncio.sleep(0.2)

                await client.write_gatt_char(input_uuid, stop_packet, response=False)

            self._position = DeskPosition.SIT if preset == "sit" else DeskPosition.STAND
            self._state = DeskState.CONNECTED
            await self._notify_state_change()
            logger.info(f"Desk moved to {preset} position")

        except Exception as e:
            self._error_message = str(e)
            self._state = DeskState.ERROR
            await self._notify_state_change()
            logger.error(f"Error moving desk to {preset}: {e}")

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
                if await self._ensure_ble_connection():
                    # Persistent connection path
                    await self._send_command_direct(direction)
                    await asyncio.sleep(duration_ms / 1000.0)
                    await self._send_command_direct("stop")
                elif self._desk_control is not None:
                    # Fallback: on-demand connections
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, self._desk_control.main, direction
                    )
                    await asyncio.sleep(duration_ms / 1000.0)
                    await loop.run_in_executor(
                        None, self._desk_control.main, "stop"
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
            if await self._ensure_ble_connection():
                await self._send_command_direct("stop")
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self._desk_control.main, "stop"
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
