"""
Smart Coaster integration for Desk Buddy.

Wraps the hardware SmartCoaster (HX711 load cell on Jetson GPIO) and provides
the same interface as HydrationTracker so the rest of the pipeline (agent, LCD,
events) works seamlessly.

When the hardware is unavailable (no Jetson GPIO), falls back to the simulated
HydrationTracker.

The load cell is polled in a background thread at a configurable interval.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .events import EventBus

logger = logging.getLogger(__name__)

# Re-use constants from the coaster module
SIP_THRESHOLD_GRAMS = 10
EMPTY_CUP_GRAMS = 50
EMPTY_CUP_TOLERANCE = 10
NUM_SAMPLES = 1000
DEFAULT_POLL_INTERVAL = 10.0  # seconds between load cell reads


class SmartCoasterTracker:
    """
    Hardware-backed hydration tracker using the HX711 load cell.

    Provides the same public interface as HydrationTracker:
        - add_intake(ml)
        - set_goal(ml)
        - reset_daily()
        - get_hydration_status() -> dict

    Additionally polls the load cell in a background thread and auto-detects
    sips, emitting SIP_DETECTED / HYDRATION_GOAL_REACHED events.
    """

    def __init__(
        self,
        goal_ml: float = 2000,
        event_bus: Optional['EventBus'] = None,
        coaster_dir: str = "coaster",
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        profile_name: Optional[str] = None,
        demo_mode: bool = False,
    ):
        self.goal_ml = goal_ml
        self.intake_ml: float = 0.0
        self.last_sip_time: Optional[float] = None
        self.event_bus = event_bus
        self._goal_reached_notified = False

        self._coaster_dir = Path(coaster_dir)
        self._poll_interval = poll_interval
        self._profile_name = profile_name

        # Hardware state (set in setup())
        self._load_cell = None
        self._calibration = None
        self._cup_tare: float = EMPTY_CUP_GRAMS
        self._cup_name: Optional[str] = None
        self._last_weight: float = 0.0

        # Background polling
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Reminder timer
        self._last_reminder_time = time.time()
        self._reminder_interval = 60 if demo_mode else 1800  # 1 min demo, 30 min normal

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> bool:
        """
        Initialize load cell hardware and load calibration.

        Returns True if hardware is ready, False if unavailable.
        """
        try:
            import Jetson.GPIO as GPIO
            # Add coaster dir to path so we can import hx711
            import sys
            coaster_path = str(self._coaster_dir.resolve())
            if coaster_path not in sys.path:
                sys.path.insert(0, coaster_path)
            from hx711 import HX711

            GPIO.setwarnings(False)

            # Blinka (LCD) sets GPIO.TEGRA_SOC at import time via t234/pin.py.
            # Jetson.GPIO only allows one setmode() per process.
            # Detect: if a mode is already active, use TEGRA_SOC pin names;
            # if standalone (no LCD), use BOARD mode like coaster/coaster.py.
            current_mode = GPIO.getmode()
            if current_mode is None:
                GPIO.setmode(GPIO.BOARD)
                DT_PIN = 29
                SCK_PIN = 31
            else:
                # BOARD 29 → GP18_CAN0_DIN, BOARD 31 → GP17_CAN0_DOUT (AGX Orin)
                DT_PIN = "GP18_CAN0_DIN"
                SCK_PIN = "GP17_CAN0_DOUT"
                logger.info(f"GPIO mode {current_mode} already active, using TEGRA_SOC pins")

            self._load_cell = HX711(SCK_PIN, DT_PIN, 128)
            logger.info("HX711 load cell initialized")

        except ImportError:
            logger.info("Jetson.GPIO / HX711 not available - coaster hardware not present")
            return False
        except Exception as e:
            logger.warning(f"Failed to init load cell: {e}")
            return False

        # Load calibration
        self._calibration = self._load_calibration()
        if self._calibration is None:
            logger.warning("No coaster calibration found. Run coaster calibration first.")
            return False

        # Load profile cup info
        self._load_profile_cup()

        # Take initial weight reading
        self._last_weight = self._read_grams()
        logger.info(f"Coaster ready. Initial weight: {self._last_weight:.1f}g, "
                    f"cup tare: {self._cup_tare:.1f}g")

        return True

    def calibrate(self) -> bool:
        """
        Run non-interactive coaster calibration.

        For the pipeline, this performs a zero-point calibration:
        assumes nothing is on the coaster and records the zero reading.
        A full calibration with reference weight should be done via the
        standalone coaster.py script.
        """
        if self._load_cell is None:
            logger.error("Load cell not initialized")
            return False

        logger.info("Coaster calibration: reading zero point (ensure coaster is empty)...")
        scale_zero = self._load_cell.read_average(NUM_SAMPLES)
        logger.info(f"Zero point recorded: {scale_zero}")

        # If we have an existing calibration, update just the zero value
        if self._calibration:
            self._calibration["raw_zero_value"] = scale_zero
        else:
            # Minimal calibration - user should do full calibration separately
            logger.warning("No existing calibration. Using default scale factor.")
            self._calibration = {
                "raw_zero_value": scale_zero,
                "raw_ref_value": 0,
                "ref_weight_grams": 0,
                "scale_factor": 1.0,
            }

        self._save_calibration()
        return True

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background load cell polling."""
        if self._load_cell is None or self._calibration is None:
            logger.warning("Cannot start polling - hardware not set up")
            return

        self._running = True
        self._last_reminder_time = time.time()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info(f"Coaster polling started (interval: {self._poll_interval}s)")

    def stop(self) -> None:
        """Stop background polling."""
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
            self._poll_thread = None

    def _poll_loop(self) -> None:
        """Background thread: read load cell and detect sips."""
        while self._running:
            try:
                weight = self._read_grams()

                with self._lock:
                    sip_ml = self._process_sip(weight)

                if sip_ml is not None and sip_ml > 0:
                    logger.info(f"Sip detected: ~{sip_ml:.0f} mL")
                    with self._lock:
                        self.intake_ml += sip_ml
                        self.last_sip_time = time.time()
                    self._emit_sip_event()

                # Check reminder
                if time.time() - self._last_reminder_time >= self._reminder_interval:
                    self._last_reminder_time = time.time()
                    self._emit_reminder_event()

            except Exception as e:
                logger.error(f"Coaster poll error: {e}")

            time.sleep(self._poll_interval)

    def _process_sip(self, weight: float) -> Optional[float]:
        """Detect a sip from weight change. Returns mL consumed or None."""
        pickup_threshold = self._cup_tare if self._cup_tare > 0 else EMPTY_CUP_GRAMS
        if weight <= (pickup_threshold - EMPTY_CUP_TOLERANCE):
            # Cup picked up - don't count as sip
            return None

        prev_water = max(0.0, self._last_weight - self._cup_tare)
        curr_water = max(0.0, weight - self._cup_tare)
        sip_amount = prev_water - curr_water
        self._last_weight = weight

        if sip_amount >= SIP_THRESHOLD_GRAMS:
            self._last_reminder_time = time.time()
            return sip_amount
        return None

    # ------------------------------------------------------------------
    # HydrationTracker-compatible interface
    # ------------------------------------------------------------------

    def add_intake(self, ml: float) -> None:
        """Log water intake manually (same as HydrationTracker)."""
        with self._lock:
            self.intake_ml += ml
            self.last_sip_time = time.time()
        logger.info(f"Water intake (manual): +{ml} mL "
                    f"(total: {self.intake_ml:.0f}/{self.goal_ml:.0f} mL)")
        self._emit_sip_event()

    def set_goal(self, ml: float) -> None:
        """Set daily water goal in mL."""
        with self._lock:
            self.goal_ml = ml
            self._goal_reached_notified = self.intake_ml >= self.goal_ml
        logger.info(f"Water goal set to {ml} mL")

    def reset_daily(self) -> None:
        """Reset intake for a new day."""
        with self._lock:
            self.intake_ml = 0.0
            self.last_sip_time = None
            self._goal_reached_notified = False

    def get_hydration_status(self) -> Dict[str, Any]:
        """Get current hydration status (same format as HydrationTracker)."""
        with self._lock:
            return {
                "intake_ml": self.intake_ml,
                "goal_ml": self.goal_ml,
                "percent": (self.intake_ml / self.goal_ml * 100) if self.goal_ml > 0 else 0,
                "last_sip_time": self.last_sip_time,
                "cup_name": self._cup_name,
            }

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _emit_sip_event(self) -> None:
        """Emit sip detected event and check goal."""
        if not self.event_bus:
            return

        from .events import Event, EventType
        self.event_bus.emit(Event(
            type=EventType.SIP_DETECTED,
            data=self.get_hydration_status(),
        ))

        with self._lock:
            if self.intake_ml >= self.goal_ml and not self._goal_reached_notified:
                self._goal_reached_notified = True
                self.event_bus.emit(Event(
                    type=EventType.HYDRATION_GOAL_REACHED,
                    data=self.get_hydration_status(),
                ))

    def _emit_reminder_event(self) -> None:
        """Emit hydration reminder event."""
        if not self.event_bus:
            return

        from .events import Event, EventType
        self.event_bus.emit(Event(
            type=EventType.HYDRATION_REMINDER,
            data=self.get_hydration_status(),
        ))

    # ------------------------------------------------------------------
    # Hardware helpers
    # ------------------------------------------------------------------

    def _read_grams(self) -> float:
        """Read weight from load cell in grams."""
        if self._load_cell is None or self._calibration is None:
            return 0.0
        raw = self._load_cell.read_average(NUM_SAMPLES)
        zero = self._calibration["raw_zero_value"]
        scale = self._calibration["scale_factor"]
        if scale == 0:
            return 0.0
        return (raw - zero) / scale

    # ------------------------------------------------------------------
    # Profile & cup management
    # ------------------------------------------------------------------

    def list_profiles(self) -> List[str]:
        """Return names of all saved profiles."""
        profiles = self._load_all_profiles()
        return list(profiles.keys())

    def get_active_profile(self) -> Optional[str]:
        """Return the name of the currently active profile."""
        return self._profile_name

    def switch_profile(self, name: str) -> bool:
        """
        Switch to an existing profile by name.

        Loads the profile's cup, goal, and reminder settings.
        Resets daily intake. Returns True if profile found.
        """
        profiles = self._load_all_profiles()
        if name not in profiles:
            logger.warning(f"Profile '{name}' not found")
            return False

        self._profile_name = name
        profile = profiles[name]

        cup = profile.get("recent_cup")
        if cup:
            self._cup_tare = cup.get("cup_weight_grams", EMPTY_CUP_GRAMS)
            self._cup_name = cup.get("name")
        else:
            self._cup_tare = EMPTY_CUP_GRAMS
            self._cup_name = None

        if profile.get("daily_goal_ml"):
            self.goal_ml = profile["daily_goal_ml"]
        if profile.get("reminder_frequency"):
            self._reminder_interval = profile["reminder_frequency"]

        self.reset_daily()

        # Re-read initial weight so sip detection starts fresh
        if self._load_cell and self._calibration:
            self._last_weight = self._read_grams()

        logger.info(f"Switched to profile '{name}' "
                    f"(cup: {self._cup_name}, goal: {self.goal_ml} mL)")
        return True

    def create_profile(self, name: str, goal_ml: float = 2000) -> bool:
        """
        Create a new profile and switch to it.

        The cup can be registered later with register_cup().
        """
        profiles = self._load_all_profiles()
        if name in profiles:
            logger.info(f"Profile '{name}' already exists, switching to it")
            return self.switch_profile(name)

        profiles[name] = {
            "name": name,
            "daily_goal_ml": goal_ml,
            "reminder_frequency": 1800,
            "recent_cup": None,
            "last_sip_time": None,
        }
        self._save_all_profiles(profiles)
        return self.switch_profile(name)

    def register_cup(self, cup_name: str) -> Optional[float]:
        """
        Tare the load cell to register a new empty cup.

        Place the empty cup on the coaster before calling this.
        Returns the measured cup weight in grams, or None on failure.
        """
        if self._load_cell is None or self._calibration is None:
            logger.error("Cannot register cup - hardware not available")
            return None

        tare = self._read_grams()
        if tare <= 0:
            logger.warning(f"Cup tare reading suspiciously low ({tare:.1f}g)")

        self._cup_tare = tare
        self._cup_name = cup_name
        logger.info(f"Registered cup '{cup_name}' with tare {tare:.1f}g")

        # Save to active profile
        if self._profile_name:
            profiles = self._load_all_profiles()
            if self._profile_name in profiles:
                profiles[self._profile_name]["recent_cup"] = {
                    "name": cup_name,
                    "cup_weight_grams": tare,
                }
                self._save_all_profiles(profiles)

        return tare

    def get_current_cup(self) -> Optional[Dict[str, Any]]:
        """Return current cup info or None."""
        if self._cup_name is None:
            return None
        return {
            "name": self._cup_name,
            "tare_grams": self._cup_tare,
        }

    def _load_all_profiles(self) -> Dict[str, Any]:
        """Load all profiles from JSON file."""
        profiles_path = self._coaster_dir / "coaster_profiles.json"
        if not profiles_path.exists():
            return {}
        try:
            with open(profiles_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_all_profiles(self, profiles: Dict[str, Any]) -> None:
        """Save all profiles to JSON file."""
        profiles_path = self._coaster_dir / "coaster_profiles.json"
        try:
            with open(profiles_path, "w") as f:
                json.dump(profiles, f, indent=2)
            logger.info(f"Profiles saved to {profiles_path}")
        except Exception as e:
            logger.warning(f"Failed to save profiles: {e}")

    # ------------------------------------------------------------------
    # Hardware helpers
    # ------------------------------------------------------------------

    def _load_calibration(self) -> Optional[dict]:
        """Load calibration from JSON file."""
        cal_path = self._coaster_dir / "calibration.json"
        if not cal_path.exists():
            return None
        try:
            with open(cal_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load coaster calibration: {e}")
            return None

    def _save_calibration(self) -> None:
        """Save calibration to JSON file."""
        cal_path = self._coaster_dir / "calibration.json"
        try:
            with open(cal_path, "w") as f:
                json.dump(self._calibration, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save coaster calibration: {e}")

    def _load_profile_cup(self) -> None:
        """Load cup tare weight from profile if available."""
        profiles = self._load_all_profiles()
        if not profiles:
            return

        # Use specified profile or first available
        name = self._profile_name
        if name and name in profiles:
            profile = profiles[name]
        elif profiles:
            name = next(iter(profiles))
            profile = profiles[name]
            self._profile_name = name
        else:
            return

        cup = profile.get("recent_cup")
        if cup:
            self._cup_tare = cup.get("cup_weight_grams", EMPTY_CUP_GRAMS)
            self._cup_name = cup.get("name")
            logger.info(f"Loaded cup '{self._cup_name}' (tare: {self._cup_tare:.1f}g) "
                        f"from profile '{name}'")

        goal = profile.get("daily_goal_ml")
        if goal:
            self.goal_ml = goal

        freq = profile.get("reminder_frequency")
        if freq:
            self._reminder_interval = freq
