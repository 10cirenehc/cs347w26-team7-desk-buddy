"""
LCD display controller with state machine and event handling.

Manages screen transitions, touch input, and event subscriptions.
"""

import logging
import time
from enum import Enum
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from PIL import Image

from .lcd_driver import LCDDriver, WIDTH, HEIGHT
from .lcd_drawing import touch_in, find_wallpapers, load_wallpaper
from . import lcd_screens as screens

if TYPE_CHECKING:
    from ..events import EventBus, Event

logger = logging.getLogger(__name__)


class LCDState(Enum):
    """LCD screen states."""
    HOME = "home"
    NOTIFICATION = "notification"
    TIMER_SETUP = "timer_setup"
    TIMER_RUN = "timer_run"
    TIMER_DONE = "timer_done"
    WALLPAPER = "wallpaper"
    WATER_GOAL_EDIT = "water_goal_edit"
    HYDRATION_DETAIL = "hydration_detail"


class LCDController:
    """
    LCD display controller with state machine and event handling.

    Subscribes to events from AlertEngine, FocusSessionManager, etc.
    Returns action dicts from tick() for the main loop to dispatch.

    Usage:
        lcd = LCDController(event_bus=bus, config=config)
        if lcd.setup():
            lcd.update(posture, focus, hydration_status, timer_status)
            action = lcd.tick()
            if action:
                handle_action(action)
    """

    def __init__(self, event_bus: Optional['EventBus'] = None, config: Optional[Dict] = None):
        config = config or {}
        self._event_bus = event_bus
        self._driver: Optional[LCDDriver] = None

        # Config
        self._render_rate = 1.0 / config.get('render_rate_hz', 5)
        self._touch_debounce = config.get('touch_debounce_ms', 250) / 1000.0
        self._wallpaper_dir = config.get('wallpaper_dir', 'data/wallpapers')

        # State machine
        self._state = LCDState.HOME
        self._hits: Dict = {}
        self._last_render = 0.0
        self._last_touch = 0.0

        # Notification queue
        self._notif_queue: List[Dict[str, str]] = []
        self._current_notif: Optional[Dict[str, str]] = None

        # Timer setup state
        self._timer_h = 0
        self._timer_m = 25
        self._timer_s = 0
        self._editing_field = 1

        # Wallpaper state
        self._wallpapers: List[str] = []
        self._wallpaper_idx = 0
        self._wallpaper_img: Optional[Image.Image] = None

        # Water goal edit state
        self._edit_goal_ml = 2000.0

        # Cached state from main loop
        self._posture_state = "unknown"
        self._focus_state = "unknown"
        self._hydration_status: Dict[str, Any] = {}
        self._timer_status: Optional[Dict[str, Any]] = None

        # Subscribe to events
        if self._event_bus:
            self._subscribe_events()

    def _subscribe_events(self) -> None:
        """Subscribe to relevant events."""
        from ..events import EventType

        self._event_bus.subscribe(EventType.ALERT_TRIGGERED, self._on_alert)
        self._event_bus.subscribe(EventType.HYDRATION_REMINDER, self._on_hydration_reminder)
        self._event_bus.subscribe(EventType.SIP_DETECTED, self._on_sip_detected)
        self._event_bus.subscribe(EventType.FOCUS_STARTED, self._on_focus_started)
        self._event_bus.subscribe(EventType.FOCUS_COMPLETED, self._on_focus_completed)
        self._event_bus.subscribe(EventType.HYDRATION_GOAL_REACHED, self._on_hydration_goal)

    def _on_alert(self, event: 'Event') -> None:
        """Queue alert as notification."""
        msg = event.data.get("message", "Alert")
        if msg:
            self._notif_queue.append({
                "msg": msg,
                "detail": event.data.get("rule_name", ""),
            })

    def _on_hydration_reminder(self, event: 'Event') -> None:
        self._notif_queue.append({
            "msg": "Time to drink water!",
            "detail": "Stay hydrated",
        })

    def _on_sip_detected(self, event: 'Event') -> None:
        # Update hydration status from event data
        self._hydration_status = event.data

    def _on_focus_started(self, event: 'Event') -> None:
        self._notif_queue.append({
            "msg": "Focus session started!",
            "detail": event.data.get("message", ""),
        })

    def _on_focus_completed(self, event: 'Event') -> None:
        self._notif_queue.append({
            "msg": "Focus session complete!",
            "detail": event.data.get("message", ""),
        })

    def _on_hydration_goal(self, event: 'Event') -> None:
        self._notif_queue.append({
            "msg": "Water goal reached!",
            "detail": "Great job staying hydrated!",
        })

    def setup(self) -> bool:
        """
        Initialize LCD hardware.

        Returns:
            True if LCD is available and initialized.
        """
        if not LCDDriver.available():
            logger.info("LCD hardware not available, skipping LCD setup")
            return False

        try:
            self._driver = LCDDriver()
        except Exception as e:
            logger.warning(f"LCD initialization failed: {e}")
            return False

        # Load wallpapers
        self._wallpapers = find_wallpapers(self._wallpaper_dir)
        if self._wallpapers:
            self._wallpaper_img = load_wallpaper(self._wallpapers[0])
        logger.info(f"LCD ready, {len(self._wallpapers)} wallpaper(s) found")

        # Initial render
        self._render()
        return True

    def update(self, posture, focus, hydration_status: Dict, timer_status: Optional[Dict]) -> None:
        """
        Push latest state from the main loop.

        Args:
            posture: PostureState object or None
            focus: FocusState object or None
            hydration_status: dict from HydrationTracker.get_hydration_status()
            timer_status: dict from FocusSessionManager.get_status() or None
        """
        self._posture_state = posture.state.value if posture else "unknown"
        self._focus_state = focus.state.value if focus else "unknown"
        self._hydration_status = hydration_status
        self._timer_status = timer_status

    def tick(self) -> Optional[Dict[str, Any]]:
        """
        Poll touch and redraw at configured rate.

        Returns:
            Action dict if a touch triggered an action, None otherwise.
            Example: {"action": "start_focus", "duration_min": 25}
        """
        if not self._driver:
            return None

        now = time.monotonic()
        action = None

        # Handle touch
        touch = self._driver.get_touch()
        if touch and (now - self._last_touch) > self._touch_debounce:
            self._last_touch = now
            action = self._handle_touch(touch[0], touch[1])

        # Check for pending notifications
        if (self._state == LCDState.HOME and self._notif_queue
                and self._current_notif is None):
            self._current_notif = self._notif_queue.pop(0)
            self._state = LCDState.NOTIFICATION
            self._render()

        # Periodic redraw
        if now - self._last_render > self._render_rate:
            self._last_render = now
            self._render()

        return action

    def _handle_touch(self, tx: int, ty: int) -> Optional[Dict[str, Any]]:
        """Handle touch at display coordinates."""

        if self._state == LCDState.HOME:
            return self._touch_home(tx, ty)
        elif self._state == LCDState.NOTIFICATION:
            return self._touch_notification(tx, ty)
        elif self._state == LCDState.TIMER_SETUP:
            return self._touch_timer_setup(tx, ty)
        elif self._state == LCDState.TIMER_RUN:
            return self._touch_timer_run(tx, ty)
        elif self._state == LCDState.TIMER_DONE:
            return self._touch_timer_done(tx, ty)
        elif self._state == LCDState.WALLPAPER:
            return self._touch_wallpaper(tx, ty)
        elif self._state == LCDState.WATER_GOAL_EDIT:
            return self._touch_water_goal(tx, ty)
        elif self._state == LCDState.HYDRATION_DETAIL:
            return self._touch_hydration_detail(tx, ty)
        return None

    def _touch_home(self, tx, ty) -> Optional[Dict]:
        if self._hits.get("wallpaper") and touch_in(tx, ty, *self._hits["wallpaper"]):
            self._state = LCDState.WALLPAPER
            self._render()
        elif self._hits.get("water") and touch_in(tx, ty, *self._hits["water"]):
            self._state = LCDState.HYDRATION_DETAIL
            self._render()
        elif self._hits.get("timer") and touch_in(tx, ty, *self._hits["timer"]):
            self._state = LCDState.TIMER_SETUP
            self._render()
        elif self._hits.get("notif") and self._hits["notif"] and touch_in(tx, ty, *self._hits["notif"]):
            if self._notif_queue:
                self._current_notif = self._notif_queue.pop(0)
            self._state = LCDState.NOTIFICATION
            self._render()
        return None

    def _touch_notification(self, tx, ty) -> Optional[Dict]:
        if self._hits.get("ack") and touch_in(tx, ty, *self._hits["ack"]):
            self._current_notif = None
            self._state = LCDState.HOME
            self._render()
        return None

    def _touch_timer_setup(self, tx, ty) -> Optional[Dict]:
        if self._hits.get("start") and touch_in(tx, ty, *self._hits["start"]):
            total = self._timer_h * 3600 + self._timer_m * 60 + self._timer_s
            if total > 0:
                duration_min = max(1, total // 60)
                self._state = LCDState.HOME
                self._render()
                return {"action": "start_focus", "duration_min": duration_min}
        elif self._hits.get("cancel") and touch_in(tx, ty, *self._hits["cancel"]):
            self._state = LCDState.HOME
            self._render()
        else:
            for i in range(3):
                if self._hits.get(f"field_{i}") and touch_in(tx, ty, *self._hits[f"field_{i}"]):
                    self._editing_field = i
                if self._hits.get(f"up_{i}") and touch_in(tx, ty, *self._hits[f"up_{i}"]):
                    if i == 0:
                        self._timer_h = (self._timer_h + 1) % 24
                    elif i == 1:
                        self._timer_m = (self._timer_m + 1) % 60
                    else:
                        self._timer_s = (self._timer_s + 1) % 60
                if self._hits.get(f"down_{i}") and touch_in(tx, ty, *self._hits[f"down_{i}"]):
                    if i == 0:
                        self._timer_h = (self._timer_h - 1) % 24
                    elif i == 1:
                        self._timer_m = (self._timer_m - 1) % 60
                    else:
                        self._timer_s = (self._timer_s - 1) % 60
            self._render()
        return None

    def _touch_timer_run(self, tx, ty) -> Optional[Dict]:
        if self._hits.get("end") and touch_in(tx, ty, *self._hits["end"]):
            self._state = LCDState.HOME
            self._render()
            return {"action": "end_focus"}
        return None

    def _touch_timer_done(self, tx, ty) -> Optional[Dict]:
        if self._hits.get("ack") and touch_in(tx, ty, *self._hits["ack"]):
            self._state = LCDState.HOME
            self._render()
        return None

    def _touch_wallpaper(self, tx, ty) -> Optional[Dict]:
        if self._hits.get("back") and touch_in(tx, ty, *self._hits["back"]):
            self._state = LCDState.HOME
            self._render()
        else:
            for i in range(len(self._wallpapers)):
                key = f"wp_{i}"
                if key in self._hits and touch_in(tx, ty, *self._hits[key]):
                    self._wallpaper_idx = i
                    self._wallpaper_img = load_wallpaper(self._wallpapers[i])
            self._render()
        return None

    def _touch_water_goal(self, tx, ty) -> Optional[Dict]:
        if self._hits.get("minus") and touch_in(tx, ty, *self._hits["minus"]):
            self._edit_goal_ml = max(250, self._edit_goal_ml - 250)
            self._render()
        elif self._hits.get("plus") and touch_in(tx, ty, *self._hits["plus"]):
            self._edit_goal_ml = min(5000, self._edit_goal_ml + 250)
            self._render()
        elif self._hits.get("done") and touch_in(tx, ty, *self._hits["done"]):
            self._state = LCDState.HYDRATION_DETAIL
            self._render()
            return {"action": "set_water_goal", "goal_ml": self._edit_goal_ml}
        return None

    def _touch_hydration_detail(self, tx, ty) -> Optional[Dict]:
        if self._hits.get("set_goal") and touch_in(tx, ty, *self._hits["set_goal"]):
            self._edit_goal_ml = self._hydration_status.get("goal_ml", 2000)
            self._state = LCDState.WATER_GOAL_EDIT
            self._render()
        elif self._hits.get("back") and touch_in(tx, ty, *self._hits["back"]):
            self._state = LCDState.HOME
            self._render()
        return None

    def _render(self) -> None:
        """Render current screen and push to display."""
        if not self._driver:
            return

        try:
            img, hits = self._render_current_screen()
            self._hits = hits
            self._driver.push(img)
        except Exception as e:
            logger.error(f"LCD render error: {e}")

    def _render_current_screen(self):
        """Dispatch to appropriate screen renderer."""
        if self._state == LCDState.HOME:
            # Sync timer display state with actual focus session
            timer_status = self._timer_status
            if timer_status and timer_status.get("active"):
                phase = timer_status.get("phase", "")
                # Show TIMER_RUN screen if we're in a focus/break session
                # but only if we were previously on home (auto-switch)
                pass  # Stay on home, show inline timer info

            return screens.render_home(
                posture_state=self._posture_state,
                focus_state=self._focus_state,
                hydration_status=self._hydration_status,
                timer_status=self._timer_status,
                wallpaper_img=self._wallpaper_img,
                notif_pending=bool(self._notif_queue),
            )

        elif self._state == LCDState.NOTIFICATION:
            notif = self._current_notif or {"msg": "Notification", "detail": ""}
            return screens.render_notification(notif["msg"], notif.get("detail", ""))

        elif self._state == LCDState.TIMER_SETUP:
            return screens.render_timer_setup(
                self._timer_h, self._timer_m, self._timer_s, self._editing_field,
            )

        elif self._state == LCDState.TIMER_RUN:
            timer = self._timer_status or {}
            remaining = int(timer.get("remaining_seconds", 0))
            total = int(timer.get("target_seconds", remaining))
            return screens.render_timer_run(remaining, total)

        elif self._state == LCDState.TIMER_DONE:
            return screens.render_timer_done()

        elif self._state == LCDState.WALLPAPER:
            return screens.render_wallpaper_picker(self._wallpaper_idx, self._wallpapers)

        elif self._state == LCDState.WATER_GOAL_EDIT:
            return screens.render_water_goal_edit(self._edit_goal_ml)

        elif self._state == LCDState.HYDRATION_DETAIL:
            return screens.render_hydration_detail(self._hydration_status)

        # Fallback
        return screens.render_home(
            "unknown", "unknown", {}, None, self._wallpaper_img, False,
        )

    def shutdown(self) -> None:
        """Clean up LCD resources."""
        if self._driver:
            self._driver.shutdown()
            logger.info("LCD shutdown complete")
