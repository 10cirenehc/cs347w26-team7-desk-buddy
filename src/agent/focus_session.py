"""
Smart productivity timer with context-aware suggestions.

Manages focus/break cycles and adapts based on detected state.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..perception.state_history import StateHistory
    from ..events import EventBus

logger = logging.getLogger(__name__)


class SessionPhase(Enum):
    """Focus session phase."""
    IDLE = "idle"           # No active session
    FOCUS = "focus"         # In focus period
    BREAK = "break"         # In break period


@dataclass
class SessionStats:
    """Statistics for a focus session."""
    target_duration_min: int = 25
    elapsed_seconds: float = 0.0
    focus_ratio: float = 0.0          # % of time actually focused (0-1)
    posture_good_ratio: float = 0.0   # % of time with good posture (0-1)
    distractions_count: int = 0       # Number of distraction events
    phone_pickups: int = 0            # Number of phone detections
    suggested_early_break: bool = False  # True if we suggested early break
    breaks_taken: int = 0


@dataclass
class SessionSuggestion:
    """A suggestion from the session manager."""
    message: str
    suggestion_type: str  # "reminder", "break_suggestion", "session_complete", "encouragement"
    priority: int = 1     # Higher = more important


class FocusSessionManager:
    """
    Manages focus/break cycles with adaptive suggestions.

    Features:
    - Standard Pomodoro-style focus/break cycles
    - Adaptive suggestions based on detected focus and posture
    - Early break recommendations when focus degrades
    - Session statistics and summaries

    Usage:
        session = FocusSessionManager(history)
        session.start_focus(duration_min=25)

        # In main loop:
        suggestion = session.check_and_suggest()
        if suggestion:
            tts.speak(suggestion.message)

        # End session:
        stats = session.end()
    """

    # Adaptive thresholds
    DEFAULT_FOCUS_DURATION_MIN = 25
    DEFAULT_BREAK_DURATION_MIN = 5
    LONG_BREAK_DURATION_MIN = 15
    SESSIONS_BEFORE_LONG_BREAK = 4

    # Early break suggestion thresholds
    EARLY_BREAK_FOCUS_THRESHOLD = 0.4  # Suggest break if < 40% focused in window
    EARLY_BREAK_CHECK_WINDOW_MIN = 5   # Check last 5 min
    EARLY_BREAK_MIN_ELAPSED_MIN = 10   # Only suggest after 10 min

    # Posture warning threshold
    POSTURE_WARNING_THRESHOLD = 0.3    # Warn if < 30% good posture

    def __init__(self, history: Optional['StateHistory'] = None, demo_mode: bool = False,
                 event_bus: Optional['EventBus'] = None):
        """
        Initialize focus session manager.

        Args:
            history: StateHistory for querying focus/posture state
            demo_mode: Use shortened thresholds for demo/presentation
            event_bus: EventBus for emitting session events
        """
        self.history = history
        self.demo_mode = demo_mode
        self.event_bus = event_bus

        self.phase = SessionPhase.IDLE
        self.start_time: Optional[float] = None
        self.target_duration_min: int = self.DEFAULT_FOCUS_DURATION_MIN
        self.stats = SessionStats()

        # Tracking
        self._sessions_completed: int = 0
        self._last_suggestion_time: float = 0.0
        self._suggestion_cooldown: float = 5.0 if demo_mode else 60.0
        self._posture_warned: bool = False
        self._distraction_events: List[float] = []

        # Override thresholds in demo mode
        if demo_mode:
            self.EARLY_BREAK_MIN_ELAPSED_MIN = 0.25   # 15 seconds
            self.EARLY_BREAK_CHECK_WINDOW_MIN = 0.25   # 15 seconds

    def _emit(self, event_type_name: str, data: dict = None) -> None:
        """Emit an event on the bus if available."""
        if not self.event_bus:
            return
        from ..events import Event, EventType
        etype = getattr(EventType, event_type_name, None)
        if etype:
            self.event_bus.emit(Event(type=etype, data=data or {}))

    def start_focus(self, duration_min: int = DEFAULT_FOCUS_DURATION_MIN) -> str:
        """
        Start a focus session.

        Args:
            duration_min: Focus session duration in minutes

        Returns:
            Confirmation message
        """
        self.phase = SessionPhase.FOCUS
        self.start_time = time.time()
        self.target_duration_min = duration_min
        self.stats = SessionStats(target_duration_min=duration_min)
        self._posture_warned = False
        self._distraction_events = []

        msg = f"Starting {duration_min}-minute focus session. I'll track your focus and posture."
        logger.info(f"Started {duration_min}-minute focus session")
        self._emit("FOCUS_STARTED", {"duration_min": duration_min, "message": msg})
        return msg

    def start_break(self, duration_min: Optional[int] = None) -> str:
        """
        Start a break.

        Args:
            duration_min: Break duration (auto-selects short/long if None)

        Returns:
            Confirmation message
        """
        # Auto-select break duration
        if duration_min is None:
            if (self._sessions_completed + 1) % self.SESSIONS_BEFORE_LONG_BREAK == 0:
                duration_min = self.LONG_BREAK_DURATION_MIN
            else:
                duration_min = self.DEFAULT_BREAK_DURATION_MIN

        self.phase = SessionPhase.BREAK
        self.start_time = time.time()
        self.target_duration_min = duration_min
        self.stats.breaks_taken += 1

        msg = f"Break time! Take {duration_min} minutes to rest your eyes and stretch."
        logger.info(f"Started {duration_min}-minute break")
        self._emit("BREAK_STARTED", {"duration_min": duration_min, "message": msg})
        return msg

    def end(self) -> SessionStats:
        """
        End the current session.

        Returns:
            Session statistics
        """
        if self.start_time:
            self.stats.elapsed_seconds = time.time() - self.start_time

        self._calculate_session_stats()
        self.phase = SessionPhase.IDLE
        self.start_time = None

        logger.info(f"Session ended: {self.stats}")
        return self.stats

    def check_and_suggest(self) -> Optional[SessionSuggestion]:
        """
        Check session state and return adaptive suggestions.

        Should be called periodically in main loop.

        Returns:
            SessionSuggestion if action needed, None otherwise
        """
        if self.phase == SessionPhase.IDLE:
            return None

        now = time.time()

        # Cooldown check
        if now - self._last_suggestion_time < self._suggestion_cooldown:
            return None

        elapsed = now - self.start_time
        elapsed_min = elapsed / 60

        suggestion = None
        if self.phase == SessionPhase.FOCUS:
            suggestion = self._check_focus_session(elapsed_min, now)
        elif self.phase == SessionPhase.BREAK:
            suggestion = self._check_break(elapsed_min)

        if suggestion and suggestion.suggestion_type != "session_complete":
            self._emit("FOCUS_SUGGESTION", {
                "message": suggestion.message,
                "suggestion_type": suggestion.suggestion_type,
            })

        return suggestion

    def _check_focus_session(self, elapsed_min: float, now: float) -> Optional[SessionSuggestion]:
        """Check focus session and generate suggestions."""

        # Check if session completed
        if elapsed_min >= self.target_duration_min:
            return self._complete_focus_session()

        # Check for degraded focus (adaptive early break suggestion)
        if elapsed_min > self.EARLY_BREAK_MIN_ELAPSED_MIN and self.history:
            recent_focus = self.history.state_ratio(
                "focus", "focused",
                window_seconds=self.EARLY_BREAK_CHECK_WINDOW_MIN * 60
            )

            if recent_focus < self.EARLY_BREAK_FOCUS_THRESHOLD:
                if not self.stats.suggested_early_break:
                    self.stats.suggested_early_break = True
                    self._last_suggestion_time = now
                    remaining = self.target_duration_min - elapsed_min

                    return SessionSuggestion(
                        message=(
                            f"You seem distracted. You've been focused only "
                            f"{recent_focus:.0%} of the last {self.EARLY_BREAK_CHECK_WINDOW_MIN} minutes. "
                            f"Want to take an early break with {remaining:.0f} minutes left?"
                        ),
                        suggestion_type="break_suggestion",
                        priority=2,
                    )

        # Check for posture degradation
        if self.history and not self._posture_warned:
            recent_posture = self.history.state_ratio(
                "posture", "good",
                window_seconds=self.EARLY_BREAK_CHECK_WINDOW_MIN * 60
            )

            if recent_posture < self.POSTURE_WARNING_THRESHOLD:
                self._posture_warned = True
                self._last_suggestion_time = now

                return SessionSuggestion(
                    message="Your posture could use some attention. Try sitting up straighter.",
                    suggestion_type="reminder",
                    priority=1,
                )

        # Periodic encouragement at halfway point
        if abs(elapsed_min - self.target_duration_min / 2) < 0.5:
            self._last_suggestion_time = now
            remaining = self.target_duration_min - elapsed_min
            return SessionSuggestion(
                message=f"Halfway there! {remaining:.0f} minutes left. Keep going!",
                suggestion_type="encouragement",
                priority=0,
            )

        return None

    def _check_break(self, elapsed_min: float) -> Optional[SessionSuggestion]:
        """Check break status."""
        if elapsed_min >= self.target_duration_min:
            return self._complete_break()
        return None

    def _complete_focus_session(self) -> SessionSuggestion:
        """Complete focus session and generate summary."""
        self._calculate_session_stats()
        self._sessions_completed += 1
        self.phase = SessionPhase.IDLE

        # Generate natural summary
        summary_parts = ["Focus session complete!"]

        if self.stats.focus_ratio > 0:
            summary_parts.append(
                f"You were focused {self.stats.focus_ratio:.0%} of the time"
            )

        if self.stats.posture_good_ratio > 0:
            summary_parts.append(
                f"with {self.stats.posture_good_ratio:.0%} good posture."
            )

        if self.stats.distractions_count > 3:
            summary_parts.append(
                f"You had {self.stats.distractions_count} distraction moments."
            )

        # Encouragement based on performance
        if self.stats.focus_ratio > 0.8:
            summary_parts.append("Great job staying focused!")
        elif self.stats.focus_ratio > 0.6:
            summary_parts.append("Decent focus, but room for improvement.")
        elif self.stats.focus_ratio > 0:
            summary_parts.append("Tough session. Consider shorter focus periods.")

        summary_parts.append("Ready for a break?")

        msg = " ".join(summary_parts)
        self._emit("FOCUS_COMPLETED", {
            "message": msg,
            "focus_ratio": self.stats.focus_ratio,
            "posture_good_ratio": self.stats.posture_good_ratio,
        })

        return SessionSuggestion(
            message=msg,
            suggestion_type="session_complete",
            priority=3,
        )

    def _complete_break(self) -> SessionSuggestion:
        """Complete break."""
        self.phase = SessionPhase.IDLE

        return SessionSuggestion(
            message="Break's over! Ready for another focus session?",
            suggestion_type="session_complete",
            priority=2,
        )

    def _calculate_session_stats(self) -> None:
        """Calculate session statistics from history."""
        if not self.history or not self.start_time:
            return

        elapsed = time.time() - self.start_time
        self.stats.elapsed_seconds = elapsed

        # Calculate ratios from history
        self.stats.focus_ratio = self.history.state_ratio(
            "focus", "focused", window_seconds=elapsed
        )
        self.stats.posture_good_ratio = self.history.state_ratio(
            "posture", "good", window_seconds=elapsed
        )

        # Count distraction events
        events = self.history._logger.get_events()
        for event in events:
            if event.timestamp >= self.start_time:
                if event.event_type == "focus_change" and event.to_state == "distracted":
                    self.stats.distractions_count += 1

    def get_status(self) -> dict:
        """
        Get current session status for agent context.

        Returns:
            Dict with session status
        """
        if self.phase == SessionPhase.IDLE:
            return {
                "active": False,
                "sessions_completed": self._sessions_completed,
            }

        elapsed = (time.time() - self.start_time) / 60 if self.start_time else 0
        remaining = max(0, self.target_duration_min - elapsed)

        return {
            "active": True,
            "phase": self.phase.value,
            "elapsed_min": elapsed,
            "target_min": self.target_duration_min,
            "remaining_min": remaining,
            "sessions_completed": self._sessions_completed,
            "suggested_early_break": self.stats.suggested_early_break,
        }

    def get_status_summary(self) -> str:
        """
        Get human-readable status summary.

        Returns:
            Status message
        """
        status = self.get_status()

        if not status["active"]:
            if status["sessions_completed"] > 0:
                return f"No active session. You've completed {status['sessions_completed']} focus sessions today."
            return "No active session. Say 'start focus' to begin."

        phase = status["phase"]
        remaining = status["remaining_min"]

        if phase == "focus":
            return f"Focus session: {remaining:.0f} minutes remaining."
        else:
            return f"Break: {remaining:.0f} minutes remaining."

    def skip_to_next(self) -> str:
        """
        Skip current phase (end early).

        Returns:
            Confirmation message
        """
        if self.phase == SessionPhase.FOCUS:
            self._calculate_session_stats()
            self._sessions_completed += 1
            self.phase = SessionPhase.IDLE
            return "Focus session ended early. Ready for a break?"
        elif self.phase == SessionPhase.BREAK:
            self.phase = SessionPhase.IDLE
            return "Break ended early. Ready for another focus session?"
        else:
            return "No active session to skip."

    def pause(self) -> str:
        """Pause current session (preserves elapsed time)."""
        # For simplicity, pause is same as end for now
        return self.end()

    @property
    def is_active(self) -> bool:
        return self.phase != SessionPhase.IDLE

    @property
    def in_focus(self) -> bool:
        return self.phase == SessionPhase.FOCUS

    @property
    def in_break(self) -> bool:
        return self.phase == SessionPhase.BREAK
