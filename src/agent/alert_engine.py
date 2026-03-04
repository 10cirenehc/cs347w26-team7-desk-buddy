"""
Adaptive alert engine with desk and voice actions.

Monitors state and triggers alerts based on:
- Session context (different behavior during focus vs idle)
- Recent trends (posture degrading? focus dropping?)
- Configurable rules and thresholds
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..perception.state_history import StateHistory
    from ..desk.desk_client import DeskClient
    from ..voice.text_to_speech import TextToSpeech
    from .focus_session import FocusSessionManager
    from ..events import EventBus

logger = logging.getLogger(__name__)


class AlertAction(Enum):
    """Type of action to take for an alert."""
    VOICE = "voice"                 # TTS announcement only
    DESK_STAND = "desk_stand"       # Move desk to standing
    DESK_SIT = "desk_sit"           # Move desk to sitting
    DESK_NUDGE = "desk_nudge"       # Brief movement reminder
    VOICE_AND_DESK = "voice_and_desk"  # Both voice and desk


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = 0       # Encouragement, tips
    MEDIUM = 1    # Gentle reminders
    HIGH = 2      # Important notifications
    URGENT = 3    # Requires immediate attention


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    condition: Callable[['StateHistory'], bool]
    action: AlertAction
    message_template: str           # For voice (can include {placeholders})
    cooldown_seconds: float
    priority: AlertPriority
    enabled: bool = True
    requires_focus_session: Optional[bool] = None  # None = always, True = only during, False = only outside
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggeredAlert:
    """Record of a triggered alert."""
    rule_name: str
    timestamp: float
    message: str
    action: AlertAction
    priority: AlertPriority


class AlertEngine:
    """
    Monitors state and triggers alerts via voice or desk movement.

    Features:
    - Adaptive rules based on session context
    - Cooldown management to prevent alert fatigue
    - Priority-based alert selection
    - Desk movement integration for physical reminders

    Usage:
        alerts = AlertEngine(desk_client=desk, tts=tts)

        # In main loop:
        await alerts.check_and_execute(history, session)
    """

    def __init__(
        self,
        desk_client: Optional['DeskClient'] = None,
        tts: Optional['TextToSpeech'] = None,
        enabled: bool = True,
        demo_mode: bool = False,
        event_bus: Optional['EventBus'] = None,
    ):
        """
        Initialize alert engine.

        Args:
            desk_client: DeskClient for desk movements
            tts: TextToSpeech for voice alerts
            enabled: Whether alerts are enabled
            demo_mode: Use shortened thresholds for demo/presentation
            event_bus: EventBus for emitting alert events
        """
        self.desk = desk_client
        self.tts = tts
        self.enabled = enabled
        self.demo_mode = demo_mode
        self.event_bus = event_bus

        self._rules: List[AlertRule] = []
        self._last_triggered: Dict[str, float] = {}
        self._alert_history: List[TriggeredAlert] = []
        self._max_history = 100

        # Build default rules
        self._build_default_rules()

    def _build_default_rules(self) -> None:
        """Build the default set of alert rules."""
        self._rules = []
        demo = self.demo_mode

        # ----- Focus Session Rules (gentler during focus) -----

        # Posture degrading during focus - silent nudge first
        self._rules.append(AlertRule(
            name="focus_posture_degrading",
            condition=lambda h: h.state_ratio("posture", "bad", 30 if demo else 300) > 0.6,
            action=AlertAction.DESK_NUDGE,
            message_template="",  # Silent nudge during focus
            cooldown_seconds=20 if demo else 600,
            priority=AlertPriority.MEDIUM,
            requires_focus_session=True,
        ))

        # Severe slouch during focus - stand up
        self._rules.append(AlertRule(
            name="focus_severe_slouch",
            condition=lambda h: h.state_ratio("posture", "bad", 60 if demo else 900) > 0.75,
            action=AlertAction.VOICE_AND_DESK,
            message_template="Your posture needs attention. Standing for a moment.",
            cooldown_seconds=30 if demo else 1800,
            priority=AlertPriority.HIGH,
            requires_focus_session=True,
        ))

        # ----- Idle (non-focus) Rules -----

        # Bad posture outside focus - more proactive
        self._rules.append(AlertRule(
            name="idle_bad_posture",
            condition=lambda h: h.state_ratio("posture", "bad", 40 if demo else 600) > 0.7,
            action=AlertAction.VOICE_AND_DESK,
            message_template="Time to stand! You've been slouching for a while.",
            cooldown_seconds=30 if demo else 1800,
            priority=AlertPriority.MEDIUM,
            requires_focus_session=False,
        ))

        # Sitting too long (presence is stable, keep duration_in_state)
        self._rules.append(AlertRule(
            name="sitting_too_long",
            condition=lambda h: h.duration_in_state("presence", "seated") > (60 if demo else 3600),
            action=AlertAction.VOICE,
            message_template="You've been sitting for an hour. Consider standing for a few minutes.",
            cooldown_seconds=30 if demo else 3600,
            priority=AlertPriority.LOW,
            requires_focus_session=False,
        ))

        # ----- Always Active Rules -----

        # Standing too long (presence is stable, keep duration_in_state)
        self._rules.append(AlertRule(
            name="standing_too_long",
            condition=lambda h: h.duration_in_state("presence", "standing") > (50 if demo else 2700),
            action=AlertAction.VOICE,
            message_template="You've been standing for 45 minutes. Take a seat if you need a rest.",
            cooldown_seconds=30 if demo else 2700,
            priority=AlertPriority.LOW,
            requires_focus_session=None,  # Always active
        ))

        # Good posture streak - positive reinforcement
        self._rules.append(AlertRule(
            name="good_posture_streak",
            condition=lambda h: h.state_ratio("posture", "good", 40 if demo else 1800) > 0.9,
            action=AlertAction.VOICE,
            message_template="Great job! 30 minutes of excellent posture.",
            cooldown_seconds=30 if demo else 3600,
            priority=AlertPriority.LOW,
            requires_focus_session=None,
        ))

        # Phone distraction warning
        self._rules.append(AlertRule(
            name="phone_distraction",
            condition=lambda h: (
                h.get_current() is not None and
                h.get_current().phone_detected and
                h.state_ratio("focus", "distracted", 20 if demo else 120) > 0.7
            ),
            action=AlertAction.VOICE,
            message_template="I notice you're on your phone. Ready to get back to work?",
            cooldown_seconds=20 if demo else 900,
            priority=AlertPriority.MEDIUM,
            requires_focus_session=True,  # Only during focus sessions
        ))

    def add_rule(self, rule: AlertRule) -> None:
        """Add a custom alert rule."""
        self._rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                del self._rules[i]
                return True
        return False

    def enable_rule(self, name: str) -> bool:
        """Enable a rule by name."""
        for rule in self._rules:
            if rule.name == name:
                rule.enabled = True
                return True
        return False

    def disable_rule(self, name: str) -> bool:
        """Disable a rule by name."""
        for rule in self._rules:
            if rule.name == name:
                rule.enabled = False
                return True
        return False

    async def check_and_execute(
        self,
        history: 'StateHistory',
        session: Optional['FocusSessionManager'] = None,
    ) -> Optional[TriggeredAlert]:
        """
        Check rules and execute highest priority triggered alert.

        Args:
            history: StateHistory to evaluate rules against
            session: FocusSessionManager for context (optional)

        Returns:
            TriggeredAlert if an alert was triggered, None otherwise
        """
        if not self.enabled:
            return None

        now = time.time()
        in_focus_session = session is not None and session.is_active

        triggered = []

        for rule in self._rules:
            # Skip disabled rules
            if not rule.enabled:
                continue

            # Check session context
            if rule.requires_focus_session is not None:
                if rule.requires_focus_session and not in_focus_session:
                    continue
                if not rule.requires_focus_session and in_focus_session:
                    continue

            # Check cooldown
            last = self._last_triggered.get(rule.name, 0)
            if now - last < rule.cooldown_seconds:
                continue

            # Check condition
            try:
                if rule.condition(history):
                    triggered.append(rule)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")

        if not triggered:
            return None

        # Select highest priority rule
        rule = max(triggered, key=lambda r: r.priority.value)
        self._last_triggered[rule.name] = now

        # Format message
        message = self._format_message(rule, history)

        # Execute action
        await self._execute_action(rule.action, message)

        # Record alert
        alert = TriggeredAlert(
            rule_name=rule.name,
            timestamp=now,
            message=message,
            action=rule.action,
            priority=rule.priority,
        )
        self._alert_history.append(alert)

        # Trim history
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history:]

        # Emit event for LCD and other subscribers
        if self.event_bus:
            from ..events import Event, EventType
            self.event_bus.emit(Event(
                type=EventType.ALERT_TRIGGERED,
                data={
                    "rule_name": rule.name,
                    "message": message,
                    "priority": rule.priority.value,
                },
            ))

        logger.info(f"Alert triggered: {rule.name} - {message}")
        return alert

    def _format_message(self, rule: AlertRule, history: 'StateHistory') -> str:
        """Format alert message with placeholders."""
        message = rule.message_template

        if not message:
            return ""

        # Replace common placeholders
        try:
            current = history.get_current()

            replacements = {
                "{posture_state}": current.posture_state if current else "unknown",
                "{focus_state}": current.focus_state if current else "unknown",
                "{bad_posture_duration}": self._format_duration(
                    history.duration_in_state("posture", "bad")
                ),
                "{good_posture_duration}": self._format_duration(
                    history.duration_in_state("posture", "good")
                ),
                "{seated_duration}": self._format_duration(
                    history.duration_in_state("presence", "seated")
                ),
                "{standing_duration}": self._format_duration(
                    history.duration_in_state("presence", "standing")
                ),
            }

            for placeholder, value in replacements.items():
                message = message.replace(placeholder, str(value))

        except Exception as e:
            logger.warning(f"Error formatting message: {e}")

        return message

    async def _execute_action(self, action: AlertAction, message: str) -> None:
        """Execute the specified action."""
        try:
            if action == AlertAction.VOICE:
                if self.tts and message:
                    self.tts.speak(message)

            elif action == AlertAction.DESK_STAND:
                if self.desk:
                    await self.desk.stand()

            elif action == AlertAction.DESK_SIT:
                if self.desk:
                    await self.desk.sit()

            elif action == AlertAction.DESK_NUDGE:
                if self.desk:
                    await self.desk.nudge_up(500)

            elif action == AlertAction.VOICE_AND_DESK:
                if self.tts and message:
                    self.tts.speak(message)
                if self.desk:
                    await self.desk.stand()

        except Exception as e:
            logger.error(f"Error executing alert action {action}: {e}")

    def _format_duration(self, seconds: float) -> str:
        """Format seconds as human-readable duration."""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"{mins} minute{'s' if mins != 1 else ''}"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            if mins > 0:
                return f"{hours} hour{'s' if hours != 1 else ''} {mins} min"
            return f"{hours} hour{'s' if hours != 1 else ''}"

    def get_alert_history(self, limit: int = 10) -> List[TriggeredAlert]:
        """Get recent alert history."""
        return self._alert_history[-limit:]

    def get_rules(self) -> List[AlertRule]:
        """Get all rules."""
        return self._rules.copy()

    def reset_cooldowns(self) -> None:
        """Reset all cooldowns (allow immediate re-triggering)."""
        self._last_triggered.clear()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the alert engine."""
        self.enabled = enabled
        if not enabled:
            logger.info("Alert engine disabled")
        else:
            logger.info("Alert engine enabled")
