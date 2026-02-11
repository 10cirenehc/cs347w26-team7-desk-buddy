"""
Natural language summaries of state history.

Generates human-readable summaries for:
- Agent context
- User queries
- Session reports
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .state_history import StateHistory

logger = logging.getLogger(__name__)


@dataclass
class StateSummary:
    """Natural language summary of state."""
    short_summary: str          # One-line summary
    detailed_summary: str       # Multi-sentence summary
    key_metrics: Dict[str, Any] # Numeric metrics
    recommendations: List[str]  # Actionable suggestions


class StateSummarizer:
    """
    Generates natural language summaries from state history.

    Provides formatted summaries for different contexts:
    - Quick status checks
    - Detailed session reports
    - Agent context building
    """

    def __init__(self, history: 'StateHistory'):
        """
        Initialize summarizer.

        Args:
            history: StateHistory to summarize
        """
        self.history = history

    def get_short_summary(self) -> str:
        """
        Get one-line status summary.

        Returns:
            Short summary string
        """
        current = self.history.get_current()

        if not current:
            return "No tracking data available yet."

        posture = current.posture_state
        focus = current.focus_state

        if posture == "good" and focus == "focused":
            return "You're doing great - good posture and focused!"
        elif posture == "bad" and focus == "distracted":
            return "Heads up - your posture needs attention and you seem distracted."
        elif posture == "bad":
            return "Your posture could use some attention."
        elif focus == "distracted":
            return "You seem a bit distracted."
        else:
            return "Everything looks okay."

    def get_detailed_summary(self, window_seconds: float = 300) -> StateSummary:
        """
        Get detailed summary with metrics and recommendations.

        Args:
            window_seconds: Time window to summarize

        Returns:
            StateSummary with all details
        """
        raw_summary = self.history.get_summary(window_seconds)
        current = self.history.get_current()

        if not raw_summary.get("has_data") or not current:
            return StateSummary(
                short_summary="No tracking data available yet.",
                detailed_summary="I need to observe you for a bit longer before I can provide a summary.",
                key_metrics={},
                recommendations=["Make sure you're visible to the camera."],
            )

        # Build detailed summary
        parts = []
        recommendations = []

        # Current state
        posture = current.posture_state
        focus = current.focus_state
        presence = current.presence_state

        # Durations
        durations = raw_summary.get("durations", {})
        ratios = raw_summary.get("ratios", {})

        # Posture analysis
        if posture == "bad":
            bad_duration = durations.get("bad_posture_seconds", 0)
            parts.append(f"Your posture needs attention - you've been slouching for {self._format_duration(bad_duration)}.")
            recommendations.append("Try sitting up straighter with your shoulders back.")
            if bad_duration > 600:  # 10 minutes
                recommendations.append("Consider standing up to reset your posture.")
        elif posture == "good":
            good_ratio = ratios.get("good_posture", 0)
            if good_ratio > 0.8:
                parts.append(f"Excellent posture! You've maintained good posture {good_ratio:.0%} of the time.")
            else:
                parts.append("Your posture has been mostly good.")

        # Focus analysis
        if focus == "distracted":
            distracted_duration = durations.get("distracted_seconds", 0)
            parts.append(f"You seem distracted - {self._format_duration(distracted_duration)} of inattention.")
            if current.phone_detected:
                recommendations.append("Put your phone away to reduce distractions.")
            else:
                recommendations.append("Try to refocus on your current task.")
        elif focus == "focused":
            focused_ratio = ratios.get("focused", 0)
            if focused_ratio > 0.8:
                parts.append(f"Great focus! You've been attentive {focused_ratio:.0%} of the time.")

        # Presence/sitting analysis
        if presence == "seated":
            seated_duration = durations.get("seated_seconds", 0)
            if seated_duration > 3600:  # 1 hour
                parts.append(f"You've been sitting for {self._format_duration(seated_duration)}.")
                recommendations.append("Consider standing up for a few minutes.")

        # Combine parts
        detailed = " ".join(parts) if parts else "Everything looks okay so far."

        # Key metrics
        key_metrics = {
            "posture_state": posture,
            "focus_state": focus,
            "presence_state": presence,
            "good_posture_ratio": ratios.get("good_posture", 0),
            "focused_ratio": ratios.get("focused", 0),
            "bad_posture_duration": durations.get("bad_posture_seconds", 0),
            "distracted_duration": durations.get("distracted_seconds", 0),
            "seated_duration": durations.get("seated_seconds", 0),
        }

        return StateSummary(
            short_summary=self.get_short_summary(),
            detailed_summary=detailed,
            key_metrics=key_metrics,
            recommendations=recommendations,
        )

    def get_session_report(self, session_duration_seconds: float) -> str:
        """
        Generate a session completion report.

        Args:
            session_duration_seconds: Duration of the completed session

        Returns:
            Report text
        """
        summary = self.history.get_summary(session_duration_seconds)

        if not summary.get("has_data"):
            return "Session completed, but I don't have tracking data for a detailed report."

        ratios = summary.get("ratios", {})
        good_posture = ratios.get("good_posture", 0)
        focused = ratios.get("focused", 0)

        parts = []

        # Overall assessment
        if good_posture > 0.8 and focused > 0.8:
            parts.append("Excellent session!")
        elif good_posture > 0.6 and focused > 0.6:
            parts.append("Good session overall.")
        else:
            parts.append("Session complete.")

        # Posture stats
        parts.append(f"Posture: {good_posture:.0%} of the time you had good posture.")

        # Focus stats
        parts.append(f"Focus: You were attentive {focused:.0%} of the time.")

        # Get state durations for more detail
        state_periods = self.history.get_state_durations("posture", session_duration_seconds)
        bad_periods = [p for p in state_periods if p["state"] == "bad"]

        if bad_periods:
            longest_bad = max(p["duration"] for p in bad_periods)
            if longest_bad > 300:  # 5 minutes
                parts.append(f"Your longest slouching period was {self._format_duration(longest_bad)}.")

        return " ".join(parts)

    def get_daily_report(self) -> str:
        """
        Generate a daily summary report.

        Returns:
            Daily report text
        """
        # Use 8 hours as max daily window
        summary = self.history.get_summary(8 * 3600)

        if not summary.get("has_data"):
            return "I don't have enough data for a daily report yet."

        ratios = summary.get("ratios", {})
        snapshot_count = summary.get("snapshot_count", 0)

        # Estimate actual tracked time
        tracked_minutes = snapshot_count  # ~1 snapshot per second, so ~60 per minute

        if tracked_minutes < 30:
            return "I haven't tracked enough time today for a meaningful report."

        parts = [f"Daily summary ({tracked_minutes // 60} hours tracked):"]

        good_posture = ratios.get("good_posture", 0)
        focused = ratios.get("focused", 0)
        seated = ratios.get("seated", 0)

        parts.append(f"- Good posture: {good_posture:.0%}")
        parts.append(f"- Focused time: {focused:.0%}")
        parts.append(f"- Time seated: {seated:.0%}")

        # Recommendations based on day
        if good_posture < 0.5:
            parts.append("\nConsider taking more frequent breaks to reset your posture.")
        if seated > 0.9:
            parts.append("\nTry to stand more throughout the day.")
        if focused < 0.5:
            parts.append("\nConsider using focus sessions to improve concentration.")

        return "\n".join(parts)

    def get_agent_context_string(self, window_seconds: float = 300) -> str:
        """
        Get formatted context string for LLM agent.

        Args:
            window_seconds: Time window for context

        Returns:
            Formatted context string
        """
        summary = self.get_detailed_summary(window_seconds)
        current = self.history.get_current()

        if not current:
            return "No tracking data available."

        lines = [
            f"Current state: posture={current.posture_state}, focus={current.focus_state}",
        ]

        metrics = summary.key_metrics
        if metrics:
            lines.append(f"Recent metrics: {metrics.get('good_posture_ratio', 0):.0%} good posture, "
                        f"{metrics.get('focused_ratio', 0):.0%} focused")

            bad_duration = metrics.get('bad_posture_duration', 0)
            if bad_duration > 60:
                lines.append(f"Bad posture duration: {self._format_duration(bad_duration)}")

        if summary.recommendations:
            lines.append(f"Recommendations: {'; '.join(summary.recommendations[:2])}")

        return "\n".join(lines)

    def _format_duration(self, seconds: float) -> str:
        """Format seconds as human-readable duration."""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"about {mins} minute{'s' if mins != 1 else ''}"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            if mins > 0:
                return f"about {hours} hour{'s' if hours != 1 else ''} {mins} min"
            return f"about {hours} hour{'s' if hours != 1 else ''}"
