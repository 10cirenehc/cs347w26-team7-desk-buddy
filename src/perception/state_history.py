"""
Ring buffer and query API for state history.

Provides convenient methods for querying duration in states,
time-windowed statistics, and trends for agent context building.
"""

import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .state_logger import StateLogger, StateSnapshot


class TrendDirection(Enum):
    """Direction of a metric trend."""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass
class TrendResult:
    """Result of trend analysis."""
    direction: TrendDirection
    slope: float                    # Change per second
    start_value: float
    end_value: float
    window_seconds: float


class StateHistory:
    """
    Query interface for state history.

    Provides methods for:
    - Duration in current state
    - Time spent in state over window
    - State ratio (percentage in state)
    - Trend analysis

    All queries are based on in-memory snapshots, so they're fast
    but limited to max_memory_snapshots worth of history.

    Usage:
        history = logger.get_history()

        # How long have I been slouching?
        bad_duration = history.duration_in_state("posture", "bad")

        # What % of last 5 minutes was I focused?
        focus_pct = history.state_ratio("focus", "focused", window_seconds=300)

        # Is my posture getting worse?
        trend = history.get_trend("posture_smoothed_prob", window_seconds=300)
    """

    def __init__(self, logger: 'StateLogger'):
        self._logger = logger

    def get_current(self) -> Optional['StateSnapshot']:
        """Get the most recent state snapshot."""
        return self._logger.get_current()

    def get_snapshots(self, limit: Optional[int] = None) -> List['StateSnapshot']:
        """Get recent snapshots."""
        return self._logger.get_snapshots(limit)

    def duration_in_state(self, state_type: str, state_value: str) -> float:
        """
        Get duration (seconds) currently in the specified state.

        Returns 0 if not currently in that state.

        Args:
            state_type: "posture", "focus", or "presence"
            state_value: The state value (e.g., "good", "bad", "focused")

        Returns:
            Seconds in current state, or 0 if not in that state
        """
        snapshots = self._logger.get_snapshots()
        if not snapshots:
            return 0.0

        current = snapshots[-1]
        current_state = self._get_state_value(current, state_type)

        if current_state != state_value:
            return 0.0

        # Walk backwards to find when we entered this state
        now = time.time()
        state_start = current.timestamp

        for snap in reversed(snapshots[:-1]):
            snap_state = self._get_state_value(snap, state_type)
            if snap_state != state_value:
                break
            state_start = snap.timestamp

        return now - state_start

    def time_in_state_last_n_minutes(
        self,
        state_type: str,
        state_value: str,
        minutes: float
    ) -> float:
        """
        Get total time (seconds) spent in state over last N minutes.

        Args:
            state_type: "posture", "focus", or "presence"
            state_value: The state value
            minutes: Window size in minutes

        Returns:
            Total seconds in state during window
        """
        return self.time_in_state_window(state_type, state_value, minutes * 60)

    def time_in_state_window(
        self,
        state_type: str,
        state_value: str,
        window_seconds: float
    ) -> float:
        """
        Get total time (seconds) spent in state over window.

        Args:
            state_type: "posture", "focus", or "presence"
            state_value: The state value
            window_seconds: Window size in seconds

        Returns:
            Total seconds in state during window
        """
        snapshots = self._get_snapshots_in_window(window_seconds)
        if len(snapshots) < 2:
            return 0.0

        total_time = 0.0
        for i in range(len(snapshots) - 1):
            snap = snapshots[i]
            next_snap = snapshots[i + 1]
            dt = next_snap.timestamp - snap.timestamp

            if self._get_state_value(snap, state_type) == state_value:
                total_time += dt

        return total_time

    def state_ratio(
        self,
        state_type: str,
        state_value: str,
        window_seconds: float = 300
    ) -> float:
        """
        Get percentage of time in state over window (0.0 to 1.0).

        Args:
            state_type: "posture", "focus", or "presence"
            state_value: The state value
            window_seconds: Window size in seconds (default 5 min)

        Returns:
            Ratio of time in state (0.0 to 1.0)
        """
        snapshots = self._get_snapshots_in_window(window_seconds)
        if len(snapshots) < 2:
            return 0.0

        total_time = 0.0
        state_time = 0.0

        for i in range(len(snapshots) - 1):
            snap = snapshots[i]
            next_snap = snapshots[i + 1]
            dt = next_snap.timestamp - snap.timestamp
            total_time += dt

            if self._get_state_value(snap, state_type) == state_value:
                state_time += dt

        return state_time / total_time if total_time > 0 else 0.0

    def state_counts(
        self,
        state_type: str,
        window_seconds: float = 300
    ) -> Dict[str, int]:
        """
        Get count of snapshots in each state over window.

        Args:
            state_type: "posture", "focus", or "presence"
            window_seconds: Window size in seconds

        Returns:
            Dict mapping state values to counts
        """
        snapshots = self._get_snapshots_in_window(window_seconds)
        counts: Dict[str, int] = {}

        for snap in snapshots:
            state = self._get_state_value(snap, state_type)
            counts[state] = counts.get(state, 0) + 1

        return counts

    def get_trend(
        self,
        metric: str,
        window_seconds: float = 300
    ) -> TrendResult:
        """
        Analyze trend of a numeric metric over window.

        Args:
            metric: Metric name (e.g., "posture_smoothed_prob", "forward_lean_z")
            window_seconds: Window size in seconds

        Returns:
            TrendResult with direction, slope, and values
        """
        snapshots = self._get_snapshots_in_window(window_seconds)

        if len(snapshots) < 10:
            return TrendResult(
                direction=TrendDirection.UNKNOWN,
                slope=0.0,
                start_value=0.0,
                end_value=0.0,
                window_seconds=window_seconds,
            )

        # Extract metric values
        values = []
        times = []
        for snap in snapshots:
            val = self._get_metric_value(snap, metric)
            if val is not None:
                values.append(val)
                times.append(snap.timestamp)

        if len(values) < 10:
            return TrendResult(
                direction=TrendDirection.UNKNOWN,
                slope=0.0,
                start_value=0.0,
                end_value=0.0,
                window_seconds=window_seconds,
            )

        # Simple linear regression for slope
        import numpy as np
        times_arr = np.array(times) - times[0]
        values_arr = np.array(values)

        # slope = covariance(t, v) / variance(t)
        mean_t = np.mean(times_arr)
        mean_v = np.mean(values_arr)
        slope = np.sum((times_arr - mean_t) * (values_arr - mean_v)) / np.sum((times_arr - mean_t) ** 2)

        # Determine direction based on metric type
        # For posture_smoothed_prob: increasing = degrading (more p_bad)
        # For forward_lean_z: decreasing (more negative) = degrading
        start_val = float(values_arr[:len(values_arr)//5].mean())
        end_val = float(values_arr[-len(values_arr)//5:].mean())

        # Determine significance threshold (5% change over window)
        threshold = 0.05 * abs(mean_v) if abs(mean_v) > 0.1 else 0.01

        if abs(slope * window_seconds) < threshold:
            direction = TrendDirection.STABLE
        elif metric in ["posture_smoothed_prob", "posture_raw_prob"]:
            # Higher prob = worse posture
            direction = TrendDirection.DEGRADING if slope > 0 else TrendDirection.IMPROVING
        elif metric == "forward_lean_z":
            # More negative = leaning forward = worse
            direction = TrendDirection.DEGRADING if slope < 0 else TrendDirection.IMPROVING
        else:
            # Generic: assume increasing is degrading
            direction = TrendDirection.DEGRADING if slope > 0 else TrendDirection.IMPROVING

        return TrendResult(
            direction=direction,
            slope=float(slope),
            start_value=start_val,
            end_value=end_val,
            window_seconds=window_seconds,
        )

    def get_state_durations(
        self,
        state_type: str,
        window_seconds: float = 3600
    ) -> List[Dict[str, Any]]:
        """
        Get list of state periods with their durations.

        Useful for detailed session analysis.

        Args:
            state_type: "posture", "focus", or "presence"
            window_seconds: Window size in seconds

        Returns:
            List of dicts with state, start_time, duration
        """
        snapshots = self._get_snapshots_in_window(window_seconds)
        if len(snapshots) < 2:
            return []

        periods = []
        current_state = self._get_state_value(snapshots[0], state_type)
        period_start = snapshots[0].timestamp

        for i in range(1, len(snapshots)):
            snap = snapshots[i]
            state = self._get_state_value(snap, state_type)

            if state != current_state:
                # End of period
                periods.append({
                    "state": current_state,
                    "start_time": period_start,
                    "duration": snap.timestamp - period_start,
                })
                current_state = state
                period_start = snap.timestamp

        # Add final period
        if snapshots:
            periods.append({
                "state": current_state,
                "start_time": period_start,
                "duration": snapshots[-1].timestamp - period_start,
            })

        return periods

    def get_summary(self, window_seconds: float = 3600) -> Dict[str, Any]:
        """
        Get summary statistics for agent context.

        Args:
            window_seconds: Window size in seconds

        Returns:
            Dict with various summary statistics
        """
        snapshots = self._get_snapshots_in_window(window_seconds)
        current = self.get_current()

        if not snapshots or not current:
            return {
                "has_data": False,
                "snapshot_count": 0,
            }

        actual_window = snapshots[-1].timestamp - snapshots[0].timestamp if len(snapshots) > 1 else 0

        return {
            "has_data": True,
            "snapshot_count": len(snapshots),
            "window_seconds": actual_window,

            # Current state
            "current": {
                "posture": current.posture_state,
                "focus": current.focus_state,
                "presence": current.presence_state,
                "face_detected": current.face_detected,
                "phone_detected": current.phone_detected,
            },

            # Current state durations
            "durations": {
                "bad_posture_seconds": self.duration_in_state("posture", "bad"),
                "good_posture_seconds": self.duration_in_state("posture", "good"),
                "distracted_seconds": self.duration_in_state("focus", "distracted"),
                "focused_seconds": self.duration_in_state("focus", "focused"),
                "standing_seconds": self.duration_in_state("presence", "standing"),
                "seated_seconds": self.duration_in_state("presence", "seated"),
            },

            # Session ratios
            "ratios": {
                "good_posture": self.state_ratio("posture", "good", window_seconds),
                "focused": self.state_ratio("focus", "focused", window_seconds),
                "seated": self.state_ratio("presence", "seated", window_seconds),
            },

            # Trends
            "trends": {
                "posture": self.get_trend("posture_smoothed_prob", min(300, window_seconds)).direction.value,
            },
        }

    def _get_snapshots_in_window(self, window_seconds: float) -> List['StateSnapshot']:
        """Get snapshots within the time window from now."""
        snapshots = self._logger.get_snapshots()
        if not snapshots:
            return []

        cutoff = time.time() - window_seconds
        return [s for s in snapshots if s.timestamp >= cutoff]

    def _get_state_value(self, snapshot: 'StateSnapshot', state_type: str) -> str:
        """Extract state value from snapshot by type."""
        if state_type == "posture":
            return snapshot.posture_state
        elif state_type == "focus":
            return snapshot.focus_state
        elif state_type == "presence":
            return snapshot.presence_state
        elif state_type == "attention":
            return snapshot.attention_state
        else:
            return "unknown"

    def _get_metric_value(self, snapshot: 'StateSnapshot', metric: str) -> Optional[float]:
        """Extract numeric metric value from snapshot."""
        if hasattr(snapshot, metric):
            val = getattr(snapshot, metric)
            return float(val) if val is not None else None
        return None
