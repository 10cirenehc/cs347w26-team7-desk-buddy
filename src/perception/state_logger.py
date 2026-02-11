"""
State logging system for capturing perception pipeline output.

Captures periodic snapshots of all perception state (posture, gaze, focus,
phone, presence) for:
- Agent context building
- Historical queries ("How long have I been slouching?")
- Session summaries and analytics
"""

import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional, List, Dict, Any

import numpy as np

from .posture_state import SmoothedPostureState, PostureLabel
from .posture_features import PostureFeatures
from .gaze_tracker import GazeResult, AttentionState
from .focus_estimator import FocusEstimation, FocusState
from .presence_detector import PresenceResult, PresenceState


def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass
class StateSnapshot:
    """
    Complete perception state at a point in time.

    Logged approximately once per second for agent context and queries.
    """
    timestamp: float                    # Unix timestamp
    session_id: str                     # Unique session identifier

    # Posture
    posture_state: str                  # "good" | "bad" | "unknown"
    posture_raw_prob: float             # Raw p_bad from classifier
    posture_smoothed_prob: float        # EWMA-smoothed p_bad
    posture_confident: bool             # True if visibility was sufficient

    # Posture features (optional - for detailed logging)
    torso_pitch: Optional[float] = None
    head_forward_ratio: Optional[float] = None
    shoulder_roll: Optional[float] = None
    lateral_lean: Optional[float] = None
    head_tilt: Optional[float] = None
    forward_lean_z: Optional[float] = None

    # Gaze
    gaze_pitch: float = 0.0
    gaze_yaw: float = 0.0
    gaze_roll: float = 0.0
    attention_state: str = "unknown"    # "focused" | "looking_away" | "looking_down"
    face_detected: bool = False

    # Phone
    phone_detected: bool = False
    phone_confidence: float = 0.0

    # Presence
    presence_state: str = "away"        # "seated" | "standing" | "away"
    presence_confidence: float = 0.0

    # Focus (fused state)
    focus_state: str = "focused"        # "focused" | "distracted" | "away"
    focus_confidence: float = 0.0
    focus_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StateSnapshot':
        """Create from dictionary."""
        return cls(**d)


@dataclass
class StateEvent:
    """
    Discrete event for state transitions.

    Logged when state changes (e.g., posture good->bad, focus->distracted).
    """
    timestamp: float
    session_id: str
    event_type: str                     # "posture_change", "focus_change", "presence_change"
    from_state: str
    to_state: str
    duration_in_previous_state: float   # Seconds in previous state
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StateLogger:
    """
    Logs perception state for agent context and historical queries.

    Features:
    - Periodic snapshot logging (~1 Hz)
    - Event logging on state transitions
    - In-memory ring buffer for recent history
    - Optional file persistence for longer sessions

    Usage:
        logger = StateLogger()
        logger.start_session()

        # In perception loop:
        logger.log(posture=posture_state, gaze=gaze_result, ...)

        # Query history:
        history = logger.get_history()
        duration = history.duration_in_state("posture", "bad")
    """

    def __init__(
        self,
        log_interval_seconds: float = 1.0,
        max_memory_snapshots: int = 3600,  # 1 hour at 1 Hz
        output_dir: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize state logger.

        Args:
            log_interval_seconds: Minimum time between snapshots
            max_memory_snapshots: Maximum snapshots to keep in memory
            output_dir: Directory for persistent logs (optional)
            config: Configuration dict (overrides other args)
        """
        if config:
            log_cfg = config.get('state_logger', {})
            log_interval_seconds = log_cfg.get('log_interval_seconds', log_interval_seconds)
            max_memory_snapshots = log_cfg.get('max_memory_snapshots', max_memory_snapshots)
            output_dir = log_cfg.get('output_dir', output_dir)

        self.log_interval = log_interval_seconds
        self.max_snapshots = max_memory_snapshots
        self.output_dir = Path(output_dir) if output_dir else None

        self._session_id: Optional[str] = None
        self._snapshots: List[StateSnapshot] = []
        self._events: List[StateEvent] = []
        self._last_log_time: float = 0.0
        self._lock = Lock()

        # State tracking for event detection
        self._last_posture_state: Optional[str] = None
        self._last_focus_state: Optional[str] = None
        self._last_presence_state: Optional[str] = None
        self._state_start_times: Dict[str, float] = {}

        # Output file handle
        self._log_file = None

        # Create history interface
        from .state_history import StateHistory
        self._history = StateHistory(self)

    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new logging session.

        Args:
            session_id: Optional custom session ID

        Returns:
            Session ID
        """
        with self._lock:
            self._session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self._snapshots = []
            self._events = []
            self._last_log_time = 0.0
            self._last_posture_state = None
            self._last_focus_state = None
            self._last_presence_state = None
            self._state_start_times = {}

            # Set up file logging if output dir specified
            if self.output_dir:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                log_path = self.output_dir / f"{self._session_id}.jsonl"
                self._log_file = open(log_path, 'w')

            return self._session_id

    def end_session(self) -> Dict[str, Any]:
        """
        End the current session.

        Returns:
            Session summary statistics
        """
        with self._lock:
            summary = self._compute_session_summary()

            if self._log_file:
                # Write summary as last line
                self._log_file.write(json.dumps({
                    "type": "session_summary",
                    "data": summary
                }, default=_json_default) + "\n")
                self._log_file.close()
                self._log_file = None

            self._session_id = None
            return summary

    def log(
        self,
        posture: Optional[SmoothedPostureState] = None,
        features: Optional[PostureFeatures] = None,
        gaze: Optional[GazeResult] = None,
        phone_detected: bool = False,
        phone_confidence: float = 0.0,
        presence: Optional[PresenceResult] = None,
        focus: Optional[FocusEstimation] = None,
        force: bool = False,
    ) -> Optional[StateSnapshot]:
        """
        Log current perception state.

        Respects log_interval_seconds unless force=True.

        Args:
            posture: Posture state from state machine
            features: Raw posture features (optional)
            gaze: Gaze tracking result
            phone_detected: Whether phone detected in hand
            phone_confidence: Phone detection confidence
            presence: Presence detection result
            focus: Focus estimation result
            force: Log even if within interval

        Returns:
            StateSnapshot if logged, None if skipped
        """
        if not self._session_id:
            return None

        now = time.time()

        # Rate limiting
        if not force and (now - self._last_log_time) < self.log_interval:
            return None

        with self._lock:
            self._last_log_time = now

            # Build snapshot
            snapshot = self._build_snapshot(
                timestamp=now,
                posture=posture,
                features=features,
                gaze=gaze,
                phone_detected=phone_detected,
                phone_confidence=phone_confidence,
                presence=presence,
                focus=focus,
            )

            # Check for state transitions and emit events
            self._check_state_transitions(snapshot, now)

            # Store snapshot
            self._snapshots.append(snapshot)

            # Enforce memory limit (ring buffer behavior)
            if len(self._snapshots) > self.max_snapshots:
                self._snapshots = self._snapshots[-self.max_snapshots:]

            # Write to file if enabled
            if self._log_file:
                self._log_file.write(json.dumps({
                    "type": "snapshot",
                    "data": snapshot.to_dict()
                }, default=_json_default) + "\n")
                self._log_file.flush()

            return snapshot

    def _build_snapshot(
        self,
        timestamp: float,
        posture: Optional[SmoothedPostureState],
        features: Optional[PostureFeatures],
        gaze: Optional[GazeResult],
        phone_detected: bool,
        phone_confidence: float,
        presence: Optional[PresenceResult],
        focus: Optional[FocusEstimation],
    ) -> StateSnapshot:
        """Build a StateSnapshot from perception outputs."""

        snapshot = StateSnapshot(
            timestamp=timestamp,
            session_id=self._session_id,
            # Posture defaults
            posture_state="unknown",
            posture_raw_prob=0.0,
            posture_smoothed_prob=0.0,
            posture_confident=False,
            # Phone
            phone_detected=phone_detected,
            phone_confidence=phone_confidence,
        )

        # Posture state
        if posture:
            snapshot.posture_state = posture.state.value
            snapshot.posture_raw_prob = posture.raw_prob
            snapshot.posture_smoothed_prob = posture.smoothed_prob
            snapshot.posture_confident = posture.is_confident

        # Posture features
        if features:
            snapshot.torso_pitch = features.torso_pitch
            snapshot.head_forward_ratio = features.head_forward_ratio
            snapshot.shoulder_roll = features.shoulder_roll
            snapshot.lateral_lean = features.lateral_lean
            snapshot.head_tilt = features.head_tilt
            snapshot.forward_lean_z = features.forward_lean_z

        # Gaze
        if gaze:
            snapshot.gaze_pitch = gaze.head_pose[0]
            snapshot.gaze_yaw = gaze.head_pose[1]
            snapshot.gaze_roll = gaze.head_pose[2]
            snapshot.attention_state = gaze.attention_state.value
            snapshot.face_detected = gaze.face_detected

        # Presence
        if presence:
            snapshot.presence_state = presence.state.value
            snapshot.presence_confidence = presence.confidence

        # Focus
        if focus:
            snapshot.focus_state = focus.state.value
            snapshot.focus_confidence = focus.confidence
            snapshot.focus_factors = focus.contributing_factors

        return snapshot

    def _check_state_transitions(self, snapshot: StateSnapshot, now: float) -> None:
        """Check for state changes and emit events."""

        # Posture state transition
        if self._last_posture_state is not None and snapshot.posture_state != self._last_posture_state:
            duration = now - self._state_start_times.get('posture', now)
            self._emit_event(
                event_type="posture_change",
                from_state=self._last_posture_state,
                to_state=snapshot.posture_state,
                duration=duration,
                timestamp=now,
            )
        if snapshot.posture_state != self._last_posture_state:
            self._state_start_times['posture'] = now
        self._last_posture_state = snapshot.posture_state

        # Focus state transition
        if self._last_focus_state is not None and snapshot.focus_state != self._last_focus_state:
            duration = now - self._state_start_times.get('focus', now)
            self._emit_event(
                event_type="focus_change",
                from_state=self._last_focus_state,
                to_state=snapshot.focus_state,
                duration=duration,
                timestamp=now,
            )
        if snapshot.focus_state != self._last_focus_state:
            self._state_start_times['focus'] = now
        self._last_focus_state = snapshot.focus_state

        # Presence state transition
        if self._last_presence_state is not None and snapshot.presence_state != self._last_presence_state:
            duration = now - self._state_start_times.get('presence', now)
            self._emit_event(
                event_type="presence_change",
                from_state=self._last_presence_state,
                to_state=snapshot.presence_state,
                duration=duration,
                timestamp=now,
            )
        if snapshot.presence_state != self._last_presence_state:
            self._state_start_times['presence'] = now
        self._last_presence_state = snapshot.presence_state

    def _emit_event(
        self,
        event_type: str,
        from_state: str,
        to_state: str,
        duration: float,
        timestamp: float,
    ) -> None:
        """Create and store a state event."""
        event = StateEvent(
            timestamp=timestamp,
            session_id=self._session_id,
            event_type=event_type,
            from_state=from_state,
            to_state=to_state,
            duration_in_previous_state=duration,
        )
        self._events.append(event)

        # Write to file if enabled
        if self._log_file:
            self._log_file.write(json.dumps({
                "type": "event",
                "data": event.to_dict()
            }, default=_json_default) + "\n")

    def _compute_session_summary(self) -> Dict[str, Any]:
        """Compute summary statistics for the session."""
        if not self._snapshots:
            return {"session_id": self._session_id, "duration_seconds": 0}

        start_time = self._snapshots[0].timestamp
        end_time = self._snapshots[-1].timestamp
        duration = end_time - start_time

        # Count time in each state
        posture_time = {"good": 0.0, "bad": 0.0, "unknown": 0.0}
        focus_time = {"focused": 0.0, "distracted": 0.0, "away": 0.0}
        presence_time = {"seated": 0.0, "standing": 0.0, "away": 0.0}

        for i, snap in enumerate(self._snapshots[:-1]):
            dt = self._snapshots[i + 1].timestamp - snap.timestamp
            if snap.posture_state in posture_time:
                posture_time[snap.posture_state] += dt
            if snap.focus_state in focus_time:
                focus_time[snap.focus_state] += dt
            if snap.presence_state in presence_time:
                presence_time[snap.presence_state] += dt

        return {
            "session_id": self._session_id,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": duration,
            "snapshot_count": len(self._snapshots),
            "event_count": len(self._events),
            "posture_time_seconds": posture_time,
            "focus_time_seconds": focus_time,
            "presence_time_seconds": presence_time,
            "posture_good_ratio": posture_time["good"] / duration if duration > 0 else 0,
            "focus_focused_ratio": focus_time["focused"] / duration if duration > 0 else 0,
        }

    def get_history(self) -> 'StateHistory':
        """Get the StateHistory interface for queries."""
        return self._history

    def get_snapshots(self, limit: Optional[int] = None) -> List[StateSnapshot]:
        """Get recent snapshots."""
        with self._lock:
            if limit:
                return self._snapshots[-limit:]
            return self._snapshots.copy()

    def get_events(self, limit: Optional[int] = None) -> List[StateEvent]:
        """Get recent events."""
        with self._lock:
            if limit:
                return self._events[-limit:]
            return self._events.copy()

    def get_current(self) -> Optional[StateSnapshot]:
        """Get the most recent snapshot."""
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def is_active(self) -> bool:
        return self._session_id is not None
