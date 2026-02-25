"""
Rule-based presence and standing detection.

Detects:
- SEATED: Person at desk in sitting position (normal operation)
- STANDING: Person stood up (hip position significantly higher)
- AWAY: Person not detected or moved away from desk

This runs alongside posture detection without interfering.
Posture (good/bad) is only evaluated when presence is SEATED.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
from collections import deque
import numpy as np

from .pose_estimator import PoseKeypoints


class PresenceState(Enum):
    """Presence/position state."""
    SEATED = "seated"       # Normal sitting at desk
    STANDING = "standing"   # Stood up from desk
    AWAY = "away"           # Not detected / left desk


@dataclass
class PresenceResult:
    """Result from presence detection."""
    state: PresenceState
    confidence: float
    hip_y_ratio: Optional[float]  # Current hip Y / calibrated hip Y (< 1 = higher = standing)
    visibility: float             # Average visibility of key landmarks
    reason: str                   # Why this state was determined


@dataclass
class PresenceCalibration:
    """Calibrated reference values for seated position."""
    hip_y_mean: float       # Mean Y position of hip midpoint when seated
    hip_y_std: float        # Std dev of hip Y
    bbox_height_mean: float # Mean bbox height when seated
    bbox_height_std: float  # Std dev of bbox height
    n_samples: int


class PresenceDetector:
    """
    Rule-based presence detection.

    Uses calibrated seated position to detect if person:
    - Is seated normally (SEATED)
    - Has stood up (STANDING) - hips significantly higher
    - Has left (AWAY) - not detected or very low visibility

    Does NOT interfere with posture detection:
    - Posture classifier runs regardless
    - But UI/logic can skip posture alerts when not SEATED
    """

    # MediaPipe landmark indices
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    def __init__(
        self,
        standing_threshold: float = 0.7,    # Hip at 70% of calibrated Y = standing
        away_visibility_threshold: float = 0.3,  # Below this = away
        smoothing_window: int = 10,
    ):
        """
        Args:
            standing_threshold: If hip_y / calibrated_hip_y < this, person is standing
            away_visibility_threshold: If avg visibility < this, person is away
            smoothing_window: Frames to smooth state over
        """
        self.standing_threshold = standing_threshold
        self.away_visibility_threshold = away_visibility_threshold
        self.smoothing_window = smoothing_window

        self._calibration: Optional[PresenceCalibration] = None
        self._calibration_samples: list = []
        self._state_history: deque = deque(maxlen=smoothing_window)
        self._current_state = PresenceState.AWAY

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def start_calibration(self) -> None:
        """Start collecting calibration samples."""
        self._calibration_samples = []

    def add_calibration_sample(self, kp: PoseKeypoints) -> None:
        """Add a pose sample during seated calibration."""
        if kp.avg_visibility < 0.5:
            return

        lm = kp.landmarks
        hip_y = (lm[self.LEFT_HIP, 1] + lm[self.RIGHT_HIP, 1]) / 2.0

        # Approximate bbox height from shoulder to hip
        shoulder_y = (lm[self.LEFT_SHOULDER, 1] + lm[self.RIGHT_SHOULDER, 1]) / 2.0
        bbox_height = abs(hip_y - shoulder_y) * 2.5  # Rough estimate

        self._calibration_samples.append({
            "hip_y": hip_y,
            "bbox_height": bbox_height,
        })

    def finish_calibration(self) -> Optional[PresenceCalibration]:
        """Finish calibration and compute reference values."""
        if len(self._calibration_samples) < 10:
            return None

        hip_ys = [s["hip_y"] for s in self._calibration_samples]
        bbox_heights = [s["bbox_height"] for s in self._calibration_samples]

        self._calibration = PresenceCalibration(
            hip_y_mean=float(np.mean(hip_ys)),
            hip_y_std=float(np.std(hip_ys)) + 1e-6,
            bbox_height_mean=float(np.mean(bbox_heights)),
            bbox_height_std=float(np.std(bbox_heights)) + 1e-6,
            n_samples=len(self._calibration_samples),
        )

        self._calibration_samples = []
        return self._calibration

    def set_calibration(self, cal: PresenceCalibration) -> None:
        """Set calibration directly (e.g., loaded from file)."""
        self._calibration = cal

    @property
    def is_calibrated(self) -> bool:
        return self._calibration is not None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        kp: Optional[PoseKeypoints],
        person_detected: bool = True,
    ) -> PresenceResult:
        """
        Detect presence state.

        Args:
            kp: Pose keypoints (None if no pose detected)
            person_detected: Whether a person bbox was detected at all

        Returns:
            PresenceResult with state and confidence
        """
        # No person detected at all
        if not person_detected or kp is None:
            result = PresenceResult(
                state=PresenceState.AWAY,
                confidence=0.9,
                hip_y_ratio=None,
                visibility=0.0,
                reason="no_person_detected",
            )
            self._update_state(result.state)
            return result

        # Low visibility = possibly away or occluded
        if kp.avg_visibility < self.away_visibility_threshold:
            result = PresenceResult(
                state=PresenceState.AWAY,
                confidence=0.7,
                hip_y_ratio=None,
                visibility=kp.avg_visibility,
                reason="low_visibility",
            )
            self._update_state(result.state)
            return result

        # Compute hip position
        lm = kp.landmarks
        hip_y = (lm[self.LEFT_HIP, 1] + lm[self.RIGHT_HIP, 1]) / 2.0

        # Without calibration, assume seated
        if not self.is_calibrated:
            result = PresenceResult(
                state=PresenceState.SEATED,
                confidence=0.5,
                hip_y_ratio=1.0,
                visibility=kp.avg_visibility,
                reason="no_calibration_assuming_seated",
            )
            self._update_state(result.state)
            return result

        # Compare to calibrated position
        hip_y_ratio = hip_y / self._calibration.hip_y_mean

        # Standing: hips significantly higher (lower Y value in image coords)
        if hip_y_ratio < self.standing_threshold:
            result = PresenceResult(
                state=PresenceState.STANDING,
                confidence=0.8,
                hip_y_ratio=hip_y_ratio,
                visibility=kp.avg_visibility,
                reason=f"hip_y_ratio={hip_y_ratio:.2f}<{self.standing_threshold}",
            )
            self._update_state(result.state)
            return result

        # Normal seated position
        result = PresenceResult(
            state=PresenceState.SEATED,
            confidence=0.85,
            hip_y_ratio=hip_y_ratio,
            visibility=kp.avg_visibility,
            reason="normal_seated_position",
        )
        self._update_state(result.state)
        return result

    def _update_state(self, state: PresenceState) -> None:
        """Update state history for smoothing."""
        self._state_history.append(state)

        # Simple majority voting for smoothing
        if len(self._state_history) >= 3:
            recent = list(self._state_history)[-5:]
            state_counts = {}
            for s in recent:
                state_counts[s] = state_counts.get(s, 0) + 1
            self._current_state = max(state_counts, key=state_counts.get)

    @property
    def smoothed_state(self) -> PresenceState:
        """Get temporally smoothed state."""
        return self._current_state

    def reset(self) -> None:
        """Reset state history."""
        self._state_history.clear()
        self._current_state = PresenceState.AWAY
