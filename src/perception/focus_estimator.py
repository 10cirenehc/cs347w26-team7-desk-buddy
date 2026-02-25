"""
Focus state estimator that fuses signals from all perception modules.

Combines posture, phone detection, and gaze tracking to determine overall focus state.
Accepts both the new SmoothedPostureState (Pipeline 2.0) and legacy PhoneDetectionResult.
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Union
import yaml
from pathlib import Path
from datetime import datetime

from .posture_state import SmoothedPostureState, PostureLabel
from .person_detector import BBox
from .presence_detector import PresenceResult, PresenceState


class FocusState(Enum):
    """Overall focus state classification."""
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    AWAY = "away"  # not at desk


@dataclass
class FocusEstimation:
    """Result from focus estimation."""
    state: FocusState
    confidence: float
    contributing_factors: List[str]  # what led to this classification
    raw_state: FocusState  # state before temporal smoothing
    duration_in_state: int  # frames in current state


class FocusEstimator:
    """
    Focus state estimator with temporal smoothing.

    Fuses signals from posture, phone detection, and gaze tracking
    to determine overall focus state.
    """

    def __init__(
        self,
        smoothing_window: int = 15,
        state_change_threshold: int = 10,
        distraction_timeout: int = 45,
        config_path: Optional[str] = None
    ):
        if config_path:
            config = self._load_config(config_path)
            focus_config = config.get('focus', {})
            smoothing_window = focus_config.get(
                'smoothing_window', smoothing_window
            )
            state_change_threshold = focus_config.get(
                'state_change_threshold', state_change_threshold
            )
            distraction_timeout = focus_config.get(
                'distraction_timeout', distraction_timeout
            )

        self.smoothing_window = smoothing_window
        self.state_change_threshold = state_change_threshold
        self.distraction_timeout = distraction_timeout

        self._state_history: deque = deque(maxlen=smoothing_window)
        self._current_state = FocusState.FOCUSED
        self._frames_in_state = 0
        self._events: List[dict] = []

    def _load_config(self, config_path: str) -> dict:
        path = Path(config_path)
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def estimate(
        self,
        posture: Optional[SmoothedPostureState] = None,
        phone_detected_in_hand: bool = False,
        phone_confidence: float = 0.0,
        presence: Optional[PresenceResult] = None,
    ) -> FocusEstimation:
        """
        Estimate focus state from detection results.

        Args:
            posture: SmoothedPostureState from the new pipeline.
            phone_detected_in_hand: True if a phone is detected near the primary person.
            phone_confidence: Confidence of the phone detection.
            presence: PresenceResult from presence detector.

        Returns:
            FocusEstimation with state, confidence, and factors.
        """
        raw_state, confidence, factors = self._compute_raw_state(
            posture, phone_detected_in_hand, phone_confidence, presence
        )

        self._state_history.append(raw_state)
        smoothed_state = self._apply_smoothing(raw_state)

        if smoothed_state == self._current_state:
            self._frames_in_state += 1
        else:
            self._log_event('state_change', {
                'from': self._current_state.value,
                'to': smoothed_state.value,
                'duration': self._frames_in_state
            })
            self._current_state = smoothed_state
            self._frames_in_state = 1

        return FocusEstimation(
            state=smoothed_state,
            confidence=confidence,
            contributing_factors=factors,
            raw_state=raw_state,
            duration_in_state=self._frames_in_state
        )

    def _compute_raw_state(
        self,
        posture: Optional[SmoothedPostureState],
        phone_in_hand: bool,
        phone_confidence: float,
        presence: Optional[PresenceResult],
    ) -> tuple:
        """
        Compute raw focus state using presence + phone signals.

        Priority rules:
        1. Presence AWAY or no presence data -> AWAY
        2. Phone in hand -> DISTRACTED
        3. Bad posture -> contributing factor only (not sufficient for DISTRACTED)
        4. Otherwise -> FOCUSED
        """
        factors = []

        # --- AWAY: person not detected or left desk ---
        if presence is None or presence.state == PresenceState.AWAY:
            factors.append("away_from_desk")
            return FocusState.AWAY, 0.8, factors

        # --- DISTRACTED: phone in hand ---
        if phone_in_hand:
            factors.append("phone_in_hand")
            return FocusState.DISTRACTED, max(0.6, phone_confidence), factors

        # --- Posture as contributing factor (not sufficient alone) ---
        if posture is not None and posture.state == PostureLabel.BAD:
            factors.append("bad_posture")

        if not factors:
            factors.append("all_clear")

        return FocusState.FOCUSED, 0.8, factors

    def _apply_smoothing(self, raw_state: FocusState) -> FocusState:
        if len(self._state_history) < self.state_change_threshold:
            return raw_state

        recent = list(self._state_history)[-self.state_change_threshold:]
        state_counts: dict = {}
        for state in recent:
            state_counts[state] = state_counts.get(state, 0) + 1

        most_common = max(state_counts, key=state_counts.get)
        most_common_count = state_counts[most_common]

        threshold_ratio = 0.6
        if most_common_count >= self.state_change_threshold * threshold_ratio:
            return most_common

        return self._current_state

    def _log_event(self, event_type: str, data: dict) -> None:
        self._events.append({
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': data
        })

    def get_events(self, limit: Optional[int] = None) -> List[dict]:
        if limit:
            return self._events[-limit:]
        return self._events.copy()

    def reset(self) -> None:
        self._state_history.clear()
        self._current_state = FocusState.FOCUSED
        self._frames_in_state = 0

    def draw_status_overlay(
        self,
        frame: np.ndarray,
        result: FocusEstimation,
        y_offset: int = 30
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        color_map = {
            FocusState.FOCUSED: (0, 255, 0),
            FocusState.DISTRACTED: (0, 0, 255),
            FocusState.AWAY: (128, 128, 128)
        }
        state_color = color_map.get(result.state, (255, 255, 255))

        cv2.putText(annotated_frame, f"FOCUS: {result.state.value.upper()}",
                     (10, y_offset), font, 0.7, state_color, 2)

        y_offset += 30
        factors_str = ", ".join(result.contributing_factors[:3])
        cv2.putText(annotated_frame, f"Factors: {factors_str}",
                     (10, y_offset), font, 0.5, (255, 255, 255), 1)

        y_offset += 20
        duration_secs = result.duration_in_state / 30.0
        cv2.putText(annotated_frame, f"Duration: {duration_secs:.1f}s",
                     (10, y_offset), font, 0.5, (255, 255, 255), 1)

        return annotated_frame
