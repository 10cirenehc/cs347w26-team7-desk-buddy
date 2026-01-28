"""
Focus state estimator that fuses signals from all perception modules.

Combines posture, phone detection, and gaze tracking to determine overall focus state.
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import yaml
from pathlib import Path
from datetime import datetime

from .posture_detector import PostureResult, PostureState
from .phone_detector import PhoneDetectionResult
from .gaze_tracker import GazeResult, AttentionState


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
        """
        Initialize focus estimator.

        Args:
            smoothing_window: Number of frames for rolling window
            state_change_threshold: Min frames before state change
            distraction_timeout: Frames before marking as distracted
            config_path: Optional path to config file
        """
        # Load config if provided
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

        # State history for temporal smoothing
        self._state_history: deque = deque(maxlen=smoothing_window)

        # Current smoothed state
        self._current_state = FocusState.FOCUSED
        self._frames_in_state = 0

        # Event logging
        self._events: List[dict] = []

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def estimate(
        self,
        posture: Optional[PostureResult] = None,
        phone: Optional[PhoneDetectionResult] = None,
        gaze: Optional[GazeResult] = None
    ) -> FocusEstimation:
        """
        Estimate focus state from detection results.

        Args:
            posture: PostureResult from posture detector
            phone: PhoneDetectionResult from phone detector
            gaze: GazeResult from gaze tracker

        Returns:
            FocusEstimation with state, confidence, and factors
        """
        # Compute raw state
        raw_state, confidence, factors = self._compute_raw_state(
            posture, phone, gaze
        )

        # Add to history
        self._state_history.append(raw_state)

        # Apply temporal smoothing
        smoothed_state = self._apply_smoothing(raw_state)

        # Update state duration
        if smoothed_state == self._current_state:
            self._frames_in_state += 1
        else:
            # Log state transition
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
        posture: Optional[PostureResult],
        phone: Optional[PhoneDetectionResult],
        gaze: Optional[GazeResult]
    ) -> tuple:
        """
        Compute raw focus state from current detections.

        Priority rules:
        1. No face detected -> AWAY
        2. Phone in hand -> DISTRACTED
        3. Looking away/down for extended period -> DISTRACTED
        4. Otherwise -> FOCUSED

        Returns:
            Tuple of (FocusState, confidence, factors)
        """
        factors = []
        confidence_scores = []

        # Check if person is present (face detected)
        if gaze is not None:
            if not gaze.face_detected:
                factors.append("no_face_detected")
                return FocusState.AWAY, 0.8, factors

        # Check phone usage (highest priority distraction)
        if phone is not None:
            if phone.phone_detected and phone.in_hand:
                factors.append("phone_in_hand")
                confidence_scores.append(phone.confidence)
                return FocusState.DISTRACTED, np.mean(confidence_scores), factors

        # Check gaze/attention
        if gaze is not None and gaze.face_detected:
            if gaze.attention_state == AttentionState.LOOKING_AWAY:
                factors.append("looking_away")
                confidence_scores.append(0.7)
            elif gaze.attention_state == AttentionState.LOOKING_DOWN:
                factors.append("looking_down")
                confidence_scores.append(0.6)
                # Looking down might indicate phone use even without detection
                if phone is not None and phone.phone_detected:
                    factors.append("phone_visible_while_looking_down")
                    return FocusState.DISTRACTED, 0.75, factors

        # Check posture (secondary indicator)
        if posture is not None and posture.pose_detected:
            if posture.state != PostureState.GOOD:
                factors.append(f"posture_{posture.state.value}")
                confidence_scores.append(posture.confidence * 0.5)  # weight lower

        # Determine state based on factors
        if "looking_away" in factors:
            return FocusState.DISTRACTED, 0.7, factors
        elif "looking_down" in factors and len(factors) > 1:
            return FocusState.DISTRACTED, 0.6, factors

        # Default to focused
        if not factors:
            factors.append("all_clear")
        confidence = np.mean(confidence_scores) if confidence_scores else 0.8

        return FocusState.FOCUSED, confidence, factors

    def _apply_smoothing(self, raw_state: FocusState) -> FocusState:
        """
        Apply temporal smoothing to prevent state flickering.

        Uses a voting mechanism over recent history.

        Args:
            raw_state: Current raw state

        Returns:
            Smoothed state
        """
        if len(self._state_history) < self.state_change_threshold:
            return raw_state

        # Count states in recent window
        recent = list(self._state_history)[-self.state_change_threshold:]
        state_counts = {}
        for state in recent:
            state_counts[state] = state_counts.get(state, 0) + 1

        # Find most common state
        most_common = max(state_counts, key=state_counts.get)
        most_common_count = state_counts[most_common]

        # Only change state if new state dominates
        threshold_ratio = 0.6
        if most_common_count >= self.state_change_threshold * threshold_ratio:
            return most_common

        # Otherwise, maintain current state
        return self._current_state

    def _log_event(self, event_type: str, data: dict) -> None:
        """Log an event with timestamp."""
        self._events.append({
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': data
        })

    def get_events(self, limit: Optional[int] = None) -> List[dict]:
        """Get logged events."""
        if limit:
            return self._events[-limit:]
        return self._events.copy()

    def reset(self) -> None:
        """Reset state history and counters."""
        self._state_history.clear()
        self._current_state = FocusState.FOCUSED
        self._frames_in_state = 0

    def draw_status_overlay(
        self,
        frame: np.ndarray,
        result: FocusEstimation,
        y_offset: int = 30
    ) -> np.ndarray:
        """
        Draw focus status overlay on frame.

        Args:
            frame: BGR image to draw on
            result: FocusEstimation from estimate()
            y_offset: Vertical offset for text

        Returns:
            Frame with status overlay
        """
        annotated_frame = frame.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Color based on focus state
        color_map = {
            FocusState.FOCUSED: (0, 255, 0),     # Green
            FocusState.DISTRACTED: (0, 0, 255),  # Red
            FocusState.AWAY: (128, 128, 128)     # Gray
        }
        state_color = color_map.get(result.state, (255, 255, 255))

        # Draw focus state (large, prominent)
        cv2.putText(
            annotated_frame,
            f"FOCUS: {result.state.value.upper()}",
            (10, y_offset),
            font, font_scale, state_color, thickness
        )

        # Draw contributing factors
        y_offset += 30
        factors_str = ", ".join(result.contributing_factors[:3])
        cv2.putText(
            annotated_frame,
            f"Factors: {factors_str}",
            (10, y_offset),
            font, 0.5, (255, 255, 255), 1
        )

        # Draw duration
        y_offset += 20
        duration_secs = result.duration_in_state / 30.0  # assuming 30 fps
        cv2.putText(
            annotated_frame,
            f"Duration: {duration_secs:.1f}s",
            (10, y_offset),
            font, 0.5, (255, 255, 255), 1
        )

        return annotated_frame

    def draw_full_overlay(
        self,
        frame: np.ndarray,
        focus_result: FocusEstimation,
        posture_result: Optional[PostureResult] = None,
        phone_result: Optional[PhoneDetectionResult] = None,
        gaze_result: Optional[GazeResult] = None
    ) -> np.ndarray:
        """
        Draw comprehensive status overlay with all detection results.

        Args:
            frame: BGR image to draw on
            focus_result: FocusEstimation from estimate()
            posture_result: Optional PostureResult
            phone_result: Optional PhoneDetectionResult
            gaze_result: Optional GazeResult

        Returns:
            Frame with full status overlay
        """
        annotated_frame = frame.copy()
        frame_height = frame.shape[0]

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Focus state (top)
        color_map = {
            FocusState.FOCUSED: (0, 255, 0),
            FocusState.DISTRACTED: (0, 0, 255),
            FocusState.AWAY: (128, 128, 128)
        }
        state_color = color_map.get(focus_result.state, (255, 255, 255))

        cv2.putText(
            annotated_frame,
            f"FOCUS: {focus_result.state.value.upper()}",
            (10, 30),
            font, 0.8, state_color, 2
        )

        # Posture (right side, top)
        if posture_result:
            posture_colors = {
                PostureState.GOOD: (0, 255, 0),
                PostureState.SLOUCHING: (0, 165, 255),
                PostureState.LEANING: (0, 255, 255),
                PostureState.HUNCHING: (0, 0, 255)
            }
            p_color = posture_colors.get(posture_result.state, (255, 255, 255))
            cv2.putText(
                annotated_frame,
                f"Posture: {posture_result.state.value}",
                (frame.shape[1] - 200, 30),
                font, 0.6, p_color, 2
            )

        # Phone detection (right side, middle)
        if phone_result:
            if phone_result.phone_detected:
                p_text = "PHONE IN HAND" if phone_result.in_hand else "Phone visible"
                p_color = (0, 0, 255) if phone_result.in_hand else (0, 255, 255)
            else:
                p_text = "No phone"
                p_color = (0, 255, 0)
            cv2.putText(
                annotated_frame,
                p_text,
                (frame.shape[1] - 200, 60),
                font, 0.6, p_color, 2
            )

        # Gaze/Attention (right side, bottom)
        if gaze_result:
            gaze_colors = {
                AttentionState.FOCUSED: (0, 255, 0),
                AttentionState.LOOKING_AWAY: (0, 0, 255),
                AttentionState.LOOKING_DOWN: (0, 165, 255)
            }
            g_color = gaze_colors.get(gaze_result.attention_state, (255, 255, 255))
            cv2.putText(
                annotated_frame,
                f"Gaze: {gaze_result.attention_state.value}",
                (frame.shape[1] - 200, 90),
                font, 0.6, g_color, 2
            )

        # Duration at bottom
        duration_secs = focus_result.duration_in_state / 30.0
        cv2.putText(
            annotated_frame,
            f"In state: {duration_secs:.1f}s",
            (10, frame_height - 20),
            font, 0.5, (255, 255, 255), 1
        )

        return annotated_frame
