"""
Posture detection using MediaPipe Pose Landmarker.

Extracts body landmarks and computes geometric features for posture classification.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
import yaml
from pathlib import Path
import math
import urllib.request
import os


class PostureState(Enum):
    """Posture classification states."""
    GOOD = "good"
    SLOUCHING = "slouching"  # forward lean
    LEANING = "leaning"      # lateral tilt
    HUNCHING = "hunching"    # rounded shoulders/forward head


@dataclass
class PostureResult:
    """Result from posture detection."""
    state: PostureState
    confidence: float
    metrics: Dict[str, float]  # raw computed angles/distances
    landmarks: Optional[List[Any]]  # for visualization
    pose_detected: bool


class PostureDetector:
    """
    Posture detector using MediaPipe Pose Landmarker.

    Uses 33 body landmarks to compute geometric features for posture classification:
    - Shoulder angle (lateral slouching)
    - Forward head posture (forward lean)
    - Torso inclination (hunching)
    """

    # MediaPipe Pose landmark indices
    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    # Pose connections for drawing
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
    ]

    # Use "full" model for better accuracy (lite was less reliable)
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_full.task"

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        config_path: Optional[str] = None
    ):
        """
        Initialize posture detector.

        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            config_path: Optional path to config file
        """
        # Load config if provided
        if config_path:
            config = self._load_config(config_path)
            posture_config = config.get('posture', {})
            min_detection_confidence = posture_config.get(
                'min_detection_confidence', min_detection_confidence
            )
            min_tracking_confidence = posture_config.get(
                'min_tracking_confidence', min_tracking_confidence
            )
            self.forward_head_threshold = posture_config.get(
                'forward_head_threshold', 0.06
            )
            self.shoulder_tilt_threshold = posture_config.get(
                'shoulder_tilt_threshold_deg', 12.0
            )
            self.torso_angle_threshold = posture_config.get(
                'torso_angle_threshold_deg', 15.0
            )
        else:
            self.forward_head_threshold = 0.06
            self.shoulder_tilt_threshold = 12.0
            self.torso_angle_threshold = 15.0

        # Smoothing buffer for metrics (reduces noise)
        self._smoothing_window = 2  # reduced for faster response
        self._metric_history: List[Dict[str, float]] = []

        # Download model if needed
        self._ensure_model()

        # Initialize MediaPipe Pose Landmarker
        base_options = python.BaseOptions(model_asset_path=str(self.MODEL_PATH))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def _ensure_model(self):
        """Download the pose landmarker model if not present."""
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not self.MODEL_PATH.exists():
            print(f"Downloading pose landmarker model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print(f"Model downloaded to {self.MODEL_PATH}")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def detect(self, frame: np.ndarray) -> PostureResult:
        """
        Detect posture in a frame.

        Args:
            frame: RGB image as numpy array

        Returns:
            PostureResult with state, confidence, metrics, and landmarks
        """
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Process frame with MediaPipe
        results = self.landmarker.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return PostureResult(
                state=PostureState.GOOD,
                confidence=0.0,
                metrics={},
                landmarks=None,
                pose_detected=False
            )

        # Extract landmarks (first pose)
        landmarks = results.pose_landmarks[0]

        # Compute posture metrics
        raw_metrics = self._compute_metrics(landmarks)

        # Apply smoothing to reduce noise
        metrics = self._smooth_metrics(raw_metrics)

        # Classify posture based on smoothed metrics
        state, confidence = self._classify_posture(metrics)

        return PostureResult(
            state=state,
            confidence=confidence,
            metrics=metrics,
            landmarks=landmarks,
            pose_detected=True
        )

    def _smooth_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Apply temporal smoothing to metrics to reduce noise.

        Args:
            metrics: Raw metrics from current frame

        Returns:
            Smoothed metrics averaged over recent frames
        """
        # Add to history
        self._metric_history.append(metrics)

        # Keep only recent frames
        if len(self._metric_history) > self._smoothing_window:
            self._metric_history.pop(0)

        # Average the metrics
        if len(self._metric_history) == 0:
            return metrics

        smoothed = {}
        keys = ['shoulder_tilt_deg', 'forward_head_distance', 'torso_angle_deg',
                'nose_shoulder_offset', 'avg_visibility', 'head_tilt_deg',
                'ear_shoulder_horizontal', 'hip_visibility', 'shoulder_visibility',
                'ear_visibility']

        for key in keys:
            values = [m.get(key, 0) for m in self._metric_history]
            smoothed[key] = sum(values) / len(values)

        return smoothed

    def _compute_metrics(self, landmarks) -> Dict[str, float]:
        """
        Compute geometric metrics from pose landmarks.

        Args:
            landmarks: MediaPipe pose landmarks

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # Get key landmark positions
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        left_ear = landmarks[self.LEFT_EAR]
        right_ear = landmarks[self.RIGHT_EAR]
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]
        nose = landmarks[self.NOSE]

        # 1. Shoulder tilt angle (lateral slouching)
        # Angle of shoulder line relative to horizontal
        # Note: In mirrored webcam view, right_shoulder.x < left_shoulder.x
        # so angle will be ~180° instead of ~0° when horizontal
        shoulder_dx = right_shoulder.x - left_shoulder.x
        shoulder_dy = right_shoulder.y - left_shoulder.y
        shoulder_angle_raw = math.degrees(math.atan2(shoulder_dy, shoulder_dx))

        # Normalize to deviation from horizontal (0° or 180°)
        # This handles both normal and mirrored camera views
        shoulder_tilt = abs(shoulder_angle_raw)
        if shoulder_tilt > 90:
            shoulder_tilt = 180 - shoulder_tilt
        metrics['shoulder_tilt_deg'] = shoulder_tilt

        # 2. Forward head posture
        # Horizontal distance from ear midpoint to shoulder midpoint
        # (normalized by shoulder width for scale invariance)
        shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_midpoint_y = (left_shoulder.y + right_shoulder.y) / 2
        ear_midpoint_x = (left_ear.x + right_ear.x) / 2
        ear_midpoint_y = (left_ear.y + right_ear.y) / 2

        shoulder_width = math.sqrt(shoulder_dx**2 + shoulder_dy**2)
        if shoulder_width > 0:
            # Forward distance (positive means head is forward of shoulders)
            forward_distance = (ear_midpoint_y - shoulder_midpoint_y)
            metrics['forward_head_distance'] = forward_distance
        else:
            metrics['forward_head_distance'] = 0.0

        # 3. Torso inclination (hunching/leaning forward)
        # Angle of line from hip midpoint to shoulder midpoint from vertical
        hip_midpoint_x = (left_hip.x + right_hip.x) / 2
        hip_midpoint_y = (left_hip.y + right_hip.y) / 2

        torso_dx = shoulder_midpoint_x - hip_midpoint_x
        torso_dy = shoulder_midpoint_y - hip_midpoint_y

        # Angle from vertical (0 = upright, positive = leaning forward)
        # In image coordinates, y increases downward, so we compute angle from vertical
        torso_angle = math.degrees(math.atan2(torso_dx, -torso_dy))
        metrics['torso_angle_deg'] = torso_angle

        # 4. Nose to shoulder relationship (works well for upper-body view)
        nose_shoulder_vertical = nose.y - shoulder_midpoint_y
        metrics['nose_shoulder_offset'] = nose_shoulder_vertical

        # 5. Ear-to-shoulder horizontal offset (forward head posture)
        # When head moves forward, ears move forward relative to shoulders
        ear_shoulder_horizontal = abs(ear_midpoint_x - shoulder_midpoint_x)
        metrics['ear_shoulder_horizontal'] = ear_shoulder_horizontal

        # 6. Head tilt (using ear positions)
        ear_dx = right_ear.x - left_ear.x
        ear_dy = right_ear.y - left_ear.y
        ear_angle_raw = math.degrees(math.atan2(ear_dy, ear_dx))
        head_tilt = abs(ear_angle_raw)
        if head_tilt > 90:
            head_tilt = 180 - head_tilt
        metrics['head_tilt_deg'] = head_tilt

        # 7. Visibility scores for confidence estimation
        metrics['shoulder_visibility'] = (left_shoulder.visibility + right_shoulder.visibility) / 2
        metrics['ear_visibility'] = (left_ear.visibility + right_ear.visibility) / 2
        metrics['hip_visibility'] = (left_hip.visibility + right_hip.visibility) / 2

        visibility_scores = [
            left_shoulder.visibility,
            right_shoulder.visibility,
            left_ear.visibility,
            right_ear.visibility,
        ]
        # Only include hips if they're actually visible
        if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
            visibility_scores.extend([left_hip.visibility, right_hip.visibility])

        metrics['avg_visibility'] = sum(visibility_scores) / len(visibility_scores)

        return metrics

    def _classify_posture(self, metrics: Dict[str, float]) -> tuple:
        """
        Classify posture based on computed metrics.

        Args:
            metrics: Dictionary of posture metrics

        Returns:
            Tuple of (PostureState, confidence)
        """
        issues = []

        # Check shoulder tilt (lateral leaning)
        shoulder_tilt = metrics.get('shoulder_tilt_deg', 0)
        if shoulder_tilt > self.shoulder_tilt_threshold:
            issues.append(('leaning', shoulder_tilt / 30.0))

        # Also check head tilt (more reliable for desk setup)
        head_tilt = metrics.get('head_tilt_deg', 0)
        if head_tilt > self.shoulder_tilt_threshold:
            issues.append(('leaning', head_tilt / 30.0))

        # Check forward head posture
        forward_head = metrics.get('forward_head_distance', 0)
        if forward_head > self.forward_head_threshold:
            issues.append(('slouching', forward_head / 0.15))

        # Check torso angle (only if hips are visible)
        hip_visibility = metrics.get('hip_visibility', 0)
        if hip_visibility > 0.5:
            torso_angle = abs(metrics.get('torso_angle_deg', 0))
            if torso_angle > self.torso_angle_threshold:
                issues.append(('hunching', torso_angle / 30.0))

        # Determine primary issue
        if not issues:
            return PostureState.GOOD, metrics.get('avg_visibility', 0.5)

        # Sort by severity and return most severe issue
        issues.sort(key=lambda x: x[1], reverse=True)
        primary_issue, severity = issues[0]

        confidence = min(0.95, metrics.get('avg_visibility', 0.5) * min(1.0, severity))

        if primary_issue == 'leaning':
            return PostureState.LEANING, confidence
        elif primary_issue == 'slouching':
            return PostureState.SLOUCHING, confidence
        else:
            return PostureState.HUNCHING, confidence

    def draw_landmarks(
        self,
        frame: np.ndarray,
        result: PostureResult,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        Draw pose landmarks on frame.

        Args:
            frame: BGR image to draw on
            result: PostureResult from detect()
            draw_connections: Whether to draw skeleton connections

        Returns:
            Frame with landmarks drawn
        """
        if result.landmarks is None:
            return frame

        annotated_frame = frame.copy()
        h, w = frame.shape[:2]

        # Draw landmarks
        for landmark in result.landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(annotated_frame, (x, y), 3, (0, 255, 0), -1)

        # Draw connections
        if draw_connections:
            for start_idx, end_idx in self.POSE_CONNECTIONS:
                if start_idx < len(result.landmarks) and end_idx < len(result.landmarks):
                    start = result.landmarks[start_idx]
                    end = result.landmarks[end_idx]
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(annotated_frame, start_point, end_point, (0, 0, 255), 2)

        return annotated_frame

    def draw_metrics_overlay(
        self,
        frame: np.ndarray,
        result: PostureResult
    ) -> np.ndarray:
        """
        Draw posture metrics overlay on frame.

        Args:
            frame: BGR image to draw on
            result: PostureResult from detect()

        Returns:
            Frame with metrics overlay
        """
        annotated_frame = frame.copy()

        # Set up text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Color based on posture state
        color_map = {
            PostureState.GOOD: (0, 255, 0),      # Green
            PostureState.SLOUCHING: (0, 165, 255),  # Orange
            PostureState.LEANING: (0, 255, 255),    # Yellow
            PostureState.HUNCHING: (0, 0, 255)      # Red
        }
        state_color = color_map.get(result.state, (255, 255, 255))

        # Draw state indicator
        y_offset = 30
        cv2.putText(
            annotated_frame,
            f"Posture: {result.state.value.upper()}",
            (10, y_offset),
            font, font_scale, state_color, thickness
        )

        if result.pose_detected and result.metrics:
            y_offset += 25
            cv2.putText(
                annotated_frame,
                f"Shoulder tilt: {result.metrics.get('shoulder_tilt_deg', 0):.1f} deg",
                (10, y_offset),
                font, 0.5, (255, 255, 255), 1
            )

            y_offset += 20
            cv2.putText(
                annotated_frame,
                f"Head tilt: {result.metrics.get('head_tilt_deg', 0):.1f} deg",
                (10, y_offset),
                font, 0.5, (255, 255, 255), 1
            )

            y_offset += 20
            cv2.putText(
                annotated_frame,
                f"Forward head: {result.metrics.get('forward_head_distance', 0):.3f}",
                (10, y_offset),
                font, 0.5, (255, 255, 255), 1
            )

            y_offset += 20
            hip_vis = result.metrics.get('hip_visibility', 0)
            torso_str = f"{result.metrics.get('torso_angle_deg', 0):.1f}" if hip_vis > 0.5 else "N/A"
            cv2.putText(
                annotated_frame,
                f"Torso angle: {torso_str} (hips: {hip_vis:.0%})",
                (10, y_offset),
                font, 0.5, (255, 255, 255), 1
            )

            y_offset += 20
            cv2.putText(
                annotated_frame,
                f"Confidence: {result.confidence:.2f}",
                (10, y_offset),
                font, 0.5, (255, 255, 255), 1
            )

        return annotated_frame

    def close(self) -> None:
        """Release resources."""
        self.landmarker.close()
