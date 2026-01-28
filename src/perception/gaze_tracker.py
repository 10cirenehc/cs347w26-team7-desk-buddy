"""
Gaze and head pose tracking using MediaPipe Face Landmarker.

Estimates head pose (pitch, yaw, roll) and classifies attention state.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List
import yaml
from pathlib import Path
import urllib.request


class AttentionState(Enum):
    """Attention classification states based on gaze direction."""
    FOCUSED = "focused"            # looking at screen
    LOOKING_AWAY = "looking_away"  # significant yaw (left/right)
    LOOKING_DOWN = "looking_down"  # may indicate phone use


@dataclass
class GazeResult:
    """Result from gaze tracking."""
    head_pose: Tuple[float, float, float]  # (pitch, yaw, roll) in degrees
    attention_state: AttentionState
    face_detected: bool
    landmarks: Optional[List]  # for visualization


class GazeTracker:
    """
    Gaze and head pose tracker using MediaPipe Face Landmarker.

    Uses 478 facial landmarks to estimate 3D head pose via solvePnP,
    then classifies attention state based on head orientation.
    """

    # Key landmark indices for head pose estimation
    # Using canonical face model points
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    # 3D model points for a canonical face (in arbitrary units)
    # These correspond to the landmarks above
    FACE_3D_MODEL = np.array([
        [0.0, 0.0, 0.0],           # Nose tip
        [0.0, -330.0, -65.0],      # Chin
        [-225.0, 170.0, -135.0],   # Left eye outer
        [225.0, 170.0, -135.0],    # Right eye outer
        [-150.0, -150.0, -125.0],  # Left mouth
        [150.0, -150.0, -125.0],   # Right mouth
    ], dtype=np.float64)

    # Face mesh connections for drawing (simplified)
    FACE_OVAL = [
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
        (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
        (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
        (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
        (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
        (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10)
    ]

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        config_path: Optional[str] = None
    ):
        """
        Initialize gaze tracker.

        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            config_path: Optional path to config file
        """
        # Load config if provided
        if config_path:
            config = self._load_config(config_path)
            gaze_config = config.get('gaze', {})
            min_detection_confidence = gaze_config.get(
                'min_detection_confidence', min_detection_confidence
            )
            min_tracking_confidence = gaze_config.get(
                'min_tracking_confidence', min_tracking_confidence
            )
            self.yaw_threshold = gaze_config.get('yaw_threshold_deg', 30.0)
            self.pitch_down_threshold = gaze_config.get('pitch_down_threshold_deg', 20.0)
            self.pitch_up_threshold = gaze_config.get('pitch_up_threshold_deg', 30.0)
        else:
            self.yaw_threshold = 30.0
            self.pitch_down_threshold = 20.0
            self.pitch_up_threshold = 30.0

        # Download model if needed
        self._ensure_model()

        # Initialize MediaPipe Face Landmarker
        base_options = python.BaseOptions(model_asset_path=str(self.MODEL_PATH))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # Camera matrix placeholder (will be computed based on frame size)
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def _ensure_model(self):
        """Download the face landmarker model if not present."""
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not self.MODEL_PATH.exists():
            print(f"Downloading face landmarker model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print(f"Model downloaded to {self.MODEL_PATH}")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def _get_camera_matrix(self, frame_width: int, frame_height: int) -> np.ndarray:
        """
        Get or create camera matrix for given frame size.

        Uses a reasonable approximation for webcam intrinsics.
        """
        if self._camera_matrix is None or \
           self._camera_matrix[0, 2] != frame_width / 2:
            # Approximate focal length based on frame dimensions
            focal_length = frame_width
            center = (frame_width / 2, frame_height / 2)

            self._camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

        return self._camera_matrix

    def detect(self, frame: np.ndarray) -> GazeResult:
        """
        Detect face and estimate head pose.

        Args:
            frame: RGB image as numpy array

        Returns:
            GazeResult with head pose, attention state, and landmarks
        """
        frame_height, frame_width = frame.shape[:2]

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Process frame with MediaPipe
        results = self.landmarker.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return GazeResult(
                head_pose=(0.0, 0.0, 0.0),
                attention_state=AttentionState.LOOKING_AWAY,
                face_detected=False,
                landmarks=None
            )

        # Get first face landmarks
        face_landmarks = results.face_landmarks[0]

        # Estimate head pose
        head_pose = self._estimate_head_pose(
            face_landmarks, frame_width, frame_height
        )

        # Classify attention state
        attention_state = self._classify_attention(head_pose)

        return GazeResult(
            head_pose=head_pose,
            attention_state=attention_state,
            face_detected=True,
            landmarks=face_landmarks
        )

    def _estimate_head_pose(
        self,
        face_landmarks,
        frame_width: int,
        frame_height: int
    ) -> Tuple[float, float, float]:
        """
        Estimate head pose using solvePnP.

        Args:
            face_landmarks: MediaPipe face landmarks
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        # Extract 2D landmark positions
        landmark_indices = [
            self.NOSE_TIP,
            self.CHIN,
            self.LEFT_EYE_OUTER,
            self.RIGHT_EYE_OUTER,
            self.LEFT_MOUTH,
            self.RIGHT_MOUTH
        ]

        face_2d = []
        for idx in landmark_indices:
            lm = face_landmarks[idx]
            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)
            face_2d.append([x, y])

        face_2d = np.array(face_2d, dtype=np.float64)

        # Get camera matrix
        camera_matrix = self._get_camera_matrix(frame_width, frame_height)

        # Solve PnP to get rotation and translation vectors
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.FACE_3D_MODEL,
            face_2d,
            camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return (0.0, 0.0, 0.0)

        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Get Euler angles from rotation matrix
        # Using the decomposition that gives pitch, yaw, roll
        sy = np.sqrt(rotation_mat[0, 0]**2 + rotation_mat[1, 0]**2)

        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        else:
            pitch = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = 0

        # Convert to degrees
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        roll_deg = np.degrees(roll)

        return (pitch_deg, yaw_deg, roll_deg)

    def _classify_attention(
        self,
        head_pose: Tuple[float, float, float]
    ) -> AttentionState:
        """
        Classify attention state based on head pose.

        Args:
            head_pose: Tuple of (pitch, yaw, roll) in degrees

        Returns:
            AttentionState classification
        """
        pitch, yaw, roll = head_pose

        # Check yaw (looking left/right)
        if abs(yaw) > self.yaw_threshold:
            return AttentionState.LOOKING_AWAY

        # Check pitch (looking down - potential phone use)
        if pitch > self.pitch_down_threshold:
            return AttentionState.LOOKING_DOWN

        # Otherwise, assume focused
        return AttentionState.FOCUSED

    def draw_landmarks(
        self,
        frame: np.ndarray,
        result: GazeResult,
        draw_contours: bool = True
    ) -> np.ndarray:
        """
        Draw face mesh landmarks on frame.

        Args:
            frame: BGR image to draw on
            result: GazeResult from detect()
            draw_contours: Whether to draw face contours

        Returns:
            Frame with landmarks drawn
        """
        if result.landmarks is None:
            return frame

        annotated_frame = frame.copy()
        h, w = frame.shape[:2]

        # Draw face oval if enabled
        if draw_contours:
            for start_idx, end_idx in self.FACE_OVAL:
                if start_idx < len(result.landmarks) and end_idx < len(result.landmarks):
                    start = result.landmarks[start_idx]
                    end = result.landmarks[end_idx]
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 1)

        # Draw key landmarks
        key_indices = [self.NOSE_TIP, self.LEFT_EYE_OUTER, self.RIGHT_EYE_OUTER,
                       self.LEFT_MOUTH, self.RIGHT_MOUTH, self.CHIN]
        for idx in key_indices:
            if idx < len(result.landmarks):
                lm = result.landmarks[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(annotated_frame, (x, y), 3, (0, 0, 255), -1)

        return annotated_frame

    def draw_head_pose_axes(
        self,
        frame: np.ndarray,
        result: GazeResult
    ) -> np.ndarray:
        """
        Draw head pose axes on frame.

        Args:
            frame: BGR image to draw on
            result: GazeResult from detect()

        Returns:
            Frame with pose axes drawn
        """
        if not result.face_detected or result.landmarks is None:
            return frame

        annotated_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        # Get nose position as origin
        nose = result.landmarks[self.NOSE_TIP]
        nose_x = int(nose.x * frame_width)
        nose_y = int(nose.y * frame_height)

        # Draw axes based on head pose
        pitch, yaw, roll = result.head_pose
        axis_length = 100

        # Simplified axis drawing based on angles
        # X axis (red) - affected by yaw and roll
        x_end_x = int(nose_x + axis_length * np.cos(np.radians(yaw)))
        x_end_y = int(nose_y + axis_length * np.sin(np.radians(roll)))
        cv2.line(annotated_frame, (nose_x, nose_y), (x_end_x, x_end_y), (0, 0, 255), 3)

        # Y axis (green) - affected by pitch and roll
        y_end_x = int(nose_x - axis_length * np.sin(np.radians(roll)))
        y_end_y = int(nose_y - axis_length * np.cos(np.radians(pitch)))
        cv2.line(annotated_frame, (nose_x, nose_y), (y_end_x, y_end_y), (0, 255, 0), 3)

        # Z axis (blue) - pointing out of face
        z_end_x = int(nose_x + axis_length * np.sin(np.radians(yaw)) * 0.5)
        z_end_y = int(nose_y + axis_length * np.sin(np.radians(pitch)) * 0.5)
        cv2.line(annotated_frame, (nose_x, nose_y), (z_end_x, z_end_y), (255, 0, 0), 3)

        return annotated_frame

    def draw_metrics_overlay(
        self,
        frame: np.ndarray,
        result: GazeResult,
        y_offset: int = 30
    ) -> np.ndarray:
        """
        Draw gaze metrics overlay on frame.

        Args:
            frame: BGR image to draw on
            result: GazeResult from detect()
            y_offset: Vertical offset for text

        Returns:
            Frame with metrics overlay
        """
        annotated_frame = frame.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Color based on attention state
        color_map = {
            AttentionState.FOCUSED: (0, 255, 0),       # Green
            AttentionState.LOOKING_AWAY: (0, 0, 255),  # Red
            AttentionState.LOOKING_DOWN: (0, 165, 255) # Orange
        }
        state_color = color_map.get(result.attention_state, (255, 255, 255))

        # Draw attention state
        cv2.putText(
            annotated_frame,
            f"Attention: {result.attention_state.value.upper()}",
            (10, y_offset),
            font, font_scale, state_color, thickness
        )

        if result.face_detected:
            pitch, yaw, roll = result.head_pose

            y_offset += 25
            cv2.putText(
                annotated_frame,
                f"Pitch: {pitch:.1f} deg",
                (10, y_offset),
                font, 0.5, (255, 255, 255), 1
            )

            y_offset += 20
            cv2.putText(
                annotated_frame,
                f"Yaw: {yaw:.1f} deg",
                (10, y_offset),
                font, 0.5, (255, 255, 255), 1
            )

            y_offset += 20
            cv2.putText(
                annotated_frame,
                f"Roll: {roll:.1f} deg",
                (10, y_offset),
                font, 0.5, (255, 255, 255), 1
            )
        else:
            y_offset += 25
            cv2.putText(
                annotated_frame,
                "No face detected",
                (10, y_offset),
                font, 0.5, (128, 128, 128), 1
            )

        return annotated_frame

    def close(self) -> None:
        """Release resources."""
        self.landmarker.close()
