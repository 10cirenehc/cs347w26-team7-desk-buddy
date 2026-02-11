"""
Per-person pose estimation using MediaPipe Pose Landmarker.

Crops the primary person's bounding box (with padding), runs MediaPipe
Pose on the crop, and maps landmarks back to full-frame coordinates.

Uses MediaPipe's 33 landmark format with x, y, z (relative depth), visibility.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path
import urllib.request

from .person_detector import BBox


# MediaPipe Pose landmark indices
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Upper-body landmark indices for visibility averaging
KEY_LANDMARKS = [NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]


@dataclass
class PoseKeypoints:
    """Pose estimation result for a single person."""
    landmarks: np.ndarray           # (33, 4) — x, y, z, visibility in frame coords
    bbox_in_frame: BBox             # the padded crop bbox used
    avg_visibility: float           # mean visibility across key landmarks


class PoseEstimator:
    """
    MediaPipe Pose Landmarker on a cropped primary-person bbox.

    The pipeline abstracts the pose backend behind
    ``PoseEstimator.estimate(frame, bbox) -> PoseKeypoints``.
    """

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/1/"
        "pose_landmarker_full.task"
    )
    MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_full.task"

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        padding_ratio: float = 0.3,
        config_path: Optional[str] = None,
    ):
        if config_path:
            cfg = self._load_config(config_path).get("pose_estimator", {})
            model_complexity = cfg.get("model_complexity", model_complexity)
            min_detection_confidence = cfg.get(
                "min_detection_confidence", min_detection_confidence
            )
            padding_ratio = cfg.get("padding_ratio", padding_ratio)

        self.padding_ratio = padding_ratio
        self._ensure_model()

        base_options = python.BaseOptions(
            model_asset_path=str(self.MODEL_PATH)
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, frame: np.ndarray, bbox: BBox) -> Optional[PoseKeypoints]:
        """
        Run pose estimation on the region around *bbox* in *frame*.

        Args:
            frame: Full BGR frame from the camera.
            bbox: Person bounding box (from tracker).

        Returns:
            PoseKeypoints with landmarks in full-frame pixel coordinates,
            or None if no pose was detected.
        """
        h, w = frame.shape[:2]

        # Compute padded crop region (clamped to frame bounds).
        pad_x = int((bbox.x2 - bbox.x1) * self.padding_ratio)
        pad_y = int((bbox.y2 - bbox.y1) * self.padding_ratio)
        cx1 = max(0, bbox.x1 - pad_x)
        cy1 = max(0, bbox.y1 - pad_y)
        cx2 = min(w, bbox.x2 + pad_x)
        cy2 = min(h, bbox.y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        # MediaPipe expects RGB.
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
        results = self.landmarker.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        raw = results.pose_landmarks[0]
        crop_h, crop_w = crop.shape[:2]

        # Map normalised crop coords → full-frame pixel coords.
        landmarks = np.zeros((33, 4), dtype=np.float32)
        for i, lm in enumerate(raw):
            landmarks[i, 0] = lm.x * crop_w + cx1   # x in frame
            landmarks[i, 1] = lm.y * crop_h + cy1   # y in frame
            landmarks[i, 2] = lm.z                   # relative depth (negative = closer)
            landmarks[i, 3] = lm.visibility

        avg_vis = float(np.mean(landmarks[KEY_LANDMARKS, 3]))

        padded_bbox = BBox(
            x1=cx1, y1=cy1, x2=cx2, y2=cy2,
            confidence=bbox.confidence, class_id=bbox.class_id,
        )

        return PoseKeypoints(
            landmarks=landmarks,
            bbox_in_frame=padded_bbox,
            avg_visibility=avg_vis,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.landmarker.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not self.MODEL_PATH.exists():
            print("Downloading pose landmarker model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print(f"Model downloaded to {self.MODEL_PATH}")

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}
