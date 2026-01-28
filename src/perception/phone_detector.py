"""
Phone detection using YOLOv8.

Detects cell phones in frame and estimates if phone is being held/used.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import yaml
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class PhoneDetectionResult:
    """Result from phone detection."""
    phone_detected: bool
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    in_hand: bool  # heuristic: phone in typical hand-holding position
    all_detections: List[dict]  # all phone detections if multiple


class PhoneDetector:
    """
    Phone detector using YOLOv8.

    Uses YOLOv8n (nano) model for real-time phone detection.
    COCO dataset includes "cell phone" as class 67.
    """

    PHONE_CLASS_ID = 67  # COCO class ID for cell phone

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        model_name: str = 'yolov8n.pt',
        config_path: Optional[str] = None
    ):
        """
        Initialize phone detector.

        Args:
            confidence_threshold: Minimum confidence for detection
            model_name: YOLOv8 model to use (auto-downloads if needed)
            config_path: Optional path to config file
        """
        if YOLO is None:
            raise ImportError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            )

        # Load config if provided
        if config_path:
            config = self._load_config(config_path)
            phone_config = config.get('phone_detection', {})
            confidence_threshold = phone_config.get(
                'confidence_threshold', confidence_threshold
            )
            self.hand_proximity_threshold = phone_config.get(
                'hand_proximity_threshold', 0.15
            )
        else:
            self.hand_proximity_threshold = 0.15

        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_name)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def detect(
        self,
        frame: np.ndarray,
        hand_positions: Optional[List[Tuple[float, float]]] = None
    ) -> PhoneDetectionResult:
        """
        Detect phones in frame.

        Args:
            frame: BGR image as numpy array
            hand_positions: Optional list of hand center positions (normalized 0-1)
                           for in-hand detection

        Returns:
            PhoneDetectionResult with detection status and details
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)

        phone_detections = []
        frame_height, frame_width = frame.shape[:2]

        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Filter for phone class with sufficient confidence
                if class_id == self.PHONE_CLASS_ID and confidence >= self.confidence_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Compute center for in-hand detection
                    center_x = (x1 + x2) / 2 / frame_width
                    center_y = (y1 + y2) / 2 / frame_height

                    phone_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'center': (center_x, center_y)
                    })

        if not phone_detections:
            return PhoneDetectionResult(
                phone_detected=False,
                confidence=0.0,
                bounding_box=None,
                in_hand=False,
                all_detections=[]
            )

        # Get highest confidence detection
        best_detection = max(phone_detections, key=lambda x: x['confidence'])

        # Check if phone is in hand (using heuristic or hand positions)
        in_hand = self._check_in_hand(
            best_detection, hand_positions, frame_height
        )

        return PhoneDetectionResult(
            phone_detected=True,
            confidence=best_detection['confidence'],
            bounding_box=best_detection['bbox'],
            in_hand=in_hand,
            all_detections=phone_detections
        )

    def _check_in_hand(
        self,
        phone_detection: dict,
        hand_positions: Optional[List[Tuple[float, float]]],
        frame_height: int
    ) -> bool:
        """
        Determine if phone is likely being held.

        Uses position heuristics and optional hand positions.

        Args:
            phone_detection: Detection dict with bbox and center
            hand_positions: Optional list of hand center positions
            frame_height: Frame height for position calculations

        Returns:
            True if phone appears to be in hand
        """
        phone_center = phone_detection['center']
        phone_center_x, phone_center_y = phone_center

        # Heuristic 1: Phone in typical hand-holding region
        # Usually lower-middle to lower portion of frame when being held
        in_hand_region = (
            0.2 < phone_center_x < 0.8 and  # not at edges
            0.3 < phone_center_y < 0.9      # middle to lower portion
        )

        # Heuristic 2: Phone size suggests it's close (being held)
        bbox = phone_detection['bbox']
        phone_height = (bbox[3] - bbox[1]) / frame_height
        appears_close = phone_height > 0.1  # takes up >10% of frame height

        # If hand positions provided, check proximity
        near_hand = False
        if hand_positions:
            for hand_x, hand_y in hand_positions:
                distance = np.sqrt(
                    (phone_center_x - hand_x)**2 +
                    (phone_center_y - hand_y)**2
                )
                if distance < self.hand_proximity_threshold:
                    near_hand = True
                    break

        # Combine heuristics
        if hand_positions:
            return near_hand or (in_hand_region and appears_close)
        else:
            return in_hand_region and appears_close

    def draw_detections(
        self,
        frame: np.ndarray,
        result: PhoneDetectionResult
    ) -> np.ndarray:
        """
        Draw phone detections on frame.

        Args:
            frame: BGR image to draw on
            result: PhoneDetectionResult from detect()

        Returns:
            Frame with detections drawn
        """
        annotated_frame = frame.copy()

        for detection in result.all_detections:
            bbox = detection['bbox']
            conf = detection['confidence']

            # Draw bounding box
            color = (0, 0, 255) if result.in_hand else (0, 255, 255)
            cv2.rectangle(
                annotated_frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color, 2
            )

            # Draw label
            label = f"Phone: {conf:.2f}"
            if result.in_hand:
                label += " (IN HAND)"

            cv2.putText(
                annotated_frame,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

        return annotated_frame

    def draw_status_overlay(
        self,
        frame: np.ndarray,
        result: PhoneDetectionResult,
        y_offset: int = 30
    ) -> np.ndarray:
        """
        Draw phone detection status overlay on frame.

        Args:
            frame: BGR image to draw on
            result: PhoneDetectionResult from detect()
            y_offset: Vertical offset for text

        Returns:
            Frame with status overlay
        """
        annotated_frame = frame.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        if result.phone_detected:
            if result.in_hand:
                status = "PHONE IN HAND"
                color = (0, 0, 255)  # Red
            else:
                status = "Phone visible"
                color = (0, 255, 255)  # Yellow
        else:
            status = "No phone"
            color = (0, 255, 0)  # Green

        cv2.putText(
            annotated_frame,
            f"Phone: {status}",
            (10, y_offset),
            font, font_scale, color, thickness
        )

        if result.phone_detected:
            cv2.putText(
                annotated_frame,
                f"Confidence: {result.confidence:.2f}",
                (10, y_offset + 25),
                font, 0.5, (255, 255, 255), 1
            )

        return annotated_frame
