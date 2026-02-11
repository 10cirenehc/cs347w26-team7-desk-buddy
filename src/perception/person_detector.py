"""
Person and phone detection using YOLOv8.

Single YOLO inference pass detects both persons (COCO class 0) and
cell phones (COCO class 67), returning separated bounding-box lists.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import yaml
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class BBox:
    """Axis-aligned bounding box with metadata."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


PERSON_CLASS = 0
PHONE_CLASS = 67


class PersonDetector:
    """
    YOLOv8s detector for persons + phones in a single forward pass.

    Returns two lists of BBox: one for persons, one for phones.
    """

    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        confidence: float = 0.4,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
        config_path: Optional[str] = None,
    ):
        if YOLO is None:
            raise ImportError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            )

        if config_path:
            cfg = self._load_config(config_path).get("person_detector", {})
            model_name = cfg.get("model", model_name)
            confidence = cfg.get("confidence", confidence)
            device = cfg.get("device", device)
            classes = cfg.get("classes", classes)

        self.confidence = confidence
        self.device = device
        self.classes = classes or [PERSON_CLASS, PHONE_CLASS]
        self.model = YOLO(model_name)

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def detect(self, frame: np.ndarray) -> Tuple[List[BBox], List[BBox]]:
        """
        Run detection on a BGR frame.

        Returns:
            (persons, phones) — two lists of BBox.
        """
        results = self.model(
            frame,
            verbose=False,
            conf=self.confidence,
            device=self.device,
            classes=self.classes,
        )

        persons: List[BBox] = []
        phones: List[BBox] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = BBox(
                    x1=int(x1), y1=int(y1),
                    x2=int(x2), y2=int(y2),
                    confidence=conf, class_id=cls,
                )
                if cls == PERSON_CLASS:
                    persons.append(bbox)
                elif cls == PHONE_CLASS:
                    phones.append(bbox)

        return persons, phones
