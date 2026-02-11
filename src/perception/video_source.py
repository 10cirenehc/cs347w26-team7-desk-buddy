"""
Video source with configurable resolution/FPS and frame-drop policy.

Replaces the old camera.py with a "latest" drop policy that drains
stale buffered frames so pose estimation always runs on the most
recent frame.
"""

import cv2
import numpy as np
from typing import Optional
import yaml
from pathlib import Path
import platform


class VideoSource:
    """Webcam capture wrapper with frame-drop policy."""

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        drop_policy: str = "latest",
        config_path: Optional[str] = None,
    ):
        if config_path:
            cfg = self._load_config(config_path).get("video_source", {})
            device_id = cfg.get("device_id", device_id)
            width = cfg.get("width", width)
            height = cfg.get("height", height)
            fps = cfg.get("fps", fps)
            drop_policy = cfg.get("drop_policy", drop_policy)

        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.drop_policy = drop_policy
        self.cap: Optional[cv2.VideoCapture] = None
        self._buffer_size_supported = False

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def open(self) -> bool:
        """Open the camera. Returns True on success."""
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Try to minimise internal buffer (works on Linux V4L2, may be
        # ignored on macOS AVFoundation).
        if self.drop_policy == "latest":
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Check if the backend actually honours the setting.
            actual = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
            self._buffer_size_supported = actual == 1.0

        return True

    def read(self) -> Optional[np.ndarray]:
        """
        Return the most recent BGR frame, or None on failure.

        When drop_policy='latest' and the backend does not support
        CAP_PROP_BUFFERSIZE=1, we do a grab-loop to drain stale frames
        before retrieving the latest one.
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        if self.drop_policy == "latest" and not self._buffer_size_supported:
            # Drain buffered frames – grab() is cheap (no decode).
            for _ in range(4):
                self.cap.grab()
            ret, frame = self.cap.retrieve()
        else:
            ret, frame = self.cap.read()

        return frame if ret else None

    def read_rgb(self) -> Optional[np.ndarray]:
        """Read and convert to RGB."""
        frame = self.read()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_frame_size(self) -> tuple:
        if self.cap is None:
            return (self.width, self.height)
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # Context-manager support
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
