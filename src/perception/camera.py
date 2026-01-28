"""
Camera utility for webcam capture.
"""

import cv2
import numpy as np
from typing import Optional
import yaml
from pathlib import Path


class Camera:
    """Simple wrapper for OpenCV webcam capture."""

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        config_path: Optional[str] = None
    ):
        """
        Initialize camera capture.

        Args:
            device_id: Camera device index (default 0 for primary webcam)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second
            config_path: Optional path to config file to load settings
        """
        # Load from config if provided
        if config_path:
            config = self._load_config(config_path)
            camera_config = config.get('camera', {})
            device_id = camera_config.get('device_id', device_id)
            width = camera_config.get('width', width)
            height = camera_config.get('height', height)
            fps = camera_config.get('fps', fps)

        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def open(self) -> bool:
        """
        Open the camera for capture.

        Returns:
            True if camera opened successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        return True

    def read(self) -> Optional[np.ndarray]:
        """
        Read a frame from the camera.

        Returns:
            BGR frame as numpy array, or None if read failed
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def read_rgb(self) -> Optional[np.ndarray]:
        """
        Read a frame and convert to RGB.

        Returns:
            RGB frame as numpy array, or None if read failed
        """
        frame = self.read()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self) -> bool:
        """Check if camera is currently open."""
        return self.cap is not None and self.cap.isOpened()

    def get_frame_size(self) -> tuple:
        """Get the actual frame size being captured."""
        if self.cap is None:
            return (self.width, self.height)
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
