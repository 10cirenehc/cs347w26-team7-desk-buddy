"""
Desk Buddy Perception Layer

This module provides computer vision capabilities for:
- Posture detection and classification
- Phone detection
- Gaze/attention tracking
- Focus state estimation
"""

from .camera import Camera
from .posture_detector import PostureDetector, PostureState, PostureResult
from .phone_detector import PhoneDetector, PhoneDetectionResult
from .gaze_tracker import GazeTracker, AttentionState, GazeResult
from .focus_estimator import FocusEstimator, FocusState

__all__ = [
    'Camera',
    'PostureDetector',
    'PostureState',
    'PostureResult',
    'PhoneDetector',
    'PhoneDetectionResult',
    'GazeTracker',
    'AttentionState',
    'GazeResult',
    'FocusEstimator',
    'FocusState',
]
