"""
Desk Buddy Perception Layer (Pipeline 2.0)

Modular pipeline: person detection -> tracking -> pose estimation ->
calibrated features -> learned classifier -> smoothed state output.
"""

from .video_source import VideoSource
from .person_detector import PersonDetector, BBox, PERSON_CLASS, PHONE_CLASS
from .primary_tracker import PrimaryTracker, TrackedPerson
from .pose_estimator import PoseEstimator, PoseKeypoints
from .posture_features import extract_features, PostureFeatures
from .calibration import CalibrationManager, CalibrationProfile
from .posture_model import PostureClassifier, PostureClassification
from .posture_state import PostureStateMachine, SmoothedPostureState, PostureLabel
from .gaze_tracker import GazeTracker, AttentionState, GazeResult
from .focus_estimator import FocusEstimator, FocusState, FocusEstimation
from .skeleton_renderer import render_skeleton, render_skeleton_rgb, render_skeleton_depth
from .presence_detector import PresenceDetector, PresenceState, PresenceResult, PresenceCalibration
from .state_logger import StateLogger, StateSnapshot, StateEvent
from .state_history import StateHistory, TrendDirection, TrendResult
from .state_summarizer import StateSummarizer, StateSummary

__all__ = [
    # Video
    "VideoSource",
    # Detection + Tracking
    "PersonDetector", "BBox", "PERSON_CLASS", "PHONE_CLASS",
    "PrimaryTracker", "TrackedPerson",
    # Pose
    "PoseEstimator", "PoseKeypoints",
    # Features + Calibration
    "extract_features", "PostureFeatures",
    "CalibrationManager", "CalibrationProfile",
    # Classification + State
    "PostureClassifier", "PostureClassification",
    "PostureStateMachine", "SmoothedPostureState", "PostureLabel",
    # Gaze
    "GazeTracker", "AttentionState", "GazeResult",
    # Focus fusion
    "FocusEstimator", "FocusState", "FocusEstimation",
    # Skeleton rendering
    "render_skeleton", "render_skeleton_rgb", "render_skeleton_depth",
    # Presence detection
    "PresenceDetector", "PresenceState", "PresenceResult", "PresenceCalibration",
    # State logging
    "StateLogger", "StateSnapshot", "StateEvent",
    "StateHistory", "TrendDirection", "TrendResult",
    "StateSummarizer", "StateSummary",
]
