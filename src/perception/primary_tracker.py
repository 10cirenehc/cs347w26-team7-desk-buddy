"""
Multi-person tracker with sticky primary-user selection.

Uses supervision.ByteTrack for IoU-based multi-object tracking, then
selects the primary user (largest bbox by default) and keeps them
sticky across frames.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from pathlib import Path

import supervision as sv

from .person_detector import BBox, PERSON_CLASS


@dataclass
class TrackedPerson:
    """A tracked person with a stable ID."""
    tracker_id: int
    bbox: BBox
    is_primary: bool = False


class PrimaryTracker:
    """
    ByteTrack wrapper with primary-user selection.

    * Receives person BBox list from PersonDetector each detection cycle.
    * Maintains stable tracker IDs via supervision.ByteTrack.
    * Selects the primary user as the largest bbox on first detection.
    * Sticks with the same tracker_id until the track is lost for
      ``max_lost_frames`` consecutive detection cycles.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost_frames: int = 30,
        primary_strategy: str = "largest",
        size_switch_threshold: float = 1.8,  # Switch if someone is 1.8x larger
        config_path: Optional[str] = None,
    ):
        if config_path:
            cfg = self._load_config(config_path).get("tracker", {})
            iou_threshold = cfg.get("iou_threshold", iou_threshold)
            max_lost_frames = cfg.get("max_lost_frames", max_lost_frames)
            primary_strategy = cfg.get("primary_strategy", primary_strategy)
            size_switch_threshold = cfg.get("size_switch_threshold", size_switch_threshold)

        self.primary_strategy = primary_strategy
        self.max_lost_frames = max_lost_frames
        self.size_switch_threshold = size_switch_threshold

        self._byte_tracker = sv.ByteTrack(
            minimum_matching_threshold=iou_threshold,
            lost_track_buffer=max_lost_frames,
        )

        self._primary_id: Optional[int] = None
        self._frames_since_primary_seen: int = 0
        self._tracked: List[TrackedPerson] = []

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, persons: List[BBox], frame: np.ndarray) -> List[TrackedPerson]:
        """
        Feed new person detections into the tracker.

        Args:
            persons: Person BBox list from PersonDetector.
            frame: Current BGR frame (used only for frame shape).

        Returns:
            List of TrackedPerson with one marked ``is_primary=True``.
        """
        if not persons:
            self._frames_since_primary_seen += 1
            if self._frames_since_primary_seen > self.max_lost_frames:
                self._primary_id = None
            self._tracked = []
            return self._tracked

        # Build supervision Detections array.
        xyxy = np.array([[b.x1, b.y1, b.x2, b.y2] for b in persons], dtype=np.float32)
        confs = np.array([b.confidence for b in persons], dtype=np.float32)
        class_ids = np.array([b.class_id for b in persons], dtype=int)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=class_ids,
        )

        detections = self._byte_tracker.update_with_detections(detections)

        # Rebuild tracked persons with tracker IDs.
        tracked: List[TrackedPerson] = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            bbox = BBox(
                x1=int(x1), y1=int(y1),
                x2=int(x2), y2=int(y2),
                confidence=float(detections.confidence[i]) if detections.confidence is not None else 0.0,
                class_id=int(detections.class_id[i]) if detections.class_id is not None else PERSON_CLASS,
            )
            tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            tracked.append(TrackedPerson(tracker_id=tid, bbox=bbox))

        # --- Primary selection logic ---
        # Strategy: ALWAYS track the largest person (user at desk)
        # Minimal stickiness to prevent single-frame jitter

        if not tracked:
            self._frames_since_primary_seen += 1
            if self._frames_since_primary_seen > self.max_lost_frames:
                self._primary_id = None
            self._tracked = tracked
            return tracked

        # Find the largest person in frame
        largest = max(tracked, key=lambda tp: tp.bbox.area)

        # Check if we have a current primary
        current_primary = None
        if self._primary_id is not None:
            for tp in tracked:
                if tp.tracker_id == self._primary_id:
                    current_primary = tp
                    break

        # Decision: Almost always follow the largest person
        # Only stick with current if they're ALMOST as large (within 20%)
        if current_primary is None:
            # No current primary - use largest
            largest.is_primary = True
            self._primary_id = largest.tracker_id
            self._frames_since_primary_seen = 0
        elif len(tracked) == 1:
            # Only one person - must be them
            tracked[0].is_primary = True
            self._primary_id = tracked[0].tracker_id
            self._frames_since_primary_seen = 0
        else:
            # Multiple people: prefer largest, but allow small fluctuations
            # Only stick with current if they're at least 80% the size of largest
            min_area_ratio = 0.8
            current_area_ratio = current_primary.bbox.area / max(largest.bbox.area, 1)

            if current_area_ratio < min_area_ratio:
                # Current primary is significantly smaller - switch to largest
                largest.is_primary = True
                self._primary_id = largest.tracker_id
            else:
                # Current primary is still reasonably large - keep them (prevents jitter)
                current_primary.is_primary = True

            self._frames_since_primary_seen = 0

        self._tracked = tracked
        return tracked

    def get_primary(self) -> Optional[TrackedPerson]:
        """Return the current primary person (even between detection cycles)."""
        for tp in self._tracked:
            if tp.is_primary:
                return tp
        return None

    def reset(self) -> None:
        self._byte_tracker.reset()
        self._primary_id = None
        self._frames_since_primary_seen = 0
        self._tracked = []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _select_primary(self, tracked: List[TrackedPerson]) -> Optional[TrackedPerson]:
        if not tracked:
            return None
        if self.primary_strategy == "largest":
            return max(tracked, key=lambda tp: tp.bbox.area)
        # Default fallback: largest.
        return max(tracked, key=lambda tp: tp.bbox.area)
