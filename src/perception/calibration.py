"""
Neutral-posture calibration manager.

Collects feature samples during a calibration window, computes per-feature
mean/std, and normalises incoming features as z-scores so the downstream
classifier is independent of the user's body proportions and camera angle.
"""

import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import yaml

from .posture_features import PostureFeatures


@dataclass
class CalibrationProfile:
    """Stored calibration statistics."""
    means: np.ndarray      # (6,)
    stds: np.ndarray       # (6,)
    n_samples: int = 0
    timestamp: float = 0.0


class CalibrationManager:
    """
    10-second neutral-pose baseline collection and delta normalization.

    Usage::

        cal = CalibrationManager(duration_seconds=10)
        cal.start()
        while not cal.is_ready():
            features = extract_features(kp)
            cal.add_sample(features)
        profile = cal.finish()
        cal.save(profile, "data/calibration_profile.json")

        # later
        profile = CalibrationManager.load("data/calibration_profile.json")
        normed = CalibrationManager.normalize(features, profile)
    """

    def __init__(
        self,
        duration_seconds: float = 10.0,
        save_path: str = "data/calibration_profile.json",
        config_path: Optional[str] = None,
    ):
        if config_path:
            cfg = self._load_config(config_path).get("calibration", {})
            duration_seconds = cfg.get("duration_seconds", duration_seconds)
            save_path = cfg.get("save_path", save_path)

        self.duration_seconds = duration_seconds
        self.save_path = save_path

        self._samples: list = []
        self._start_time: Optional[float] = None

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    # ------------------------------------------------------------------
    # Collection API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin the calibration window."""
        self._samples = []
        self._start_time = time.time()

    def add_sample(self, features: PostureFeatures) -> None:
        """Add a sample if visibility is acceptable."""
        if self._start_time is None:
            return
        if features.avg_visibility < 0.5:
            return
        self._samples.append(features.raw_vector.copy())

    def elapsed(self) -> float:
        """Seconds since calibration started."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def is_ready(self) -> bool:
        """True when the calibration window has elapsed."""
        return self.elapsed() >= self.duration_seconds

    def finish(self) -> CalibrationProfile:
        """Compute and return the calibration profile."""
        if not self._samples:
            # Fallback: identity normalisation (7 features including forward_lean_z).
            return CalibrationProfile(
                means=np.zeros(7, dtype=np.float32),
                stds=np.ones(7, dtype=np.float32),
                n_samples=0,
                timestamp=time.time(),
            )
        arr = np.stack(self._samples)  # (N, 7)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)
        stds = np.maximum(stds, 1e-6)  # guard div-by-zero
        return CalibrationProfile(
            means=means.astype(np.float32),
            stds=stds.astype(np.float32),
            n_samples=len(self._samples),
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(features: PostureFeatures, profile: CalibrationProfile) -> np.ndarray:
        """
        Z-score normalize a feature vector against the calibration profile.

        Returns:
            (6,) numpy array of normalised features.
        """
        return (features.raw_vector - profile.means) / profile.stds

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save(profile: CalibrationProfile, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "means": profile.means.tolist(),
            "stds": profile.stds.tolist(),
            "n_samples": profile.n_samples,
            "timestamp": profile.timestamp,
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> CalibrationProfile:
        with open(path, "r") as f:
            data = json.load(f)
        return CalibrationProfile(
            means=np.array(data["means"], dtype=np.float32),
            stds=np.array(data["stds"], dtype=np.float32),
            n_samples=data.get("n_samples", 0),
            timestamp=data.get("timestamp", 0.0),
        )
