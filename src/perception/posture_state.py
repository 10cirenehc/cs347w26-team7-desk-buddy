"""
EWMA smoothing + hysteresis state machine for stable posture output.

Converts a noisy per-frame ``p_bad`` probability into a stable
posture state (good / bad / unknown) that doesn't flicker.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import yaml
from pathlib import Path


class PostureLabel(Enum):
    GOOD = "good"
    BAD = "bad"
    UNKNOWN = "unknown"


@dataclass
class SmoothedPostureState:
    """Output of the posture state machine."""
    state: PostureLabel
    raw_prob: float             # p_bad from classifier this frame
    smoothed_prob: float        # EWMA-smoothed p_bad
    frames_in_state: int        # consecutive frames in current state
    is_confident: bool          # True when visibility was above threshold


class PostureStateMachine:
    """
    EWMA smoothing with hysteresis thresholds.

    * ``smoothed = alpha * p_bad + (1 - alpha) * prev``
    * good → bad when ``smoothed > t_on``
    * bad → good when ``smoothed < t_off``
    * unknown when ``avg_visibility < unknown_visibility``
    """

    def __init__(
        self,
        ewma_alpha: float = 0.3,
        t_on: float = 0.65,
        t_off: float = 0.45,
        unknown_visibility: float = 0.5,
        config_path: Optional[str] = None,
    ):
        if config_path:
            cfg = self._load_config(config_path).get("posture_state", {})
            ewma_alpha = cfg.get("ewma_alpha", ewma_alpha)
            t_on = cfg.get("t_on", t_on)
            t_off = cfg.get("t_off", t_off)
            unknown_visibility = cfg.get("unknown_visibility", unknown_visibility)

        self.alpha = ewma_alpha
        self.t_on = t_on
        self.t_off = t_off
        self.unknown_visibility = unknown_visibility

        self._smoothed: float = 0.0
        self._state: PostureLabel = PostureLabel.GOOD
        self._frames_in_state: int = 0

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def update(self, p_bad: float, avg_visibility: float) -> SmoothedPostureState:
        """
        Process one frame's classifier output.

        Args:
            p_bad: Probability of bad posture from PostureClassifier.
            avg_visibility: Average landmark visibility from PoseKeypoints.

        Returns:
            SmoothedPostureState.
        """
        is_confident = avg_visibility >= self.unknown_visibility

        if not is_confident:
            # Don't update the EWMA when we can't trust the input.
            self._frames_in_state += 1
            return SmoothedPostureState(
                state=PostureLabel.UNKNOWN,
                raw_prob=p_bad,
                smoothed_prob=self._smoothed,
                frames_in_state=self._frames_in_state,
                is_confident=False,
            )

        # EWMA update.
        self._smoothed = self.alpha * p_bad + (1 - self.alpha) * self._smoothed

        # Hysteresis transitions.
        new_state = self._state
        if self._state == PostureLabel.GOOD and self._smoothed > self.t_on:
            new_state = PostureLabel.BAD
        elif self._state == PostureLabel.BAD and self._smoothed < self.t_off:
            new_state = PostureLabel.GOOD
        elif self._state == PostureLabel.UNKNOWN:
            # Re-entering from unknown — decide based on smoothed value.
            new_state = PostureLabel.BAD if self._smoothed > self.t_on else PostureLabel.GOOD

        if new_state != self._state:
            self._state = new_state
            self._frames_in_state = 1
        else:
            self._frames_in_state += 1

        return SmoothedPostureState(
            state=self._state,
            raw_prob=p_bad,
            smoothed_prob=self._smoothed,
            frames_in_state=self._frames_in_state,
            is_confident=True,
        )

    def reset(self) -> None:
        self._smoothed = 0.0
        self._state = PostureLabel.GOOD
        self._frames_in_state = 0
