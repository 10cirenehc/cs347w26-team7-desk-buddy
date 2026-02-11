"""
Pure-function feature extraction from pose keypoints.

Computes interpretable geometric features that are invariant to
camera mirroring and work well with calibration + learned classifiers.

Uses MediaPipe's 33 landmark format with z-depth for forward lean detection.
"""

import math
import numpy as np
from dataclasses import dataclass

from .pose_estimator import PoseKeypoints

# MediaPipe Pose landmark indices
_NOSE = 0
_LEFT_EAR = 7
_RIGHT_EAR = 8
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24


@dataclass
class PostureFeatures:
    """Interpretable posture feature vector."""
    torso_pitch: float          # angle of hip→shoulder vs vertical (degrees, positive = forward)
    head_forward_ratio: float   # ear-shoulder forward offset / shoulder width
    shoulder_roll: float        # shoulder line tilt from horizontal (degrees, always positive)
    lateral_lean: float         # shoulder_mid horizontal offset from hip_mid / shoulder width
    head_tilt: float            # ear line tilt from horizontal (degrees, always positive)
    avg_visibility: float       # mean visibility of key landmarks
    forward_lean_z: float       # shoulder z vs hip z (negative = shoulders forward)

    @property
    def raw_vector(self) -> np.ndarray:
        """All 7 features as a flat numpy array."""
        return np.array([
            self.torso_pitch,
            self.head_forward_ratio,
            self.shoulder_roll,
            self.lateral_lean,
            self.head_tilt,
            self.avg_visibility,
            self.forward_lean_z,
        ], dtype=np.float32)


def extract_features(kp: PoseKeypoints) -> PostureFeatures:
    """
    Compute posture features from pose keypoints.

    All angular features use ``abs()`` where appropriate so that
    mirrored webcam views produce the same values.

    Args:
        kp: PoseKeypoints from PoseEstimator (MediaPipe 33-landmark format).

    Returns:
        PostureFeatures dataclass.
    """
    lm = kp.landmarks  # (33, 4) — x, y, z, visibility

    # Key points (pixel coords).
    ls = lm[_LEFT_SHOULDER, :2]
    rs = lm[_RIGHT_SHOULDER, :2]
    le = lm[_LEFT_EAR, :2]
    re = lm[_RIGHT_EAR, :2]
    lh = lm[_LEFT_HIP, :2]
    rh = lm[_RIGHT_HIP, :2]

    # Z coordinates (depth - negative means closer to camera)
    ls_z = lm[_LEFT_SHOULDER, 2]
    rs_z = lm[_RIGHT_SHOULDER, 2]
    lh_z = lm[_LEFT_HIP, 2]
    rh_z = lm[_RIGHT_HIP, 2]

    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0
    ear_mid = (le + re) / 2.0

    shoulder_dx = rs[0] - ls[0]
    shoulder_dy = rs[1] - ls[1]
    shoulder_width = math.sqrt(shoulder_dx ** 2 + shoulder_dy ** 2)
    shoulder_width = max(shoulder_width, 1e-6)  # guard div-by-zero

    # 1. Torso pitch: angle of hip_mid→shoulder_mid vs vertical.
    #    Vertical in image coords points downward (+y), so the "up"
    #    reference vector is (0, -1).  A positive angle means the
    #    shoulders are forward (closer to camera) relative to hips.
    torso_dx = shoulder_mid[0] - hip_mid[0]
    torso_dy = shoulder_mid[1] - hip_mid[1]
    # atan2(horizontal_component, -vertical_component)
    torso_pitch = math.degrees(math.atan2(torso_dx, -torso_dy))

    # 2. Head-forward ratio: vertical offset of ear_mid above
    #    shoulder_mid, normalised by shoulder width.
    #    In image coords a *smaller* y means higher.  When the head
    #    is pushed forward, ear_mid.y gets closer to shoulder_mid.y.
    head_forward_ratio = (ear_mid[1] - shoulder_mid[1]) / shoulder_width

    # 3. Shoulder roll: tilt of the shoulder line from horizontal.
    shoulder_angle_raw = math.degrees(math.atan2(shoulder_dy, shoulder_dx))
    shoulder_roll = abs(shoulder_angle_raw)
    if shoulder_roll > 90:
        shoulder_roll = 180 - shoulder_roll

    # 4. Lateral lean: horizontal displacement of shoulder_mid from
    #    hip_mid, normalised by shoulder width.
    lateral_lean = (shoulder_mid[0] - hip_mid[0]) / shoulder_width

    # 5. Head tilt: tilt of the ear line from horizontal.
    ear_dx = re[0] - le[0]
    ear_dy = re[1] - le[1]
    ear_angle_raw = math.degrees(math.atan2(ear_dy, ear_dx))
    head_tilt = abs(ear_angle_raw)
    if head_tilt > 90:
        head_tilt = 180 - head_tilt

    # 6. Forward lean from z-coordinates (depth).
    #    Negative z means closer to camera.
    #    If shoulders are more negative than hips, person is leaning forward.
    shoulder_z_avg = (ls_z + rs_z) / 2.0
    hip_z_avg = (lh_z + rh_z) / 2.0
    forward_lean_z = shoulder_z_avg - hip_z_avg  # negative = forward lean

    return PostureFeatures(
        torso_pitch=torso_pitch,
        head_forward_ratio=head_forward_ratio,
        shoulder_roll=shoulder_roll,
        lateral_lean=lateral_lean,
        head_tilt=head_tilt,
        avg_visibility=kp.avg_visibility,
        forward_lean_z=forward_lean_z,
    )
