"""
LAViTSPose-style skeleton rendering for MediaPipe 33 keypoints.

Renders MediaPipe Pose landmarks as a binary skeleton image with
solid rectangles for limbs (not thin lines). This produces more
robust features for ViT-based posture classification.

Reference: LAViTSPose (MDPI Entropy, Nov 2025)
"""

import cv2
import numpy as np
from typing import Tuple

# MediaPipe Pose connections (33 landmarks)
MEDIAPIPE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),  # left eye path to ear
    (0, 4), (4, 5), (5, 6), (6, 8),  # right eye path to ear
    (9, 10),  # mouth
    # Torso
    (11, 12),  # shoulders
    (11, 23), (12, 24),  # shoulder to hip
    (23, 24),  # hips
    # Left arm
    (11, 13), (13, 15),  # shoulder -> elbow -> wrist
    (15, 17), (15, 19), (15, 21), (17, 19),  # hand
    # Right arm
    (12, 14), (14, 16),  # shoulder -> elbow -> wrist
    (16, 18), (16, 20), (16, 22), (18, 20),  # hand
    # Left leg
    (23, 25), (25, 27),  # hip -> knee -> ankle
    (27, 29), (27, 31), (29, 31),  # foot
    # Right leg
    (24, 26), (26, 28),  # hip -> knee -> ankle
    (28, 30), (28, 32), (30, 32),  # foot
]

# Upper body connections only (for seated posture where legs may be occluded)
UPPER_BODY_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12),  # shoulders
    (11, 23), (12, 24),  # shoulder to hip
    (23, 24),  # hips
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
]

# Number of keypoints in MediaPipe Pose
NUM_KEYPOINTS = 33


def render_skeleton(
    landmarks: np.ndarray,
    output_size: int = 224,
    line_width: int = 4,
    min_visibility: float = 0.5,
    upper_body_only: bool = False,
) -> np.ndarray:
    """
    Render MediaPipe pose landmarks as a binary skeleton image.

    Following LAViTSPose paper, limbs are drawn as solid rectangles
    (not thin lines) to enhance structural continuity and feature density.

    Args:
        landmarks: (33, 4) array of x, y, z, visibility from MediaPipe Pose.
        output_size: Output image dimension (square). Default 224 for ViT.
        line_width: Width of limb rectangles in pixels. Default 4 per paper.
        min_visibility: Skip limbs where either endpoint visibility < this.
        upper_body_only: If True, only render upper body connections.

    Returns:
        (output_size, output_size) uint8 image, white skeleton on black.
    """
    if landmarks.shape[0] != NUM_KEYPOINTS:
        raise ValueError(f"Expected {NUM_KEYPOINTS} keypoints, got {landmarks.shape[0]}")

    # Find bounding box of visible landmarks for letterbox scaling
    visible_mask = landmarks[:, 3] >= min_visibility
    if not np.any(visible_mask):
        # No visible landmarks, return blank image
        return np.zeros((output_size, output_size), dtype=np.uint8)

    visible_pts = landmarks[visible_mask, :2]
    min_x, min_y = visible_pts.min(axis=0)
    max_x, max_y = visible_pts.max(axis=0)

    # Add padding
    padding = 0.1
    width = max_x - min_x
    height = max_y - min_y
    if width < 1:
        width = 1
    if height < 1:
        height = 1

    min_x -= width * padding
    max_x += width * padding
    min_y -= height * padding
    max_y += height * padding

    width = max_x - min_x
    height = max_y - min_y

    # Letterbox scaling: fit into square while preserving aspect ratio
    scale = (output_size - 2 * line_width) / max(width, height)
    offset_x = (output_size - width * scale) / 2 - min_x * scale
    offset_y = (output_size - height * scale) / 2 - min_y * scale

    # Transform landmarks to output coordinates
    def transform(pt: np.ndarray) -> Tuple[int, int]:
        x = int(pt[0] * scale + offset_x)
        y = int(pt[1] * scale + offset_y)
        return (x, y)

    # Create output canvas
    canvas = np.zeros((output_size, output_size), dtype=np.uint8)

    # Select connections
    connections = UPPER_BODY_CONNECTIONS if upper_body_only else MEDIAPIPE_CONNECTIONS

    # Draw limbs as thick lines (which appear as rectangles)
    for start_idx, end_idx in connections:
        # Check visibility of both endpoints
        if landmarks[start_idx, 3] < min_visibility:
            continue
        if landmarks[end_idx, 3] < min_visibility:
            continue

        pt1 = transform(landmarks[start_idx, :2])
        pt2 = transform(landmarks[end_idx, :2])

        # Draw thick line (rectangle-like appearance)
        cv2.line(canvas, pt1, pt2, 255, thickness=line_width, lineType=cv2.LINE_AA)

    # Draw joint circles at keypoints
    joint_radius = max(2, line_width // 2)
    for i in range(NUM_KEYPOINTS):
        if landmarks[i, 3] >= min_visibility:
            pt = transform(landmarks[i, :2])
            cv2.circle(canvas, pt, joint_radius, 255, thickness=-1, lineType=cv2.LINE_AA)

    return canvas


def render_skeleton_rgb(
    landmarks: np.ndarray,
    output_size: int = 224,
    line_width: int = 4,
    min_visibility: float = 0.5,
    upper_body_only: bool = False,
) -> np.ndarray:
    """
    Render skeleton as RGB image (white on black).

    Convenience wrapper that returns 3-channel image for models
    expecting RGB input.

    Args:
        Same as render_skeleton.

    Returns:
        (output_size, output_size, 3) uint8 RGB image.
    """
    gray = render_skeleton(
        landmarks,
        output_size=output_size,
        line_width=line_width,
        min_visibility=min_visibility,
        upper_body_only=upper_body_only,
    )
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def render_skeleton_depth(
    landmarks: np.ndarray,
    output_size: int = 224,
    line_width: int = 4,
    min_visibility: float = 0.5,
    upper_body_only: bool = False,
) -> np.ndarray:
    """
    Render skeleton with depth (z) encoded as color.

    Uses a blue-to-red colormap where:
    - Blue = farther from camera (positive z)
    - Red = closer to camera (negative z)

    This preserves depth information that would be lost in binary rendering.

    Args:
        landmarks: (33, 4) array of x, y, z, visibility from MediaPipe Pose.
        output_size: Output image dimension (square). Default 224 for ViT.
        line_width: Width of limb rectangles in pixels.
        min_visibility: Skip limbs where either endpoint visibility < this.
        upper_body_only: If True, only render upper body connections.

    Returns:
        (output_size, output_size, 3) uint8 BGR image with depth coloring.
    """
    if landmarks.shape[0] != NUM_KEYPOINTS:
        raise ValueError(f"Expected {NUM_KEYPOINTS} keypoints, got {landmarks.shape[0]}")

    # Find bounding box of visible landmarks for letterbox scaling
    visible_mask = landmarks[:, 3] >= min_visibility
    if not np.any(visible_mask):
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)

    visible_pts = landmarks[visible_mask, :2]
    min_x, min_y = visible_pts.min(axis=0)
    max_x, max_y = visible_pts.max(axis=0)

    # Add padding
    padding = 0.1
    width = max(max_x - min_x, 1)
    height = max(max_y - min_y, 1)

    min_x -= width * padding
    max_x += width * padding
    min_y -= height * padding
    max_y += height * padding

    width = max_x - min_x
    height = max_y - min_y

    # Letterbox scaling
    scale = (output_size - 2 * line_width) / max(width, height)
    offset_x = (output_size - width * scale) / 2 - min_x * scale
    offset_y = (output_size - height * scale) / 2 - min_y * scale

    def transform(pt: np.ndarray) -> Tuple[int, int]:
        x = int(pt[0] * scale + offset_x)
        y = int(pt[1] * scale + offset_y)
        return (x, y)

    # Normalize z values to 0-255 for colormapping
    z_values = landmarks[visible_mask, 2]
    z_min, z_max = z_values.min(), z_values.max()
    z_range = max(z_max - z_min, 1e-6)

    def z_to_color(z: float) -> Tuple[int, int, int]:
        """Map z to BGR color (blue=far, red=near)."""
        # Normalize to 0-1 (0 = near/negative z, 1 = far/positive z)
        norm_z = (z - z_min) / z_range
        # Blue channel increases with distance, red decreases
        b = int(norm_z * 255)
        r = int((1 - norm_z) * 255)
        g = 50  # slight green for visibility
        return (b, g, r)

    # Create output canvas
    canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)

    connections = UPPER_BODY_CONNECTIONS if upper_body_only else MEDIAPIPE_CONNECTIONS

    # Draw limbs with depth-based coloring
    for start_idx, end_idx in connections:
        if landmarks[start_idx, 3] < min_visibility:
            continue
        if landmarks[end_idx, 3] < min_visibility:
            continue

        pt1 = transform(landmarks[start_idx, :2])
        pt2 = transform(landmarks[end_idx, :2])

        # Average z of the two endpoints
        avg_z = (landmarks[start_idx, 2] + landmarks[end_idx, 2]) / 2
        color = z_to_color(avg_z)

        cv2.line(canvas, pt1, pt2, color, thickness=line_width, lineType=cv2.LINE_AA)

    # Draw joints with depth coloring
    joint_radius = max(2, line_width // 2)
    for i in range(NUM_KEYPOINTS):
        if landmarks[i, 3] >= min_visibility:
            pt = transform(landmarks[i, :2])
            color = z_to_color(landmarks[i, 2])
            cv2.circle(canvas, pt, joint_radius, color, thickness=-1, lineType=cv2.LINE_AA)

    return canvas
