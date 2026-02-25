#!/usr/bin/env python3
"""
Session-based posture data collection for multi-angle training.

Records timed sessions (default 30 seconds) of posture data with:
- Raw keypoints (132 values: 33 landmarks * 4 values each)
- Derived features (6 values)
- Skeleton images for future LAViTSPose-style ViT training
- Angle and session metadata for stratified analysis

No calibration required - raw data is preserved for flexible training.

Usage:
    python scripts/collect_posture_sessions.py
    python scripts/collect_posture_sessions.py --session-duration 20
    python scripts/collect_posture_sessions.py --no-skeleton-images

Controls:
    1-4     Select camera angle (1=front, 2=side-left, 3=side-right, 4=45-deg)
    g/b     Select posture label (good/bad)
    SPACE   Start/stop recording session
    q       Quit
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.video_source import VideoSource
from src.perception.person_detector import PersonDetector
from src.perception.primary_tracker import PrimaryTracker
from src.perception.pose_estimator import PoseEstimator, PoseKeypoints
from src.perception.posture_features import extract_features, PostureFeatures
from src.perception.skeleton_renderer import render_skeleton


ANGLE_TAGS = {
    ord("1"): "front",
    ord("2"): "side-left",
    ord("3"): "side-right",
    ord("4"): "45-deg",
}

LABEL_TAGS = {
    ord("g"): "good",
    ord("b"): "bad",
}


def get_csv_header() -> list:
    """Generate CSV header with all columns."""
    header = ["timestamp", "session_id", "angle", "label"]

    # Raw keypoints: 33 landmarks * 4 values (x, y, z, visibility) - MediaPipe format
    for i in range(33):
        header.extend([f"kp_{i}_x", f"kp_{i}_y", f"kp_{i}_z", f"kp_{i}_vis"])

    # Derived features (7 total, including depth-based forward_lean_z)
    header.extend([
        "torso_pitch", "head_forward_ratio", "shoulder_roll",
        "lateral_lean", "head_tilt", "avg_visibility", "forward_lean_z",
    ])

    # Skeleton image path
    header.append("skeleton_image_path")

    return header


def keypoints_to_row(
    kp: PoseKeypoints,
    features: PostureFeatures,
    session_id: str,
    angle: str,
    label: str,
    skeleton_path: Optional[str],
) -> list:
    """Convert keypoints and features to a CSV row."""
    row = [time.time(), session_id, angle, label]

    # Raw keypoints (flattened) - MediaPipe 33-landmark format
    landmarks = kp.landmarks  # (33, 4) - x, y, z, visibility
    for i in range(33):
        row.extend([
            f"{landmarks[i, 0]:.4f}",
            f"{landmarks[i, 1]:.4f}",
            f"{landmarks[i, 2]:.6f}",
            f"{landmarks[i, 3]:.4f}",
        ])

    # Derived features (7 total)
    row.extend([
        f"{features.torso_pitch:.4f}",
        f"{features.head_forward_ratio:.4f}",
        f"{features.shoulder_roll:.4f}",
        f"{features.lateral_lean:.4f}",
        f"{features.head_tilt:.4f}",
        f"{features.avg_visibility:.4f}",
        f"{features.forward_lean_z:.6f}",
    ])

    # Skeleton image path (relative)
    row.append(skeleton_path if skeleton_path else "")

    return row


class SessionRecorder:
    """Manages recording sessions and data persistence."""

    def __init__(
        self,
        output_dir: Path,
        skeleton_dir: Path,
        save_skeletons: bool = True,
        skeleton_size: int = 224,
    ):
        self.output_dir = output_dir
        self.skeleton_dir = skeleton_dir
        self.save_skeletons = save_skeletons
        self.skeleton_size = skeleton_size

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_skeletons:
            self.skeleton_dir.mkdir(parents=True, exist_ok=True)

        # Current session state
        self.session_id: Optional[str] = None
        self.csv_file = None
        self.csv_writer = None
        self.frame_idx = 0

        # Statistics
        self.total_samples = {"good": 0, "bad": 0}

    def start_session(self, angle: str, label: str) -> str:
        """Start a new recording session."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = ts
        filename = f"session_{ts}_{angle}_{label}.csv"
        filepath = self.output_dir / filename

        self.csv_file = open(filepath, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(get_csv_header())
        self.frame_idx = 0

        print(f"\nStarted session: {filename}")
        return self.session_id

    def record_frame(
        self,
        kp: PoseKeypoints,
        features: PostureFeatures,
        angle: str,
        label: str,
    ) -> None:
        """Record a single frame to the current session."""
        if self.csv_writer is None or self.session_id is None:
            return

        skeleton_path = None
        if self.save_skeletons:
            # Render and save skeleton image
            skeleton_img = render_skeleton(
                kp.landmarks,
                output_size=self.skeleton_size,
                upper_body_only=True,
            )
            skeleton_filename = f"{self.session_id}_{self.frame_idx:06d}.png"
            skeleton_filepath = self.skeleton_dir / skeleton_filename
            cv2.imwrite(str(skeleton_filepath), skeleton_img)
            skeleton_path = f"skeleton_images/{skeleton_filename}"

        row = keypoints_to_row(
            kp, features, self.session_id, angle, label, skeleton_path
        )
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self.frame_idx += 1
        self.total_samples[label] = self.total_samples.get(label, 0) + 1

    def stop_session(self) -> int:
        """Stop the current session and return frame count."""
        frames = self.frame_idx
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        self.session_id = None
        self.frame_idx = 0
        print(f"Session complete: {frames} frames recorded")
        return frames

    def is_recording(self) -> bool:
        return self.session_id is not None

    def close(self):
        if self.csv_file:
            self.csv_file.close()


def draw_overlay(
    frame: np.ndarray,
    angle: str,
    label: str,
    recording: bool,
    countdown: float,
    session_frames: int,
    total_samples: dict,
    features: Optional[PostureFeatures],
) -> np.ndarray:
    """Draw status overlay on frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Recording status (top center)
    if recording:
        status_text = f"RECORDING {countdown:.1f}s"
        status_color = (0, 0, 255)  # Red
        # Flashing red border
        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
    else:
        status_text = "IDLE - Press SPACE to record"
        status_color = (0, 255, 0)  # Green

    text_size = cv2.getTextSize(status_text, font, 0.8, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(annotated, status_text, (text_x, 35), font, 0.8, status_color, 2)

    # Current settings (top left)
    y = 70
    cv2.putText(annotated, f"Angle: {angle}", (10, y), font, 0.6, (255, 255, 0), 2)
    y += 25
    label_color = (0, 255, 0) if label == "good" else (0, 0, 255)
    cv2.putText(annotated, f"Label: {label}", (10, y), font, 0.6, label_color, 2)

    if recording:
        y += 25
        cv2.putText(annotated, f"Frames: {session_frames}", (10, y), font, 0.5, (255, 255, 255), 1)

    # Features (left side, below settings)
    if features is not None:
        y += 35
        cv2.putText(annotated, f"torso_pitch: {features.torso_pitch:.1f}",
                    (10, y), font, 0.45, (200, 200, 200), 1)
        y += 18
        cv2.putText(annotated, f"head_fwd: {features.head_forward_ratio:.2f}",
                    (10, y), font, 0.45, (200, 200, 200), 1)
        y += 18
        cv2.putText(annotated, f"shldr_roll: {features.shoulder_roll:.1f}",
                    (10, y), font, 0.45, (200, 200, 200), 1)
        y += 18
        cv2.putText(annotated, f"lat_lean: {features.lateral_lean:.2f}",
                    (10, y), font, 0.45, (200, 200, 200), 1)
        y += 18
        cv2.putText(annotated, f"head_tilt: {features.head_tilt:.1f}",
                    (10, y), font, 0.45, (200, 200, 200), 1)
        y += 18
        cv2.putText(annotated, f"visibility: {features.avg_visibility:.2f}",
                    (10, y), font, 0.45, (200, 200, 200), 1)
        y += 18
        # Depth-based forward lean (negative = leaning forward)
        fwd_color = (0, 0, 255) if features.forward_lean_z < -0.05 else (200, 200, 200)
        cv2.putText(annotated, f"fwd_lean_z: {features.forward_lean_z:.3f}",
                    (10, y), font, 0.45, fwd_color, 1)
    else:
        y += 35
        cv2.putText(annotated, "No pose detected", (10, y), font, 0.5, (0, 0, 255), 1)

    # Total samples (bottom left)
    samples_text = f"Total: good={total_samples.get('good', 0)} bad={total_samples.get('bad', 0)}"
    cv2.putText(annotated, samples_text, (10, h - 40), font, 0.5, (255, 255, 255), 1)

    # Controls (bottom)
    cv2.putText(annotated, "[1-4]=angle  [g/b]=label  [SPACE]=record  [q]=quit",
                (10, h - 15), font, 0.4, (128, 128, 128), 1)

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Session-based posture data collection")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--session-duration", type=float, default=30.0,
                        help="Duration of each recording session in seconds (default: 30)")
    parser.add_argument("--no-skeleton-images", action="store_true",
                        help="Don't save skeleton images (saves disk space)")
    parser.add_argument("--skeleton-size", type=int, default=224,
                        help="Skeleton image size (default: 224)")
    args = parser.parse_args()

    config_path = str(Path(__file__).parent.parent / args.config)
    base_dir = Path(__file__).parent.parent / "data"

    # Initialize pipeline components
    video = VideoSource(config_path=config_path)
    detector = PersonDetector(config_path=config_path)
    tracker = PrimaryTracker(config_path=config_path)
    pose = PoseEstimator(config_path=config_path)

    # Load detect_every_n from config
    import yaml as _yaml
    with open(config_path) as f:
        _cfg = _yaml.safe_load(f) or {}
    detect_every_n = _cfg.get("pipeline", {}).get("detect_every_n", 5)

    # Initialize recorder
    recorder = SessionRecorder(
        output_dir=base_dir / "posture_sessions",
        skeleton_dir=base_dir / "skeleton_images",
        save_skeletons=not args.no_skeleton_images,
        skeleton_size=args.skeleton_size,
    )

    if not video.open():
        print("Error: could not open camera")
        return 1

    window_name = "Posture Session Collection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    # State
    current_angle = "front"
    current_label = "good"
    recording_start_time = None
    frame_count = 0
    persons = []

    print("\n=== Posture Session Collection ===")
    print(f"Session duration: {args.session_duration}s")
    print(f"Skeleton images: {'disabled' if args.no_skeleton_images else 'enabled'}")
    print("\nControls:")
    print("  1-4:   Select angle (1=front, 2=side-left, 3=side-right, 4=45-deg)")
    print("  g/b:   Select label (good/bad)")
    print("  SPACE: Start/stop recording")
    print("  q:     Quit\n")

    try:
        while True:
            frame = video.read()
            if frame is None:
                continue

            # Detection (every N frames)
            if frame_count % detect_every_n == 0:
                persons, _ = detector.detect(frame)
                tracker.update(persons, frame)

            # Pose estimation
            primary = tracker.get_primary()
            kp = None
            features = None
            if primary is not None:
                kp = pose.estimate(frame, primary.bbox)
                if kp is not None:
                    features = extract_features(kp)

            # Recording logic
            recording = recorder.is_recording()
            countdown = 0.0
            if recording:
                elapsed = time.time() - recording_start_time
                countdown = max(0, args.session_duration - elapsed)

                # Record frame if we have valid pose
                if kp is not None and features is not None:
                    recorder.record_frame(kp, features, current_angle, current_label)

                # Auto-stop when duration reached
                if countdown <= 0:
                    recorder.stop_session()
                    recording = False

            # Draw overlay
            annotated = draw_overlay(
                frame,
                angle=current_angle,
                label=current_label,
                recording=recording,
                countdown=countdown,
                session_frames=recorder.frame_idx,
                total_samples=recorder.total_samples,
                features=features,
            )

            cv2.imshow(window_name, annotated)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                # Toggle recording
                if recorder.is_recording():
                    recorder.stop_session()
                else:
                    recorder.start_session(current_angle, current_label)
                    recording_start_time = time.time()
            elif key in ANGLE_TAGS and not recorder.is_recording():
                current_angle = ANGLE_TAGS[key]
                print(f"Angle set to: {current_angle}")
            elif key in LABEL_TAGS and not recorder.is_recording():
                current_label = LABEL_TAGS[key]
                print(f"Label set to: {current_label}")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        recorder.close()
        video.release()
        pose.close()
        cv2.destroyAllWindows()

    print(f"\nCollection complete!")
    print(f"Total samples: good={recorder.total_samples.get('good', 0)}, "
          f"bad={recorder.total_samples.get('bad', 0)}")
    print(f"Data saved to: {base_dir / 'posture_sessions'}")
    if not args.no_skeleton_images:
        print(f"Skeleton images: {base_dir / 'skeleton_images'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
