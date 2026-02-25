#!/usr/bin/env python3
"""
Collect labelled posture training data.

Runs the detection → tracking → pose → feature pipeline with a live
webcam overlay.  Press hotkeys to label the current frame's features
and append them to a CSV file for later training.

Requires a calibration profile (run ``run_pipeline.py`` first).

Usage:
    python scripts/collect_posture_data.py --calibration data/calibration_profile.json
    python scripts/collect_posture_data.py --calibration data/calibration_profile.json --output data/posture_labels/session1.csv

Controls:
    g - Label current frame as GOOD
    b - Label current frame as BAD
    u - Label current frame as UNKNOWN (skip)
    q - Quit
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.video_source import VideoSource
from src.perception.person_detector import PersonDetector
from src.perception.primary_tracker import PrimaryTracker
from src.perception.pose_estimator import PoseEstimator
from src.perception.posture_features import extract_features
from src.perception.calibration import CalibrationManager, CalibrationProfile


def main():
    parser = argparse.ArgumentParser(description="Collect posture training data")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--calibration", required=True,
                        help="Path to calibration profile JSON")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: data/posture_labels/<timestamp>.csv)")
    args = parser.parse_args()

    config_path = str(Path(__file__).parent.parent / args.config)

    cal_profile = CalibrationManager.load(args.calibration)
    print(f"Loaded calibration ({cal_profile.n_samples} samples)")

    # Output path.
    if args.output:
        out_path = Path(args.output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path(__file__).parent.parent / f"data/posture_labels/{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialise pipeline.
    video = VideoSource(config_path=config_path)
    detector = PersonDetector(config_path=config_path)
    tracker = PrimaryTracker(config_path=config_path)
    pose = PoseEstimator(config_path=config_path)

    import yaml as _yaml
    with open(config_path) as f:
        _cfg = _yaml.safe_load(f) or {}
    detect_every_n = _cfg.get("pipeline", {}).get("detect_every_n", 5)

    if not video.open():
        print("Error: could not open camera")
        return 1

    window_name = "Posture Data Collection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    csv_file = open(out_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "timestamp", "label",
        "torso_pitch", "head_forward_ratio", "shoulder_roll",
        "lateral_lean", "head_tilt", "avg_visibility",
        "norm_torso_pitch", "norm_head_forward", "norm_shoulder_roll",
        "norm_lateral_lean", "norm_head_tilt", "norm_visibility",
    ])

    print(f"\nWriting to: {out_path}")
    print("Controls: [g]=good  [b]=bad  [u]=unknown  [q]=quit\n")

    frame_count = 0
    label_count = {"good": 0, "bad": 0, "unknown": 0}
    persons = []
    current_normed = None
    current_features = None

    try:
        while True:
            frame = video.read()
            if frame is None:
                continue

            if frame_count % detect_every_n == 0:
                persons, _ = detector.detect(frame)
                tracker.update(persons, frame)

            primary = tracker.get_primary()
            kp = None
            features = None
            normed = None
            if primary is not None:
                kp = pose.estimate(frame, primary.bbox)
                if kp is not None:
                    features = extract_features(kp)
                    normed = CalibrationManager.normalize(features, cal_profile)
                    current_features = features
                    current_normed = normed

            # Draw.
            annotated = frame.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX

            if features is not None:
                y = 30
                cv2.putText(annotated, f"torso_pitch: {features.torso_pitch:.1f}",
                             (10, y), font, 0.5, (255, 255, 255), 1)
                y += 20
                cv2.putText(annotated, f"head_fwd: {features.head_forward_ratio:.2f}",
                             (10, y), font, 0.5, (255, 255, 255), 1)
                y += 20
                cv2.putText(annotated, f"shldr_roll: {features.shoulder_roll:.1f}",
                             (10, y), font, 0.5, (255, 255, 255), 1)
                y += 20
                cv2.putText(annotated, f"lat_lean: {features.lateral_lean:.2f}",
                             (10, y), font, 0.5, (255, 255, 255), 1)
                y += 20
                cv2.putText(annotated, f"head_tilt: {features.head_tilt:.1f}",
                             (10, y), font, 0.5, (255, 255, 255), 1)
                y += 20
                cv2.putText(annotated, f"visibility: {features.avg_visibility:.2f}",
                             (10, y), font, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(annotated, "No pose detected", (10, 30),
                             font, 0.6, (0, 0, 255), 2)

            # Label counts.
            cv2.putText(annotated,
                         f"Labels — good:{label_count['good']}  "
                         f"bad:{label_count['bad']}  "
                         f"unknown:{label_count['unknown']}",
                         (10, annotated.shape[0] - 30), font, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, "[g]=good  [b]=bad  [u]=unknown  [q]=quit",
                         (10, annotated.shape[0] - 10), font, 0.4, (128, 128, 128), 1)

            cv2.imshow(window_name, annotated)

            # Keys.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key in (ord("g"), ord("b"), ord("u")):
                label_map = {ord("g"): "good", ord("b"): "bad", ord("u"): "unknown"}
                label = label_map[key]
                if current_features is not None and current_normed is not None:
                    raw = current_features.raw_vector
                    writer.writerow([
                        time.time(), label,
                        *[f"{v:.6f}" for v in raw],
                        *[f"{v:.6f}" for v in current_normed],
                    ])
                    csv_file.flush()
                    label_count[label] += 1
                    print(f"  Labelled: {label}  (total: {sum(label_count.values())})")
                else:
                    print("  No features to label — skipped")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        csv_file.close()
        video.release()
        pose.close()
        cv2.destroyAllWindows()

    print(f"\nSaved {sum(label_count.values())} samples to {out_path}")
    print(f"  good: {label_count['good']}, bad: {label_count['bad']}, "
          f"unknown: {label_count['unknown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
