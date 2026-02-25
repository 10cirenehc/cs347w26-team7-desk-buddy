#!/usr/bin/env python3
"""
Debug script to diagnose posture classification issues.

Shows raw features, z-scores, L2 norm, and classification result in real-time.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.video_source import VideoSource
from src.perception.person_detector import PersonDetector
from src.perception.primary_tracker import PrimaryTracker
from src.perception.pose_estimator import PoseEstimator
from src.perception.posture_features import extract_features
from src.perception.calibration import CalibrationManager
from src.perception.posture_model import PostureClassifier

config_path = str(Path(__file__).parent.parent / "config/pipeline.yaml")
cal_path = "data/calibration_profile.json"

print("=" * 70)
print("POSTURE CLASSIFICATION DEBUG")
print("=" * 70)

# Load modules
video = VideoSource(config_path=config_path)
detector = PersonDetector(config_path=config_path)
tracker = PrimaryTracker(config_path=config_path)
pose = PoseEstimator(config_path=config_path)
classifier = PostureClassifier(config_path=config_path)

# Load calibration
if not Path(cal_path).exists():
    print(f"\nERROR: Calibration profile not found at {cal_path}")
    print("Run: python scripts/run_pipeline.py")
    sys.exit(1)

cal_profile = CalibrationManager.load(cal_path)

print(f"\nCalibration Profile ({cal_profile.n_samples} samples):")
print("=" * 70)
feature_names = [
    "torso_pitch",
    "head_forward_ratio",
    "shoulder_roll",
    "lateral_lean",
    "head_tilt",
    "avg_visibility"
]
print(f"{'Feature':<25} {'Mean':<12} {'Std':<12} {'Notes'}")
print("-" * 70)
for i, name in enumerate(feature_names):
    mean = cal_profile.means[i]
    std = cal_profile.stds[i]
    note = ""
    if std < 0.01:
        note = "⚠️ VERY SMALL STD!"
    elif std < 0.1:
        note = "⚠️ Small std"
    print(f"{name:<25} {mean:<12.4f} {std:<12.6f} {note}")

print("\n" + "=" * 70)
print(f"Classifier: {'MODEL LOADED' if classifier.has_model else 'L2-NORM FALLBACK'}")
print(f"L2-norm threshold: {classifier.threshold_fallback}")
print("=" * 70)

if not video.open():
    print("Error: could not open camera")
    sys.exit(1)

window_name = "Classification Debug"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 800)

print("\nPress 'q' to quit\n")

frame_count = 0

try:
    while True:
        frame = video.read()
        if frame is None:
            continue

        # Detection pipeline
        if frame_count % 3 == 0:
            persons, phones = detector.detect(frame)
            tracker.update(persons, frame)

        primary = tracker.get_primary()
        if primary is None:
            cv2.putText(frame, "No primary person detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            frame_count += 1
            continue

        # Pose estimation
        kp = pose.estimate(frame, primary.bbox)
        if kp is None:
            cv2.putText(frame, "Pose estimation failed", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            frame_count += 1
            continue

        # Extract features
        features = extract_features(kp)
        raw = features.raw_vector

        # Normalize
        normed = CalibrationManager.normalize(features, cal_profile)

        # Classify
        result = classifier.predict(normed)

        # Calculate L2 norm (what the fallback uses)
        l2_norm = float(np.linalg.norm(normed[:5]))

        # --- Display ---
        debug_frame = frame.copy()
        h, w = debug_frame.shape[:2]

        # Classification result
        y = 30
        color = (0, 255, 0) if result.p_bad < 0.5 else (0, 0, 255)
        cv2.putText(debug_frame, f"Classification: {'BAD' if result.p_bad > 0.5 else 'GOOD'}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 35
        cv2.putText(debug_frame, f"p_bad = {result.p_bad:.3f}  [{result.method}]",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 30
        cv2.putText(debug_frame, f"L2 norm (z-scores) = {l2_norm:.3f}  (threshold = {classifier.threshold_fallback})",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Feature table
        y += 40
        cv2.putText(debug_frame, "Feature Analysis:",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y += 30

        cv2.putText(debug_frame, f"{'Feature':<20} {'Raw':<10} {'Z-score':<10}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 25

        for i in range(5):  # Skip avg_visibility for brevity
            name = feature_names[i][:18]
            raw_val = raw[i]
            z_val = normed[i]

            # Color code z-scores
            if abs(z_val) > 2.0:
                z_color = (0, 0, 255)  # Red = large deviation
            elif abs(z_val) > 1.0:
                z_color = (0, 165, 255)  # Orange = moderate
            else:
                z_color = (0, 255, 0)  # Green = small

            cv2.putText(debug_frame, f"{name:<20}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(debug_frame, f"{raw_val:<10.3f}", (210, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(debug_frame, f"{z_val:<10.3f}", (340, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, z_color, 1)
            y += 22

        # Interpretation
        y += 20
        cv2.putText(debug_frame, "Interpretation:",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y += 30

        if l2_norm > classifier.threshold_fallback:
            msg = f"L2 norm ({l2_norm:.2f}) > threshold ({classifier.threshold_fallback}) → BAD posture"
            cv2.putText(debug_frame, msg, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            p_bad_pct = int((l2_norm / classifier.threshold_fallback) * 100)
            msg = f"L2 norm ({l2_norm:.2f}) is {p_bad_pct}% of threshold → p_bad = {result.p_bad:.2f}"
            cv2.putText(debug_frame, msg, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        y += 30
        if np.any(cal_profile.stds[:5] < 0.1):
            cv2.putText(debug_frame, "⚠️ Warning: Very small std in calibration → hypersensitive!",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            y += 25
            cv2.putText(debug_frame, "   Try recalibrating with more natural movement.",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        cv2.imshow(window_name, debug_frame)

        # Console output
        print(f"\rp_bad: {result.p_bad:.3f} | L2: {l2_norm:.3f} | "
              f"z-scores: [{', '.join([f'{z:.2f}' for z in normed[:5]])}]  ",
              end="", flush=True)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame_count += 1

except KeyboardInterrupt:
    print("\n\nInterrupted")
finally:
    video.release()
    pose.close()
    cv2.destroyAllWindows()

print("\nDone.")
