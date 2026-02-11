#!/usr/bin/env python3
"""
Posture Pipeline 2.0 — main demo loop.

Dual-rate loop: YOLO person detection every Nth frame, MediaPipe pose
estimation every frame on the primary person's crop.  Includes a
10-second calibration phase on startup, followed by continuous posture
classification with EWMA-smoothed state output.

Usage:
    python scripts/run_pipeline.py [--config CONFIG]
    python scripts/run_pipeline.py --skip-calibration --calibration data/calibration_profile.json
    python scripts/run_pipeline.py --use-cnn  # Use trained CNN model

Controls:
    q - Quit
    c - Re-calibrate
    d - Toggle debug overlay
    r - Reset state machine
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Allow imports from project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.video_source import VideoSource
from src.perception.person_detector import PersonDetector, BBox
from src.perception.primary_tracker import PrimaryTracker
from src.perception.pose_estimator import PoseEstimator, PoseKeypoints
from src.perception.posture_features import extract_features, PostureFeatures
from src.perception.calibration import CalibrationManager, CalibrationProfile
from src.perception.posture_model import PostureClassifier
from src.perception.posture_state import PostureStateMachine, PostureLabel
from src.perception.skeleton_renderer import render_skeleton, render_skeleton_depth
from src.perception.posture_cnn import PostureCNNClassifier, load_model as load_cnn_model

# MediaPipe pose connections for drawing skeleton (33 landmarks)
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),  # left eye to ear
    (0, 4), (4, 5), (5, 6), (6, 8),  # right eye to ear
    # Torso
    (11, 12),  # shoulders
    (11, 23), (12, 24),  # shoulders to hips
    (23, 24),  # hips
    # Arms
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    # Legs
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
]


def draw_skeleton(frame: np.ndarray, kp: PoseKeypoints) -> np.ndarray:
    """Draw pose landmarks and connections on the frame (MediaPipe 33-landmark format)."""
    lm = kp.landmarks  # (33, 4) - x, y, z, visibility
    for i in range(33):
        if lm[i, 3] < 0.3:  # visibility threshold
            continue
        x, y = int(lm[i, 0]), int(lm[i, 1])
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    for a, b in POSE_CONNECTIONS:
        if lm[a, 3] < 0.3 or lm[b, 3] < 0.3:
            continue
        pt1 = (int(lm[a, 0]), int(lm[a, 1]))
        pt2 = (int(lm[b, 0]), int(lm[b, 1]))
        cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
    return frame


def draw_bbox(frame: np.ndarray, bbox: BBox, label: str, color: tuple) -> np.ndarray:
    cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
    cv2.putText(frame, label, (bbox.x1, bbox.y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def draw_status(frame: np.ndarray, state_label: str, color: tuple,
                smoothed: float, raw: float, features: PostureFeatures | None,
                fps: float, debug: bool, method: str,
                calibrating: bool = False, cal_remaining: float = 0.0) -> np.ndarray:
    """Draw status overlay in top-left."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 30

    if calibrating:
        cv2.putText(frame, f"CALIBRATING... {cal_remaining:.1f}s remaining",
                     (10, y), font, 0.7, (0, 255, 255), 2)
        y += 30
        cv2.putText(frame, "Sit in your normal GOOD posture",
                     (10, y), font, 0.6, (255, 255, 255), 1)
        return frame

    cv2.putText(frame, f"Posture: {state_label}", (10, y), font, 0.7, color, 2)
    y += 25
    cv2.putText(frame, f"p_bad: {raw:.2f}  smoothed: {smoothed:.2f}  [{method}]",
                 (10, y), font, 0.45, (255, 255, 255), 1)

    if debug and features is not None:
        y += 20
        cv2.putText(frame, f"torso_pitch: {features.torso_pitch:.1f}  "
                           f"head_fwd: {features.head_forward_ratio:.2f}  "
                           f"shldr_roll: {features.shoulder_roll:.1f}",
                     (10, y), font, 0.4, (200, 200, 200), 1)
        y += 18
        cv2.putText(frame, f"lat_lean: {features.lateral_lean:.2f}  "
                           f"head_tilt: {features.head_tilt:.1f}  "
                           f"vis: {features.avg_visibility:.2f}",
                     (10, y), font, 0.4, (200, 200, 200), 1)
        y += 18
        fwd_color = (0, 0, 255) if features.forward_lean_z < -0.05 else (200, 200, 200)
        cv2.putText(frame, f"fwd_lean_z: {features.forward_lean_z:.3f}",
                     (10, y), font, 0.4, fwd_color, 1)

    # FPS in bottom-right.
    cv2.putText(frame, f"FPS: {fps:.1f}",
                 (frame.shape[1] - 100, frame.shape[0] - 10),
                 font, 0.5, (255, 255, 255), 1)

    # Controls hint at bottom.
    cv2.putText(frame, "q:quit  c:calibrate  d:debug  r:reset",
                 (10, frame.shape[0] - 10), font, 0.4, (128, 128, 128), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Posture Pipeline 2.0")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--calibration", default=None,
                        help="Path to existing calibration JSON")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip calibration (requires --calibration)")
    parser.add_argument("--use-cnn", action="store_true",
                        help="Use trained CNN model instead of LogisticRegression")
    parser.add_argument("--cnn-model", default="data/trained_models/posture_cnn.pt",
                        help="Path to CNN model checkpoint")
    parser.add_argument("--depth-images", action="store_true",
                        help="Use depth-encoded skeleton images (requires CNN trained with --depth-images)")
    parser.add_argument("--debug-cnn", action="store_true",
                        help="Show CNN debug info: skeleton image, raw features, raw probability")
    args = parser.parse_args()

    config_path = str(Path(__file__).parent.parent / args.config)

    print("=" * 50)
    print("Desk Buddy — Posture Pipeline 2.0")
    print("=" * 50)

    # --- Initialise modules ---
    print("\nInitialising modules...")
    video = VideoSource(config_path=config_path)
    detector = PersonDetector(config_path=config_path)
    tracker = PrimaryTracker(config_path=config_path)
    pose = PoseEstimator(config_path=config_path)
    classifier = PostureClassifier(config_path=config_path)
    state_machine = PostureStateMachine(config_path=config_path)
    cal_manager = CalibrationManager(config_path=config_path)

    # --- Load CNN model if requested ---
    cnn_model = None
    cnn_device = "cpu"
    cnn_threshold = 0.55  # Default threshold (raised to handle ~0.49 clustering from overfitting)
    use_cnn = args.use_cnn

    cnn_path = Path(__file__).parent.parent / args.cnn_model
    if use_cnn:
        if cnn_path.exists():
            print(f"Loading CNN model from {cnn_path}...")
            cnn_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            cnn_model, cnn_metadata = load_cnn_model(str(cnn_path), device=cnn_device)
            cnn_model.eval()
            print(f"  CNN loaded on {cnn_device}")
            print(f"  Model config: channels={cnn_model.cnn.conv1.in_channels}, "
                  f"use_features={cnn_model.use_features}")
            if cnn_metadata:
                epochs = cnn_metadata.get('epochs', '?')
                acc = cnn_metadata.get('test_accuracy')
                acc_str = f"{acc:.3f}" if acc is not None else "?"
                print(f"  Trained with: {epochs} epochs, acc={acc_str}")
                # Load optimal threshold from metadata (but enforce minimum of 0.55 to handle clustering)
                cnn_threshold = max(0.55, cnn_metadata.get('optimal_threshold', 0.55))
                target_recall = cnn_metadata.get('target_recall', 0.9)
                print(f"  Using threshold: {cnn_threshold:.2f} (optimized for {target_recall:.0%} recall)")
        else:
            print(f"Warning: CNN model not found at {cnn_path}, falling back to LogisticRegression")
            use_cnn = False
    elif cnn_path.exists():
        print(f"Note: CNN model available at {cnn_path} (use --use-cnn to enable)")

    use_depth_images = args.depth_images
    if use_cnn and cnn_model is not None:
        # Check if model expects depth images (3 channels) or grayscale (1 channel)
        in_channels = cnn_model.cnn.conv1.in_channels
        if in_channels == 3 and not use_depth_images:
            print("  Note: Model expects 3-channel input, enabling depth images")
            use_depth_images = True
        elif in_channels == 1 and use_depth_images:
            print("  Note: Model expects 1-channel input, disabling depth images")
            use_depth_images = False

    # Load pipeline settings.
    import yaml as _yaml
    with open(config_path) as f:
        _cfg = _yaml.safe_load(f) or {}
    detect_every_n = _cfg.get("pipeline", {}).get("detect_every_n", 5)

    # --- Open camera ---
    print("Opening camera...")
    if not video.open():
        print("Error: could not open camera")
        return 1

    window_name = "Posture Pipeline 2.0"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    # --- Calibration phase ---
    cal_profile: CalibrationProfile | None = None
    calibrating = False

    if args.skip_calibration and args.calibration:
        cal_profile = CalibrationManager.load(args.calibration)
        print(f"Loaded calibration ({cal_profile.n_samples} samples)")
    elif args.calibration and Path(args.calibration).exists():
        cal_profile = CalibrationManager.load(args.calibration)
        print(f"Loaded calibration ({cal_profile.n_samples} samples)")
    else:
        calibrating = True
        cal_manager.start()
        print("Starting 10-second calibration — sit with GOOD posture.")

    # --- Main loop state ---
    frame_count = 0
    show_debug = False
    fps_t0 = time.time()
    fps_count = 0
    current_fps = 0.0

    persons = []
    phones = []
    last_features: PostureFeatures | None = None

    try:
        while True:
            frame = video.read()
            if frame is None:
                continue

            # --- Detection (every N frames) ---
            if frame_count % detect_every_n == 0:
                persons, phones = detector.detect(frame)
                tracker.update(persons, frame)

            primary = tracker.get_primary()

            # --- Pose on primary person ---
            kp: PoseKeypoints | None = None
            features: PostureFeatures | None = None
            if primary is not None:
                kp = pose.estimate(frame, primary.bbox)
                if kp is not None:
                    features = extract_features(kp)
                    last_features = features

            # --- Calibration or classification ---
            state_label = "---"
            state_color = (128, 128, 128)
            raw_p = 0.0
            smoothed_p = 0.0
            method = ""

            if calibrating:
                if features is not None:
                    cal_manager.add_sample(features)
                cal_remaining = max(0, cal_manager.duration_seconds - cal_manager.elapsed())
                if cal_manager.is_ready():
                    cal_profile = cal_manager.finish()
                    CalibrationManager.save(cal_profile, cal_manager.save_path)
                    print(f"\nCalibration complete ({cal_profile.n_samples} samples) "
                          f"— saved to {cal_manager.save_path}")
                    calibrating = False
            elif cal_profile is not None and features is not None:
                # --- CNN classification ---
                if use_cnn and cnn_model is not None and kp is not None:
                    # Render skeleton image
                    if use_depth_images:
                        skel_img = render_skeleton_depth(
                            kp.landmarks, output_size=224, upper_body_only=True
                        )
                        # Convert BGR to RGB, then to tensor (C, H, W)
                        skel_tensor = torch.from_numpy(skel_img).permute(2, 0, 1).float() / 255.0
                    else:
                        skel_img = render_skeleton(
                            kp.landmarks, output_size=224, upper_body_only=True
                        )
                        # Add channel dimension: (H, W) -> (1, H, W)
                        skel_tensor = torch.from_numpy(skel_img).unsqueeze(0).float() / 255.0

                    # Add batch dimension: (C, H, W) -> (1, C, H, W)
                    skel_tensor = skel_tensor.unsqueeze(0).to(cnn_device)

                    # Prepare geometric features if model uses them
                    feat_tensor = None
                    if cnn_model.use_features:
                        feat_array = np.array([
                            features.torso_pitch,
                            features.head_forward_ratio,
                            features.shoulder_roll,
                            features.lateral_lean,
                            features.head_tilt,
                            features.avg_visibility,
                            features.forward_lean_z,
                        ], dtype=np.float32)
                        feat_tensor = torch.from_numpy(feat_array).unsqueeze(0).to(cnn_device)

                    # Run inference
                    with torch.no_grad():
                        p_bad = cnn_model.predict_proba(skel_tensor, feat_tensor)
                        raw_p_before_rescale = p_bad.item()
                        raw_p = raw_p_before_rescale

                    # Debug: show skeleton image and raw values
                    if args.debug_cnn:
                        # Show skeleton image
                        if use_depth_images:
                            cv2.imshow("CNN Skeleton Input", skel_img)
                        else:
                            cv2.imshow("CNN Skeleton Input", skel_img)

                        # Print raw values (before rescaling)
                        print(f"\n[CNN DEBUG] raw_prob={raw_p_before_rescale:.4f} | "
                              f"pitch={features.torso_pitch:.1f} head_fwd={features.head_forward_ratio:.2f} "
                              f"roll={features.shoulder_roll:.1f} lean={features.lateral_lean:.2f} "
                              f"tilt={features.head_tilt:.1f} vis={features.avg_visibility:.2f} "
                              f"fwd_z={features.forward_lean_z:.3f}")

                    # Adjust probability based on optimal threshold
                    # If threshold < 0.5, we shift probabilities up so that
                    # the threshold maps to 0.5 (for state machine compatibility)
                    if cnn_threshold != 0.5:
                        # Linear rescaling: threshold -> 0.5
                        if raw_p < cnn_threshold:
                            # Scale [0, threshold] to [0, 0.5]
                            raw_p = raw_p * (0.5 / cnn_threshold)
                        else:
                            # Scale [threshold, 1] to [0.5, 1]
                            raw_p = 0.5 + (raw_p - cnn_threshold) * (0.5 / (1 - cnn_threshold))

                    method = f"CNN@{cnn_threshold:.2f}"

                # --- LogisticRegression fallback ---
                else:
                    normed = CalibrationManager.normalize(features, cal_profile)
                    cls_result = classifier.predict(normed)
                    raw_p = cls_result.p_bad
                    method = cls_result.method

                sm_result = state_machine.update(raw_p, features.avg_visibility)
                smoothed_p = sm_result.smoothed_prob
                state_label = sm_result.state.value.upper()
                if sm_result.state == PostureLabel.GOOD:
                    state_color = (0, 255, 0)
                elif sm_result.state == PostureLabel.BAD:
                    state_color = (0, 0, 255)
                else:
                    state_color = (128, 128, 128)

            # --- Draw ---
            annotated = frame.copy()

            # Draw all tracked person bboxes.
            for tp in tracker._tracked:
                color = (0, 255, 0) if tp.is_primary else (200, 200, 200)
                lbl = f"ID {tp.tracker_id}" + (" [PRIMARY]" if tp.is_primary else "")
                annotated = draw_bbox(annotated, tp.bbox, lbl, color)

            # Draw phone bboxes.
            for ph in phones:
                annotated = draw_bbox(annotated, ph, f"Phone {ph.confidence:.2f}", (0, 0, 255))

            # Draw skeleton on primary.
            if kp is not None:
                annotated = draw_skeleton(annotated, kp)

            # Status overlay.
            cal_remaining = 0.0
            if calibrating:
                cal_remaining = max(0, cal_manager.duration_seconds - cal_manager.elapsed())
            annotated = draw_status(
                annotated, state_label, state_color,
                smoothed_p, raw_p, last_features,
                current_fps, show_debug, method,
                calibrating=calibrating, cal_remaining=cal_remaining,
            )

            cv2.imshow(window_name, annotated)

            # --- FPS ---
            fps_count += 1
            elapsed = time.time() - fps_t0
            if elapsed >= 1.0:
                current_fps = fps_count / elapsed
                fps_count = 0
                fps_t0 = time.time()

            # --- Console ---
            if not calibrating:
                focus_str = state_label
                if state_label == "GOOD":
                    focus_str = f"\033[92m{state_label}\033[0m"
                elif state_label == "BAD":
                    focus_str = f"\033[91m{state_label}\033[0m"
                print(f"\rPosture: {focus_str:<20} | "
                      f"p_bad: {raw_p:.2f} | "
                      f"smooth: {smoothed_p:.2f} | "
                      f"FPS: {current_fps:.1f}  ",
                      end="", flush=True)

            # --- Keys ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQuitting...")
                break
            elif key == ord("c"):
                calibrating = True
                cal_manager.start()
                state_machine.reset()
                print("\nRe-calibrating — sit with GOOD posture.")
            elif key == ord("d"):
                show_debug = not show_debug
                print(f"\nDebug: {'ON' if show_debug else 'OFF'}")
            elif key == ord("r"):
                state_machine.reset()
                print("\nState machine reset")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        video.release()
        pose.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
