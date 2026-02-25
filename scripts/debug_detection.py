#!/usr/bin/env python3
"""
Debug script to diagnose detection issues at different angles.

Shows what each stage of the pipeline is seeing.
"""

import sys
from pathlib import Path
import cv2
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.video_source import VideoSource
from src.perception.person_detector import PersonDetector
from src.perception.primary_tracker import PrimaryTracker
from src.perception.pose_estimator import PoseEstimator

config_path = str(Path(__file__).parent.parent / "config/pipeline.yaml")

print("=" * 60)
print("DETECTION DEBUG - Testing each pipeline stage")
print("=" * 60)

video = VideoSource(config_path=config_path)
detector = PersonDetector(config_path=config_path)
tracker = PrimaryTracker(config_path=config_path)
pose = PoseEstimator(config_path=config_path)

if not video.open():
    print("Error: could not open camera")
    sys.exit(1)

window_name = "Detection Debug"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

frame_count = 0
fps_t0 = time.time()
fps_count = 0
current_fps = 0.0

print("\nPosition yourself at different angles and watch the debug output.")
print("Press 'q' to quit\n")

try:
    while True:
        frame = video.read()
        if frame is None:
            continue

        # Always run detection for debug purposes
        persons, phones = detector.detect(frame)
        tracked = tracker.update(persons, frame)
        primary = tracker.get_primary()

        # Try pose estimation if we have a primary
        kp = None
        if primary is not None:
            kp = pose.estimate(frame, primary.bbox)

        # --- Draw debug info ---
        debug_frame = frame.copy()
        h, w = debug_frame.shape[:2]

        # Stage 1: Person detection
        y_offset = 30
        status_color = (0, 255, 0) if len(persons) > 0 else (0, 0, 255)
        cv2.putText(debug_frame, f"1. YOLOv8 Person Detection: {len(persons)} person(s)",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        y_offset += 30

        # Show detection confidences
        for i, p in enumerate(persons):
            conf_color = (0, 255, 0) if p.confidence >= 0.4 else (0, 165, 255)
            cv2.putText(debug_frame, f"   Person {i+1}: conf={p.confidence:.3f}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
            # Draw bbox
            cv2.rectangle(debug_frame, (p.x1, p.y1), (p.x2, p.y2), conf_color, 2)
            cv2.putText(debug_frame, f"{p.confidence:.2f}", (p.x1, p.y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)
            y_offset += 25

        y_offset += 10

        # Stage 2: Tracking
        track_color = (0, 255, 0) if primary is not None else (0, 0, 255)
        cv2.putText(debug_frame, f"2. Primary Tracker: {'LOCKED' if primary else 'NO PRIMARY'}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 2)
        y_offset += 30

        if primary:
            cv2.putText(debug_frame, f"   ID: {primary.tracker_id}, bbox area: {primary.bbox.area}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Draw primary bbox in green
            cv2.rectangle(debug_frame, (primary.bbox.x1, primary.bbox.y1),
                         (primary.bbox.x2, primary.bbox.y2), (0, 255, 0), 3)
            cv2.putText(debug_frame, f"PRIMARY {primary.tracker_id} ({primary.bbox.area}px)",
                       (primary.bbox.x1, primary.bbox.y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 25

        # Show ALL tracked people with their sizes
        if len(tracker._tracked) > 1:
            y_offset += 5
            cv2.putText(debug_frame, "   All tracked:",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20
            for tp in sorted(tracker._tracked, key=lambda t: t.bbox.area, reverse=True):
                is_prim = " ← PRIMARY" if tp.is_primary else ""
                cv2.putText(debug_frame, f"      ID {tp.tracker_id}: {tp.bbox.area}px{is_prim}",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                # Draw bbox for non-primary
                if not tp.is_primary:
                    cv2.rectangle(debug_frame, (tp.bbox.x1, tp.bbox.y1),
                                 (tp.bbox.x2, tp.bbox.y2), (100, 100, 100), 2)
                    cv2.putText(debug_frame, f"ID {tp.tracker_id}",
                               (tp.bbox.x1, tp.bbox.y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                y_offset += 20

        y_offset += 10

        # Stage 3: Pose estimation
        pose_color = (0, 255, 0) if kp is not None else (0, 0, 255)
        cv2.putText(debug_frame, f"3. MediaPipe Pose: {'DETECTED' if kp else 'FAILED'}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
        y_offset += 30

        if kp:
            cv2.putText(debug_frame, f"   Avg visibility: {kp.avg_visibility:.3f}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

            # Draw key landmarks
            lm = kp.landmarks
            key_indices = [0, 7, 8, 11, 12, 23, 24]  # nose, ears, shoulders, hips
            for idx in key_indices:
                if lm[idx, 3] > 0.3:  # visibility threshold
                    x, y = int(lm[idx, 0]), int(lm[idx, 1])
                    vis = lm[idx, 3]
                    color = (0, 255, 0) if vis > 0.5 else (0, 165, 255)
                    cv2.circle(debug_frame, (x, y), 5, color, -1)
                    cv2.putText(debug_frame, f"{vis:.2f}", (x+7, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Stage 4: Visibility threshold check
        y_offset += 10
        if kp:
            vis_ok = kp.avg_visibility >= 0.3
            vis_color = (0, 255, 0) if vis_ok else (0, 165, 255)
            threshold_status = "PASS" if vis_ok else "BELOW THRESHOLD (→ UNKNOWN)"
            cv2.putText(debug_frame, f"4. Visibility Check: {threshold_status}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vis_color, 2)
        else:
            cv2.putText(debug_frame, f"4. Visibility Check: N/A (no pose)",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

        # FPS
        cv2.putText(debug_frame, f"FPS: {current_fps:.1f}",
                    (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(window_name, debug_frame)

        # Console output
        vis_str = f"{kp.avg_visibility:.3f}" if kp else "0.000"
        print(f"\rFrame {frame_count:4d} | "
              f"YOLO: {len(persons)} persons | "
              f"Track: {'PRIMARY' if primary else 'none    '} | "
              f"Pose: {'OK' if kp else 'FAIL'} | "
              f"Vis: {vis_str} | "
              f"FPS: {current_fps:.1f}  ",
              end="", flush=True)

        # FPS calculation
        fps_count += 1
        elapsed = time.time() - fps_t0
        if elapsed >= 1.0:
            current_fps = fps_count / elapsed
            fps_count = 0
            fps_t0 = time.time()

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
