#!/usr/bin/env python3
"""
Full perception pipeline demo.

Runs all detectors (posture, phone, gaze) and fuses them into
a combined focus state estimation.

Usage:
    python scripts/run_perception.py [--config CONFIG_PATH]

Controls:
    q - Quit
    p - Toggle posture overlay
    g - Toggle gaze overlay
    h - Toggle phone detection
    d - Toggle debug info
    r - Reset focus estimator
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.camera import Camera
from src.perception.posture_detector import PostureDetector
from src.perception.phone_detector import PhoneDetector
from src.perception.gaze_tracker import GazeTracker
from src.perception.focus_estimator import FocusEstimator, FocusState


def main():
    parser = argparse.ArgumentParser(description='Run full perception pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='config/thresholds.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID'
    )
    parser.add_argument(
        '--no-phone',
        action='store_true',
        help='Disable phone detection (faster startup)'
    )
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        config_path = None
    else:
        config_path = str(config_path)

    print("=" * 50)
    print("Desk Buddy - Full Perception Pipeline")
    print("=" * 50)

    # Initialize detectors
    print("\nInitializing detectors...")

    print("  - Posture detector...")
    posture_detector = PostureDetector(config_path=config_path)

    phone_detector = None
    if not args.no_phone:
        print("  - Phone detector (may download model)...")
        try:
            phone_detector = PhoneDetector(config_path=config_path)
        except ImportError as e:
            print(f"    Warning: {e}")
            print("    Phone detection disabled. Install: pip install ultralytics")

    print("  - Gaze tracker...")
    gaze_tracker = GazeTracker(config_path=config_path)

    print("  - Focus estimator...")
    focus_estimator = FocusEstimator(config_path=config_path)

    print("\nOpening camera...")
    camera = Camera(device_id=args.camera)
    if not camera.open():
        print("Error: Could not open camera")
        return 1

    print("\nControls:")
    print("  q - Quit")
    print("  p - Toggle posture overlay")
    print("  g - Toggle gaze overlay")
    print("  h - Toggle phone detection")
    print("  d - Toggle debug info")
    print("  r - Reset focus estimator")
    print("=" * 50)

    window_name = 'Desk Buddy - Perception Pipeline'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    # Toggle states
    show_posture = True
    show_gaze = True
    show_phone = True
    show_debug = False

    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    try:
        while True:
            frame_start = time.time()

            # Read frame
            frame = camera.read()
            if frame is None:
                print("Warning: Could not read frame")
                continue

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run detectors
            posture_result = posture_detector.detect(rgb_frame)

            phone_result = None
            if phone_detector and show_phone:
                phone_result = phone_detector.detect(frame)

            gaze_result = gaze_tracker.detect(rgb_frame)

            # Fuse into focus state
            focus_result = focus_estimator.estimate(
                posture=posture_result,
                phone=phone_result,
                gaze=gaze_result
            )

            # Start with original frame
            annotated_frame = frame.copy()

            # Draw posture landmarks if enabled
            if show_posture:
                annotated_frame = posture_detector.draw_landmarks(
                    annotated_frame, posture_result, draw_connections=True
                )

            # Draw gaze landmarks if enabled
            if show_gaze and gaze_result.face_detected:
                annotated_frame = gaze_tracker.draw_landmarks(
                    annotated_frame, gaze_result, draw_contours=True
                )
                annotated_frame = gaze_tracker.draw_head_pose_axes(
                    annotated_frame, gaze_result
                )

            # Draw phone detections if enabled
            if phone_result and show_phone:
                annotated_frame = phone_detector.draw_detections(
                    annotated_frame, phone_result
                )

            # Draw full status overlay
            annotated_frame = focus_estimator.draw_full_overlay(
                annotated_frame,
                focus_result,
                posture_result=posture_result,
                phone_result=phone_result,
                gaze_result=gaze_result
            )

            # Calculate and display FPS
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()

            cv2.putText(
                annotated_frame,
                f"FPS: {current_fps:.1f}",
                (annotated_frame.shape[1] - 100, annotated_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            # Debug info if enabled
            if show_debug:
                debug_y = 150
                cv2.putText(
                    annotated_frame,
                    f"Posture: {posture_result.state.value} ({posture_result.confidence:.2f})",
                    (10, debug_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
                )
                debug_y += 15
                if gaze_result.face_detected:
                    pitch, yaw, roll = gaze_result.head_pose
                    cv2.putText(
                        annotated_frame,
                        f"Head: P={pitch:.0f} Y={yaw:.0f} R={roll:.0f}",
                        (10, debug_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
                    )
                debug_y += 15
                if phone_result:
                    cv2.putText(
                        annotated_frame,
                        f"Phone: {'Yes' if phone_result.phone_detected else 'No'} "
                        f"({'hand' if phone_result.in_hand else '-'})",
                        (10, debug_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
                    )
                debug_y += 15
                cv2.putText(
                    annotated_frame,
                    f"Factors: {', '.join(focus_result.contributing_factors[:3])}",
                    (10, debug_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
                )

            # Toggle indicators
            toggles = f"[P]osture:{'ON' if show_posture else 'OFF'} " \
                      f"[G]aze:{'ON' if show_gaze else 'OFF'} " \
                      f"[H]Phone:{'ON' if show_phone else 'OFF'} " \
                      f"[D]ebug:{'ON' if show_debug else 'OFF'}"
            cv2.putText(
                annotated_frame,
                toggles,
                (10, annotated_frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1
            )

            cv2.putText(
                annotated_frame,
                "Press 'q' to quit | 'r' to reset",
                (10, annotated_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1
            )

            # Display
            cv2.imshow(window_name, annotated_frame)

            # Console output
            focus_str = focus_result.state.value.upper()
            if focus_result.state == FocusState.FOCUSED:
                focus_str = f"\033[92m{focus_str}\033[0m"
            elif focus_result.state == FocusState.DISTRACTED:
                focus_str = f"\033[91m{focus_str}\033[0m"
            else:
                focus_str = f"\033[90m{focus_str}\033[0m"

            duration = focus_result.duration_in_state / 30.0
            print(f"\rFocus: {focus_str:<25} | "
                  f"Duration: {duration:>5.1f}s | "
                  f"FPS: {current_fps:>5.1f} | "
                  f"Factors: {', '.join(focus_result.contributing_factors[:2]):<30}",
                  end='', flush=True)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('p'):
                show_posture = not show_posture
                print(f"\nPosture overlay: {'ON' if show_posture else 'OFF'}")
            elif key == ord('g'):
                show_gaze = not show_gaze
                print(f"\nGaze overlay: {'ON' if show_gaze else 'OFF'}")
            elif key == ord('h'):
                show_phone = not show_phone
                print(f"\nPhone detection: {'ON' if show_phone else 'OFF'}")
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"\nDebug info: {'ON' if show_debug else 'OFF'}")
            elif key == ord('r'):
                focus_estimator.reset()
                print("\nFocus estimator reset")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        camera.release()
        posture_detector.close()
        gaze_tracker.close()
        cv2.destroyAllWindows()

    # Print session summary
    print("\n" + "=" * 50)
    print("Session Summary")
    print("=" * 50)
    events = focus_estimator.get_events()
    state_changes = [e for e in events if e['type'] == 'state_change']
    print(f"State changes: {len(state_changes)}")
    for event in state_changes[-5:]:  # Show last 5
        data = event['data']
        print(f"  {data['from']} -> {data['to']} "
              f"(after {data['duration']/30:.1f}s)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
