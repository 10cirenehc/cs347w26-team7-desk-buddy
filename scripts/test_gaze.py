#!/usr/bin/env python3
"""
Standalone gaze/head pose tracking testing script.

Opens webcam feed, runs face mesh detection with head pose estimation,
and displays annotated video with gaze metrics overlay.

Usage:
    python scripts/test_gaze.py [--config CONFIG_PATH]

Controls:
    q - Quit
    m - Toggle mesh visibility
    a - Toggle pose axes
"""

import argparse
import sys
from pathlib import Path

import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.video_source import VideoSource
from src.perception.gaze_tracker import GazeTracker, AttentionState


def main():
    parser = argparse.ArgumentParser(description='Test gaze tracking')
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
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        config_path = None
    else:
        config_path = str(config_path)

    print("Initializing gaze tracker...")
    tracker = GazeTracker(config_path=config_path)

    print("Opening camera...")
    camera = VideoSource(device_id=args.camera)
    if not camera.open():
        print("Error: Could not open camera")
        return 1

    print("\nGaze/Head Pose Tracking Test")
    print("=" * 40)
    print("Controls:")
    print("  q - Quit")
    print("  m - Toggle mesh visibility")
    print("  a - Toggle pose axes")
    print("=" * 40)
    print("\nTry looking left, right, up, and down!")
    print("Watch how the attention state changes.\n")

    window_name = 'Gaze Tracking Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    show_mesh = True
    show_axes = True

    try:
        while True:
            # Read frame
            frame = camera.read()
            if frame is None:
                print("Warning: Could not read frame")
                continue

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect gaze
            result = tracker.detect(rgb_frame)

            # Start with original frame
            annotated_frame = frame.copy()

            # Draw face mesh if enabled
            if show_mesh and result.face_detected:
                annotated_frame = tracker.draw_landmarks(annotated_frame, result)

            # Draw pose axes if enabled
            if show_axes and result.face_detected:
                annotated_frame = tracker.draw_head_pose_axes(annotated_frame, result)

            # Draw metrics overlay
            annotated_frame = tracker.draw_metrics_overlay(annotated_frame, result, y_offset=30)

            # Add toggle status
            toggle_text = f"Mesh: {'ON' if show_mesh else 'OFF'} | Axes: {'ON' if show_axes else 'OFF'}"
            cv2.putText(
                annotated_frame,
                toggle_text,
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (128, 128, 128), 1
            )

            # Add instructions
            cv2.putText(
                annotated_frame,
                "Press 'q' to quit | 'm' toggle mesh | 'a' toggle axes",
                (10, annotated_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (128, 128, 128), 1
            )

            # Display
            cv2.imshow(window_name, annotated_frame)

            # Print to console
            if result.face_detected:
                pitch, yaw, roll = result.head_pose
                state = result.attention_state

                if state == AttentionState.FOCUSED:
                    state_str = f"\033[92m{state.value.upper()}\033[0m"
                elif state == AttentionState.LOOKING_AWAY:
                    state_str = f"\033[91m{state.value.upper()}\033[0m"
                else:
                    state_str = f"\033[93m{state.value.upper()}\033[0m"

                print(f"\rAttention: {state_str:<30} | "
                      f"Pitch: {pitch:>6.1f}° | "
                      f"Yaw: {yaw:>6.1f}° | "
                      f"Roll: {roll:>6.1f}°",
                      end='', flush=True)
            else:
                print(f"\r\033[90mNo face detected\033[0m"
                      f"                                              ",
                      end='', flush=True)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('m'):
                show_mesh = not show_mesh
                print(f"\nMesh: {'ON' if show_mesh else 'OFF'}")
            elif key == ord('a'):
                show_axes = not show_axes
                print(f"\nAxes: {'ON' if show_axes else 'OFF'}")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    sys.exit(main())
