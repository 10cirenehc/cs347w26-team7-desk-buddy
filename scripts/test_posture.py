#!/usr/bin/env python3
"""
Standalone posture detection testing script.

Opens webcam feed, runs posture detection, and displays annotated video
with posture metrics overlay.

Usage:
    python scripts/test_posture.py [--config CONFIG_PATH]

Controls:
    q - Quit
    r - Reset metrics display
"""

import argparse
import sys
from pathlib import Path

import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.camera import Camera
from src.perception.posture_detector import PostureDetector, PostureState


def main():
    parser = argparse.ArgumentParser(description='Test posture detection')
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

    print("Initializing posture detector...")
    detector = PostureDetector(config_path=config_path)

    print("Opening camera...")
    camera = Camera(device_id=args.camera)
    if not camera.open():
        print("Error: Could not open camera")
        return 1

    print("\nPosture Detection Test")
    print("=" * 40)
    print("Controls:")
    print("  q - Quit")
    print("  r - Reset")
    print("=" * 40)

    window_name = 'Posture Detection Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            # Read frame
            frame = camera.read()
            if frame is None:
                print("Warning: Could not read frame")
                continue

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect posture
            result = detector.detect(rgb_frame)

            # Draw landmarks and overlay
            annotated_frame = detector.draw_landmarks(frame, result)
            annotated_frame = detector.draw_metrics_overlay(annotated_frame, result)

            # Add instructions
            cv2.putText(
                annotated_frame,
                "Press 'q' to quit",
                (10, annotated_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (128, 128, 128), 1
            )

            # Display
            cv2.imshow(window_name, annotated_frame)

            # Print to console periodically
            if result.pose_detected:
                state_str = result.state.value.upper()
                if result.state == PostureState.GOOD:
                    state_str = f"\033[92m{state_str}\033[0m"  # Green
                else:
                    state_str = f"\033[91m{state_str}\033[0m"  # Red
                print(f"\rPosture: {state_str:<20} | "
                      f"Shoulder: {result.metrics.get('shoulder_tilt_deg', 0):>6.1f}° | "
                      f"Forward: {result.metrics.get('forward_head_distance', 0):>6.3f} | "
                      f"Torso: {result.metrics.get('torso_angle_deg', 0):>6.1f}°",
                      end='', flush=True)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                print("\nReset")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    sys.exit(main())
