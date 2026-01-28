#!/usr/bin/env python3
"""
Standalone phone detection testing script.

Opens webcam feed, runs YOLOv8 phone detection, and displays annotated video
with detection results.

Usage:
    python scripts/test_phone_detection.py [--config CONFIG_PATH]

Controls:
    q - Quit

Note: First run will download the YOLOv8n model (~6MB).
"""

import argparse
import sys
from pathlib import Path

import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.camera import Camera
from src.perception.phone_detector import PhoneDetector


def main():
    parser = argparse.ArgumentParser(description='Test phone detection')
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
        '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold'
    )
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        config_path = None
    else:
        config_path = str(config_path)

    print("Initializing phone detector (may download model on first run)...")
    try:
        detector = PhoneDetector(
            confidence_threshold=args.confidence,
            config_path=config_path
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("Install ultralytics: pip install ultralytics")
        return 1

    print("Opening camera...")
    camera = Camera(device_id=args.camera)
    if not camera.open():
        print("Error: Could not open camera")
        return 1

    print("\nPhone Detection Test")
    print("=" * 40)
    print("Controls:")
    print("  q - Quit")
    print("=" * 40)
    print("\nTry showing your phone to the camera!")
    print("Position phone in different locations to test detection.\n")

    window_name = 'Phone Detection Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    detection_count = 0

    try:
        while True:
            # Read frame
            frame = camera.read()
            if frame is None:
                print("Warning: Could not read frame")
                continue

            frame_count += 1

            # Detect phone
            result = detector.detect(frame)

            # Draw detections and status
            annotated_frame = detector.draw_detections(frame, result)
            annotated_frame = detector.draw_status_overlay(annotated_frame, result, y_offset=30)

            # Add FPS and detection stats
            if result.phone_detected:
                detection_count += 1

            detection_rate = (detection_count / frame_count * 100) if frame_count > 0 else 0

            cv2.putText(
                annotated_frame,
                f"Detection rate: {detection_rate:.1f}%",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            # Add instructions
            cv2.putText(
                annotated_frame,
                "Press 'q' to quit | Show phone to test detection",
                (10, annotated_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (128, 128, 128), 1
            )

            # Display
            cv2.imshow(window_name, annotated_frame)

            # Print to console
            if result.phone_detected:
                status = "IN HAND" if result.in_hand else "VISIBLE"
                print(f"\r\033[93mPhone {status}\033[0m | "
                      f"Confidence: {result.confidence:.2f} | "
                      f"Detection rate: {detection_rate:.1f}%",
                      end='', flush=True)
            else:
                print(f"\r\033[92mNo phone detected\033[0m | "
                      f"Detection rate: {detection_rate:.1f}%              ",
                      end='', flush=True)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\nSession stats: {detection_count}/{frame_count} frames with phone detected")
    return 0


if __name__ == '__main__':
    sys.exit(main())
