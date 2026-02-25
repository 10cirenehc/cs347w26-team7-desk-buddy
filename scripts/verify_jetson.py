#!/usr/bin/env python3
"""
Desk Buddy — Jetson AGX Orin component verification script.

Runs each subsystem independently and reports pass/fail.
Usage: python3 scripts/verify_jetson.py
"""

import sys
import time
from pathlib import Path

# Ensure repo root is on the path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results: list[tuple[str, str, str]] = []


def record(name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f"  —  {detail}" if detail else ""))


def record_skip(name: str, reason: str = ""):
    results.append((name, SKIP, reason))
    print(f"  [{SKIP}] {name}" + (f"  —  {reason}" if reason else ""))


# ─── Tests ───────────────────────────────────────────────────────

def test_camera():
    """10a. Camera capture."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            record("Camera", True, f"frame shape: {frame.shape}")
        else:
            record("Camera", False, "no frame returned")
    except Exception as e:
        record("Camera", False, str(e))


def test_pytorch_cuda():
    """Verify PyTorch sees CUDA."""
    try:
        import torch
        available = torch.cuda.is_available()
        detail = f"PyTorch {torch.__version__}"
        if available:
            detail += f", GPU: {torch.cuda.get_device_name(0)}"
        record("PyTorch CUDA", available, detail)
    except Exception as e:
        record("PyTorch CUDA", False, str(e))


def test_opencv():
    """5. OpenCV with CUDA."""
    try:
        import cv2
        cuda_count = 0
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception:
            pass
        record("OpenCV", True, f"v{cv2.__version__}, CUDA devices: {cuda_count}")
    except Exception as e:
        record("OpenCV", False, str(e))


def test_yolo_cuda():
    """10b. YOLOv8 on CUDA."""
    try:
        import numpy as np
        from ultralytics import YOLO

        model_path = REPO_ROOT / "models" / "yolov8s.pt"
        if not model_path.exists():
            # Let ultralytics auto-download
            model = YOLO("yolov8s.pt")
        else:
            model = YOLO(str(model_path))

        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        r = model(dummy, verbose=False, device="cuda:0", classes=[0, 67])
        record("YOLOv8 CUDA", True, f"{len(r[0].boxes)} detections on dummy frame")
    except Exception as e:
        record("YOLOv8 CUDA", False, str(e))


def test_mediapipe_pose():
    """10c. MediaPipe Pose."""
    try:
        from src.perception.pose_estimator import PoseEstimator
        pe = PoseEstimator(config_path=str(REPO_ROOT / "config" / "pipeline.yaml"))
        pe.close()
        record("MediaPipe Pose", True)
    except Exception as e:
        record("MediaPipe Pose", False, str(e))


def test_mediapipe_face():
    """10d. MediaPipe Face (Gaze)."""
    try:
        from src.perception.gaze_tracker import GazeTracker
        gt = GazeTracker(config_path=str(REPO_ROOT / "config" / "pipeline.yaml"))
        gt.close()
        record("MediaPipe Face (Gaze)", True)
    except Exception as e:
        record("MediaPipe Face (Gaze)", False, str(e))


def test_posture_cnn():
    """10e. Posture CNN on CUDA."""
    cnn_path = REPO_ROOT / "data" / "trained_models" / "posture_cnn.pt"
    if not cnn_path.exists():
        record_skip("Posture CNN", "model not found at data/trained_models/posture_cnn.pt")
        return
    try:
        import torch
        from src.perception.posture_cnn import load_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, meta = load_model(str(cnn_path), device=device)
        record("Posture CNN", True, f"on {device}, metadata: {meta}")
    except Exception as e:
        record("Posture CNN", False, str(e))


def test_posture_lr():
    """10f. Posture LogisticRegression."""
    model_path = REPO_ROOT / "data" / "trained_models" / "posture_model.pkl"
    if not model_path.exists():
        record_skip("Posture LR", "model not found at data/trained_models/posture_model.pkl")
        return
    try:
        import numpy as np
        from src.perception.posture_model import PostureClassifier

        clf = PostureClassifier(model_path=str(model_path))
        r = clf.predict(np.zeros(6))
        record("Posture LR", True, f"p_bad={r.p_bad:.4f}")
    except Exception as e:
        record("Posture LR", False, str(e))


def test_whisper():
    """10g. Whisper STT on CUDA."""
    try:
        from faster_whisper import WhisperModel

        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        m = WhisperModel("small", device=device, compute_type=compute)
        record("Whisper STT", True, f"on {device}")
        del m
    except Exception as e:
        record("Whisper STT", False, str(e))


def test_microphone():
    """10h. Microphone via PyAudio."""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        info = pa.get_default_input_device_info()
        pa.terminate()
        record("Microphone", True, f"device: {info['name']}")
    except Exception as e:
        record("Microphone", False, str(e))


def test_tts():
    """10i. Piper TTS."""
    try:
        from src.voice.text_to_speech import TextToSpeech
        tts = TextToSpeech(voice="en_US-lessac-medium")
        record("Piper TTS", tts.is_available, f"backend: {tts._backend}")
    except Exception as e:
        record("Piper TTS", False, str(e))


def test_llm():
    """10j. Llama.cpp LLM."""
    # Search common model locations
    model_paths = [
        REPO_ROOT / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        Path.home() / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    ]
    model_path = None
    for p in model_paths:
        if p.exists():
            model_path = p
            break

    if model_path is None:
        record_skip("LLM (llama.cpp)", "no GGUF model found in models/")
        return

    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path=str(model_path),
            n_ctx=512,
            n_gpu_layers=-1,
            verbose=False,
        )
        out = llm("Hello", max_tokens=10)
        text = out["choices"][0]["text"].strip()[:40]
        record("LLM (llama.cpp)", True, f"response: {text!r}")
        del llm
    except Exception as e:
        record("LLM (llama.cpp)", False, str(e))


# ─── Main ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Desk Buddy — Jetson Component Verification")
    print("=" * 60)
    print()

    tests = [
        ("Core", [test_pytorch_cuda, test_opencv, test_camera]),
        ("Perception", [test_yolo_cuda, test_mediapipe_pose, test_mediapipe_face,
                        test_posture_cnn, test_posture_lr]),
        ("Voice & LLM", [test_whisper, test_microphone, test_tts, test_llm]),
    ]

    for section_name, section_tests in tests:
        print(f"\n── {section_name} ──")
        for test_fn in section_tests:
            test_fn()

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, s, _ in results if "PASS" in s)
    failed = sum(1 for _, s, _ in results if "FAIL" in s)
    skipped = sum(1 for _, s, _ in results if "SKIP" in s)
    total = len(results)
    print(f"  Results: {passed}/{total} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print(f"\n  Failed components:")
        for name, status, detail in results:
            if "FAIL" in status:
                print(f"    - {name}: {detail}")
        print(f"\n  See docs/JETSON_DEPLOY.md for troubleshooting.")

    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
