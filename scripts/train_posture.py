#!/usr/bin/env python3
"""
Train posture classifier on collected CSV data.

Supports two data formats:
1. Old format (from collect_posture_data.py): calibration-normalized features
2. New format (from collect_posture_sessions.py): raw features with angle/session metadata

Loads data from both data/posture_labels/ (old) and data/posture_sessions/ (new),
trains a LogisticRegression model, reports metrics stratified by angle,
and saves the model.

Usage:
    python scripts/train_posture.py
    python scripts/train_posture.py --data-dir data/posture_sessions
    python scripts/train_posture.py --output data/trained_models/posture_model.pkl
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perception.posture_model import PostureClassifier

# Feature columns for the new session-based format (raw, no calibration)
# Includes forward_lean_z which uses MediaPipe's depth (z) coordinate
RAW_FEATURE_NAMES = [
    "torso_pitch", "head_forward_ratio", "shoulder_roll",
    "lateral_lean", "head_tilt", "avg_visibility", "forward_lean_z",
]

# Feature columns for the old calibration-normalized format (6 features, no depth)
NORM_FEATURE_NAMES = [
    "norm_torso_pitch", "norm_head_forward", "norm_shoulder_roll",
    "norm_lateral_lean", "norm_head_tilt", "norm_visibility",
]


def detect_format(csv_path: Path) -> str:
    """Detect CSV format by checking for session_id column."""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "session_id" in fieldnames:
            return "session"
        elif "norm_torso_pitch" in fieldnames:
            return "calibrated"
        else:
            return "unknown"


def compute_forward_lean_z(row: dict) -> float:
    """Compute forward_lean_z from raw keypoint z-coordinates."""
    # MediaPipe landmark indices: 11=left_shoulder, 12=right_shoulder, 23=left_hip, 24=right_hip
    try:
        ls_z = float(row["kp_11_z"])
        rs_z = float(row["kp_12_z"])
        lh_z = float(row["kp_23_z"])
        rh_z = float(row["kp_24_z"])
        shoulder_z_avg = (ls_z + rs_z) / 2.0
        hip_z_avg = (lh_z + rh_z) / 2.0
        return shoulder_z_avg - hip_z_avg  # negative = forward lean
    except (KeyError, ValueError):
        return 0.0


def load_session_data(csv_files: List[Path]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load data from session-based CSVs.

    Handles both old format (6 features) and new format (7 features).
    Computes forward_lean_z from raw keypoints if missing.

    Returns:
        X: Feature array (N, 7)
        y: Labels (N,) - 0=good, 1=bad
        angles: List of angle tags per sample
        sessions: List of session IDs per sample
    """
    rows_X = []
    rows_y = []
    angles = []
    sessions = []

    # Features without forward_lean_z (for old format)
    base_features = RAW_FEATURE_NAMES[:-1]  # First 6 features

    for csv_path in csv_files:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            has_forward_lean_z = "forward_lean_z" in fieldnames

            for row in reader:
                label = row.get("label", "").strip()
                if label not in ("good", "bad"):
                    continue

                try:
                    # Load base 6 features
                    feats = [float(row[col]) for col in base_features]

                    # Get or compute forward_lean_z
                    if has_forward_lean_z:
                        forward_lean_z = float(row["forward_lean_z"])
                    else:
                        forward_lean_z = compute_forward_lean_z(row)

                    feats.append(forward_lean_z)
                except (KeyError, ValueError):
                    continue

                rows_X.append(feats)
                rows_y.append(0 if label == "good" else 1)
                angles.append(row.get("angle", "unknown"))
                sessions.append(row.get("session_id", "unknown"))

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.int32)
    return X, y, angles, sessions


def load_calibrated_data(csv_files: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from calibration-normalized CSVs (old format).

    Returns:
        X: Feature array (N, 6)
        y: Labels (N,) - 0=good, 1=bad
    """
    rows_X = []
    rows_y = []

    for csv_path in csv_files:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("label", "").strip()
                if label not in ("good", "bad"):
                    continue
                try:
                    feats = [float(row[col]) for col in NORM_FEATURE_NAMES]
                except (KeyError, ValueError):
                    continue

                rows_X.append(feats)
                rows_y.append(0 if label == "good" else 1)

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.int32)
    return X, y


def load_all_data(data_dirs: List[Path]) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]], Optional[List[str]]]:
    """
    Load data from all specified directories, auto-detecting format.

    Returns:
        X: Feature array
        y: Labels
        angles: Angle tags (None if using calibrated format)
        sessions: Session IDs (None if using calibrated format)
    """
    session_files = []
    calibrated_files = []

    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for csv_path in sorted(data_dir.glob("*.csv")):
            fmt = detect_format(csv_path)
            if fmt == "session":
                session_files.append(csv_path)
                print(f"  [session] {csv_path.name}")
            elif fmt == "calibrated":
                calibrated_files.append(csv_path)
                print(f"  [calibrated] {csv_path.name}")
            else:
                print(f"  [unknown format, skipped] {csv_path.name}")

    all_X = []
    all_y = []
    all_angles = []
    all_sessions = []
    has_session_data = False

    # Load session-based data
    if session_files:
        X_sess, y_sess, angles, sessions = load_session_data(session_files)
        if len(y_sess) > 0:
            all_X.append(X_sess)
            all_y.append(y_sess)
            all_angles.extend(angles)
            all_sessions.extend(sessions)
            has_session_data = True

    # Load calibrated data
    if calibrated_files:
        X_cal, y_cal = load_calibrated_data(calibrated_files)
        if len(y_cal) > 0:
            all_X.append(X_cal)
            all_y.append(y_cal)
            # No angle/session info for calibrated data
            all_angles.extend(["calibrated"] * len(y_cal))
            all_sessions.extend(["calibrated"] * len(y_cal))

    if not all_X:
        return np.array([]), np.array([]), None, None

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    return X, y, all_angles if has_session_data else None, all_sessions if has_session_data else None


def split_by_session(
    X: np.ndarray,
    y: np.ndarray,
    sessions: List[str],
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Split data by session to avoid data leakage.

    Entire sessions go to train or test, not individual frames.
    """
    np.random.seed(random_seed)

    unique_sessions = list(set(sessions))
    np.random.shuffle(unique_sessions)

    n_test = max(1, int(len(unique_sessions) * test_ratio))
    test_sessions = set(unique_sessions[:n_test])
    train_sessions = set(unique_sessions[n_test:])

    train_mask = np.array([s in train_sessions for s in sessions])
    test_mask = np.array([s in test_sessions for s in sessions])

    sessions_arr = np.array(sessions)

    return (
        X[train_mask], y[train_mask],
        X[test_mask], y[test_mask],
        sessions_arr[train_mask].tolist(),
        sessions_arr[test_mask].tolist(),
    )


def compute_per_angle_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    angles: List[str],
) -> Dict[str, dict]:
    """Compute accuracy metrics stratified by angle tag."""
    angle_metrics = defaultdict(lambda: {"correct": 0, "total": 0})

    for true, pred, angle in zip(y_true, y_pred, angles):
        angle_metrics[angle]["total"] += 1
        if true == pred:
            angle_metrics[angle]["correct"] += 1

    results = {}
    for angle, counts in angle_metrics.items():
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        results[angle] = {
            "accuracy": acc,
            "correct": counts["correct"],
            "total": counts["total"],
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Train posture classifier")
    parser.add_argument("--data-dir", default=None,
                        help="Data directory (default: both data/posture_labels and data/posture_sessions)")
    parser.add_argument("--output", default="data/trained_models/posture_model.pkl")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Fraction of sessions to use for testing (default: 0.2)")
    args = parser.parse_args()

    # Determine data directories
    base_dir = Path(__file__).parent.parent / "data"
    if args.data_dir:
        data_dirs = [Path(args.data_dir)]
    else:
        data_dirs = [
            base_dir / "posture_labels",
            base_dir / "posture_sessions",
            base_dir / "old_sessions",  # Legacy data from earlier collection
        ]

    print("Loading training data...")
    X, y, angles, sessions = load_all_data(data_dirs)

    if len(y) == 0:
        print("Error: no valid data found")
        print(f"Searched directories: {[str(d) for d in data_dirs]}")
        return 1

    print(f"\nTotal samples: {len(y)}  (good={int((y == 0).sum())}, bad={int((y == 1).sum())})")

    if len(y) < 10:
        print("Error: need at least 10 labelled samples to train")
        return 1

    # Split by session if we have session metadata
    if sessions and len(set(sessions)) > 1 and "calibrated" not in sessions:
        print(f"\nSplitting by session (test_ratio={args.test_ratio})...")
        X_train, y_train, X_test, y_test, sessions_train, sessions_test = split_by_session(
            X, y, sessions, test_ratio=args.test_ratio
        )
        angles_train = [angles[i] for i, s in enumerate(sessions) if s in set(sessions_train)]
        angles_test = [angles[i] for i, s in enumerate(sessions) if s in set(sessions_test)]
        print(f"  Train: {len(y_train)} samples from {len(set(sessions_train))} sessions")
        print(f"  Test: {len(y_test)} samples from {len(set(sessions_test))} sessions")
    else:
        # Random split for calibrated data or single session
        from sklearn.model_selection import train_test_split
        print("\nUsing random train/test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_ratio, random_state=42, stratify=y
        )
        angles_train = angles_test = None
        print(f"  Train: {len(y_train)} samples")
        print(f"  Test: {len(y_test)} samples")

    # Train classifier
    print("\nTraining LogisticRegression...")
    clf = PostureClassifier()

    # Train on training set using the train() method
    metrics = clf.train(X_train, y_train)
    print(f"  Training accuracy: {metrics['accuracy']:.3f}")

    # Evaluate on test set
    y_test_pred = clf._model.predict(X_test)
    test_acc = (y_test_pred == y_test).mean()
    print(f"  Test accuracy: {test_acc:.3f}")

    # Per-feature importances
    if clf._model is not None:
        coefs = clf._model.coef_[0]
        print("\nFeature importances (|coefficient|):")
        for name, c in sorted(zip(RAW_FEATURE_NAMES, coefs), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {name:>25s}: {c:+.4f}")

    # Confusion matrix on test set
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nTest set confusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':>10s} pred_good  pred_bad")
    print(f"  {'good':>10s}  {cm[0, 0]:>8d}  {cm[0, 1]:>8d}")
    print(f"  {'bad':>10s}  {cm[1, 0]:>8d}  {cm[1, 1]:>8d}")

    print(f"\nTest set classification report:")
    print(classification_report(y_test, y_test_pred, target_names=["good", "bad"]))

    # Per-angle metrics on test set
    if angles_test and len(set(angles_test)) > 1:
        print("Per-angle test accuracy:")
        angle_metrics = compute_per_angle_metrics(y_test, y_test_pred, angles_test)
        for angle in sorted(angle_metrics.keys()):
            m = angle_metrics[angle]
            print(f"  {angle:>15s}: {m['accuracy']:.3f} ({m['correct']}/{m['total']})")

    # Per-angle metrics on full dataset (train + test combined)
    if angles and len(set(angles)) > 1:
        print("\nPer-angle accuracy (full dataset):")
        y_all_pred = clf._model.predict(X)
        angle_metrics_all = compute_per_angle_metrics(y, y_all_pred, angles)
        for angle in sorted(angle_metrics_all.keys()):
            m = angle_metrics_all[angle]
            print(f"  {angle:>15s}: {m['accuracy']:.3f} ({m['correct']}/{m['total']})")

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save(str(output_path))
    print(f"\nModel saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
