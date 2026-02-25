#!/usr/bin/env python3
"""
Train hybrid CNN + features model for posture classification.

Loads skeleton images and geometric features from session CSVs,
trains a small CNN with feature fusion, and saves the model.

Usage:
    python scripts/train_posture_cnn.py
    python scripts/train_posture_cnn.py --epochs 50 --batch-size 32
    python scripts/train_posture_cnn.py --no-features  # CNN only (ablation)
    python scripts/train_posture_cnn.py --depth-images  # Use depth-encoded skeletons

Recommended training command for confident predictions:
    python scripts/train_posture_cnn.py \\
        --epochs 50 \\
        --batch-size 16 \\
        --lr 5e-4 \\
        --dropout 0.5 \\
        --model-size normal \\
        --focal-loss \\
        --focal-alpha 0.65 \\
        --weight-decay 0.01 \\
        --scheduler cosine \\
        --early-stopping 20 \\
        --depth-images

Note: Do NOT use --label-smoothing as it compresses predictions toward 0.5,
making the model output ~0.49 for all samples during live inference.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import random

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("Error: PyTorch not installed. Run: pip install torch torchvision")
    sys.exit(1)

from src.perception.posture_cnn import PostureCNNClassifier, save_model


class FocalLoss(nn.Module):
    """
    Focal loss for hard example mining.

    Args:
        alpha: Weight for positive class (bad posture). Use alpha > 0.5 to
               up-weight the positive class for better recall.
        gamma: Focusing parameter. Higher values focus more on hard examples.
    """
    def __init__(self, alpha=0.65, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        # Apply alpha weighting: alpha for positive, (1-alpha) for negative
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class LabelSmoothingBCELoss(nn.Module):
    """
    Binary cross-entropy with label smoothing.

    Smooths labels to reduce overconfident predictions:
    - 0 → smoothing/2
    - 1 → 1 - smoothing/2
    """
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        # Smooth labels: 0 → eps, 1 → (1 - eps)
        eps = self.smoothing / 2
        smoothed_targets = targets * (1 - self.smoothing) + eps
        return nn.functional.binary_cross_entropy_with_logits(inputs, smoothed_targets)


class CombinedLoss(nn.Module):
    """Combines focal loss with label smoothing."""
    def __init__(self, alpha=0.65, gamma=1.5, smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        # Apply label smoothing
        eps = self.smoothing / 2
        smoothed_targets = targets * (1 - self.smoothing) + eps

        # Focal loss with smoothed targets
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, smoothed_targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# Feature columns (7 features including forward_lean_z)
FEATURE_NAMES = [
    "torso_pitch", "head_forward_ratio", "shoulder_roll",
    "lateral_lean", "head_tilt", "avg_visibility", "forward_lean_z",
]


def compute_forward_lean_z(row: dict) -> float:
    """Compute forward_lean_z from raw keypoint z-coordinates."""
    try:
        ls_z = float(row["kp_11_z"])
        rs_z = float(row["kp_12_z"])
        lh_z = float(row["kp_23_z"])
        rh_z = float(row["kp_24_z"])
        shoulder_z_avg = (ls_z + rs_z) / 2.0
        hip_z_avg = (lh_z + rh_z) / 2.0
        return shoulder_z_avg - hip_z_avg
    except (KeyError, ValueError):
        return 0.0


class PostureDataset(Dataset):
    """Dataset for skeleton images + geometric features."""

    def __init__(
        self,
        samples: List[dict],
        data_dir: Path,
        use_depth_images: bool = False,
        augment: bool = False,
        augment_config: Optional[dict] = None,
    ):
        """
        Args:
            samples: List of dicts with 'features', 'label', 'skeleton_path', 'session_id'
            data_dir: Base directory for skeleton images
            use_depth_images: If True, load 3-channel depth-encoded images
            augment: If True, apply data augmentation
            augment_config: Optional dict with augmentation parameters
        """
        self.samples = samples
        self.data_dir = data_dir
        self.use_depth_images = use_depth_images
        self.augment = augment

        # Augmentation config with defaults
        cfg = augment_config or {}
        self.rotation_range = cfg.get("rotation_range", 15)  # degrees
        self.scale_range = cfg.get("scale_range", (0.85, 1.15))
        self.translate_range = cfg.get("translate_range", 0.10)  # fraction of image
        self.perspective_jitter = cfg.get("perspective_jitter", 15)  # pixels
        self.random_erase_prob = cfg.get("random_erase_prob", 0.3)
        self.random_erase_size = cfg.get("random_erase_size", (10, 40))  # min/max pixels

    def __len__(self):
        return len(self.samples)

    def _apply_augmentation(self, img: np.ndarray, features: List[float]) -> Tuple[np.ndarray, List[float], bool]:
        """Apply augmentation to image and features. Returns (img, features, was_flipped)."""
        h, w = img.shape[:2]
        was_flipped = False
        features = list(features)  # Make a copy

        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            was_flipped = True
            # Flip relevant features:
            # shoulder_roll (idx 2) and head_tilt (idx 4) should be negated
            # lateral_lean (idx 3) should be negated
            features[2] = -features[2]  # shoulder_roll
            features[3] = -features[3]  # lateral_lean
            features[4] = -features[4]  # head_tilt

        # Random rotation (-rotation_range to rotation_range degrees)
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Random scaling
        scale = random.uniform(*self.scale_range)
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            img_scaled = cv2.resize(img, (new_w, new_h))
            # Center crop or pad to original size
            if scale > 1.0:
                # Crop center
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                img = img_scaled[start_y:start_y+h, start_x:start_x+w]
            else:
                # Pad with zeros
                if len(img.shape) == 3:
                    img = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
                else:
                    img = np.zeros((h, w), dtype=img_scaled.dtype)
                start_y = (h - new_h) // 2
                start_x = (w - new_w) // 2
                img[start_y:start_y+new_h, start_x:start_x+new_w] = img_scaled

        # Random translation
        if self.translate_range > 0:
            tx = random.uniform(-self.translate_range, self.translate_range) * w
            ty = random.uniform(-self.translate_range, self.translate_range) * h
            M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M_trans, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Random perspective warp
        if self.perspective_jitter > 0:
            jitter = self.perspective_jitter
            src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst_pts = np.float32([
                [random.uniform(0, jitter), random.uniform(0, jitter)],
                [w - random.uniform(0, jitter), random.uniform(0, jitter)],
                [w - random.uniform(0, jitter), h - random.uniform(0, jitter)],
                [random.uniform(0, jitter), h - random.uniform(0, jitter)],
            ])
            M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
            img = cv2.warpPerspective(img, M_persp, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Random erasing (cutout)
        if random.random() < self.random_erase_prob:
            erase_size = random.randint(*self.random_erase_size)
            ex = random.randint(0, w - erase_size)
            ey = random.randint(0, h - erase_size)
            if len(img.shape) == 3:
                img[ey:ey+erase_size, ex:ex+erase_size, :] = 0
            else:
                img[ey:ey+erase_size, ex:ex+erase_size] = 0

        return img, features, was_flipped

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load skeleton image
        img_path = self.data_dir / sample["skeleton_path"]
        if img_path.exists():
            if self.use_depth_images:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Fallback: blank image
            if self.use_depth_images:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = np.zeros((224, 224), dtype=np.uint8)

        features = list(sample["features"])

        # Data augmentation
        if self.augment:
            img, features, _ = self._apply_augmentation(img, features)

        # Convert to tensor
        img = img.astype(np.float32) / 255.0
        if self.use_depth_images:
            img = torch.from_numpy(img).permute(2, 0, 1)  # (3, 224, 224)
        else:
            img = torch.from_numpy(img).unsqueeze(0)  # (1, 224, 224)

        # Features
        features = torch.tensor(features, dtype=torch.float32)

        # Label
        label = torch.tensor([sample["label"]], dtype=torch.float32)

        return img, features, label


def load_data(data_dirs: List[Path]) -> List[dict]:
    """Load all samples from CSV files."""
    samples = []
    base_features = FEATURE_NAMES[:-1]  # First 6 features

    for data_dir in data_dirs:
        if not data_dir.exists():
            continue

        for csv_path in sorted(data_dir.glob("*.csv")):
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

                # Check format
                if "session_id" not in fieldnames:
                    continue  # Skip non-session files

                has_forward_lean_z = "forward_lean_z" in fieldnames
                has_skeleton = "skeleton_image_path" in fieldnames

                for row in reader:
                    label_str = row.get("label", "").strip()
                    if label_str not in ("good", "bad"):
                        continue

                    skeleton_path = row.get("skeleton_image_path", "")
                    if not skeleton_path or not has_skeleton:
                        continue

                    try:
                        # Load base features
                        feats = [float(row[col]) for col in base_features]

                        # Get or compute forward_lean_z
                        if has_forward_lean_z:
                            forward_lean_z = float(row["forward_lean_z"])
                        else:
                            forward_lean_z = compute_forward_lean_z(row)
                        feats.append(forward_lean_z)

                    except (KeyError, ValueError):
                        continue

                    samples.append({
                        "features": feats,
                        "label": 0 if label_str == "good" else 1,
                        "skeleton_path": skeleton_path,
                        "session_id": row.get("session_id", "unknown"),
                        "angle": row.get("angle", "unknown"),
                    })

    return samples


def split_by_session(
    samples: List[dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Split samples by session, stratified by label to ensure balanced test set."""
    random.seed(seed)

    # Group sessions by their dominant label
    session_labels = {}
    for s in samples:
        sid = s["session_id"]
        if sid not in session_labels:
            session_labels[sid] = {"good": 0, "bad": 0}
        if s["label"] == 0:
            session_labels[sid]["good"] += 1
        else:
            session_labels[sid]["bad"] += 1

    # Classify each session as "good" or "bad" based on majority
    good_sessions = []
    bad_sessions = []
    for sid, counts in session_labels.items():
        if counts["bad"] > counts["good"]:
            bad_sessions.append(sid)
        else:
            good_sessions.append(sid)

    random.shuffle(good_sessions)
    random.shuffle(bad_sessions)

    # Take test_ratio from each group
    n_test_good = max(1, int(len(good_sessions) * test_ratio))
    n_test_bad = max(1, int(len(bad_sessions) * test_ratio))

    test_sessions = set(good_sessions[:n_test_good] + bad_sessions[:n_test_bad])

    train_samples = [s for s in samples if s["session_id"] not in test_sessions]
    test_samples = [s for s in samples if s["session_id"] in test_sessions]

    # Print split info
    train_good = sum(1 for s in train_samples if s["label"] == 0)
    train_bad = sum(1 for s in train_samples if s["label"] == 1)
    test_good = sum(1 for s in test_samples if s["label"] == 0)
    test_bad = sum(1 for s in test_samples if s["label"] == 1)
    print(f"  Train: {len(train_samples)} (good={train_good}, bad={train_bad})")
    print(f"  Test: {len(test_samples)} (good={test_good}, bad={test_bad})")

    return train_samples, test_samples


def mixup_data(
    images: torch.Tensor,
    features: Optional[torch.Tensor],
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup augmentation.

    Returns:
        mixed_images, mixed_features, labels_a, labels_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_features = None
    if features is not None:
        mixed_features = lam * features + (1 - lam) * features[index]

    labels_a, labels_b = labels, labels[index]
    return mixed_images, mixed_features, labels_a, labels_b, lam


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    use_features: bool,
    mixup_alpha: float = 0.0,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0

    for images, features, labels in loader:
        images = images.to(device)
        features = features.to(device) if use_features else None
        labels = labels.to(device)

        optimizer.zero_grad()

        # Apply Mixup if enabled
        if mixup_alpha > 0 and random.random() < 0.5:
            mixed_images, mixed_features, labels_a, labels_b, lam = mixup_data(
                images, features, labels, mixup_alpha
            )
            outputs = model(mixed_images, mixed_features)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            outputs = model(images, features)
            loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_features: bool,
) -> Tuple[float, float, float]:
    """Evaluate model, return (f1, accuracy, loss)."""
    model.eval()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, features, labels in loader:
            images = images.to(device)
            features = features.to(device) if use_features else None
            labels = labels.to(device)

            outputs = model(images, features)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    accuracy = (all_preds == all_labels).mean()
    avg_loss = total_loss / len(all_labels)

    return f1, accuracy, avg_loss


def find_optimal_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_features: bool,
    min_recall: float = 0.70,
) -> Tuple[float, dict]:
    """
    Find optimal threshold that maximizes F1 score.

    Args:
        min_recall: Minimum acceptable recall (default 0.70). If no threshold
                    achieves this, returns the one with best F1 regardless.

    Returns:
        (optimal_threshold, metrics_dict)
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, features, labels in loader:
            images = images.to(device)
            features = features.to(device) if use_features else None

            outputs = model(images, features)
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Search thresholds from 0.1 to 0.9
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    best_metrics = {}

    # Also track best F1 with recall constraint
    best_f1_constrained = 0.0
    best_threshold_constrained = 0.5
    best_metrics_constrained = {}

    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(float)

        tp = ((preds == 1) & (all_labels == 1)).sum()
        tn = ((preds == 0) & (all_labels == 0)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)

        metrics = {
            "threshold": float(thresh),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

        # Track best F1 overall
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_metrics = metrics

        # Track best F1 with recall constraint
        if recall >= min_recall and f1 > best_f1_constrained:
            best_f1_constrained = f1
            best_threshold_constrained = thresh
            best_metrics_constrained = metrics

    # Prefer constrained result if it exists and F1 is close to unconstrained
    if best_metrics_constrained and best_f1_constrained >= best_f1 * 0.95:
        return best_threshold_constrained, best_metrics_constrained

    return best_threshold, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train posture CNN")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--no-features", action="store_true",
                        help="Ablation: CNN only, no geometric features")
    parser.add_argument("--depth-images", action="store_true",
                        help="Use depth-encoded (color) skeleton images")
    parser.add_argument("--output", default="data/trained_models/posture_cnn.pt")
    parser.add_argument("--device", default="auto",
                        help="Device: cpu, cuda, mps, or auto")
    parser.add_argument("--focal-loss", action="store_true",
                        help="Use focal loss for hard example mining")
    parser.add_argument("--focal-alpha", type=float, default=0.65,
                        help="Focal loss alpha (>0.5 favors bad class recall, default: 0.65)")
    parser.add_argument("--focal-gamma", type=float, default=1.5,
                        help="Focal loss gamma (focusing parameter, default: 1.5)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0=disabled, default: 0.0)")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Mixup alpha (0=disabled, default: 0.0)")
    parser.add_argument("--min-recall", type=float, default=0.70,
                        help="Minimum recall for threshold optimization (default: 0.70)")
    parser.add_argument("--weighted-sampling", action="store_true",
                        help="Enable weighted sampling to oversample minority class")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight decay for optimizer (default: 5e-4)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (0=disabled, default: 1.0)")
    parser.add_argument("--early-stopping", type=int, default=0,
                        help="Stop if no improvement for N epochs (0=disabled)")
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="Dropout rate (default: 0.4)")
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["plateau", "cosine", "step"],
                        help="LR scheduler: plateau, cosine, or step")
    parser.add_argument("--model-size", type=str, default="tiny",
                        choices=["tiny", "normal"],
                        help="Model size: tiny (~25k params, less overfitting) or normal (~250k)")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Data directories
    base_dir = Path(__file__).parent.parent / "data"
    data_dirs = [
        base_dir / "posture_sessions",
        base_dir / "old_sessions",
    ]

    # Load data
    print("\nLoading data...")
    samples = load_data(data_dirs)
    print(f"  Total samples: {len(samples)}")

    if len(samples) < 20:
        print("Error: Need at least 20 samples with skeleton images")
        return 1

    # Count labels
    n_good = sum(1 for s in samples if s["label"] == 0)
    n_bad = sum(1 for s in samples if s["label"] == 1)
    print(f"  Good: {n_good}, Bad: {n_bad}")

    # Split by session (stratified by label)
    train_samples, test_samples = split_by_session(samples, args.test_ratio)

    # Create datasets
    use_features = not args.no_features
    image_channels = 3 if args.depth_images else 1

    # Augmentation config
    augment_config = {
        "rotation_range": 15,
        "scale_range": (0.85, 1.15),
        "translate_range": 0.10,
        "perspective_jitter": 15,
        "random_erase_prob": 0.3,
        "random_erase_size": (10, 40),
    }

    train_dataset = PostureDataset(
        train_samples, base_dir,
        use_depth_images=args.depth_images,
        augment=True,
        augment_config=augment_config,
    )
    test_dataset = PostureDataset(
        test_samples, base_dir,
        use_depth_images=args.depth_images,
        augment=False,
    )

    # Create data loaders with optional weighted sampling
    if args.weighted_sampling:
        # Calculate class weights for weighted sampling
        train_labels = [s["label"] for s in train_samples]
        n_good_train = sum(1 for l in train_labels if l == 0)
        n_bad_train = sum(1 for l in train_labels if l == 1)

        # Give higher weight to minority class (bad posture)
        weight_good = 1.0
        weight_bad = n_good_train / max(1, n_bad_train) * 1.2  # 1.2x boost for recall
        sample_weights = [weight_bad if l == 1 else weight_good for l in train_labels]

        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_samples),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0
        )
        print(f"  Weighted sampling enabled: good={weight_good:.2f}, bad={weight_bad:.2f}")
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Create model
    model = PostureCNNClassifier(
        image_channels=image_channels,
        num_features=7,
        use_features=use_features,
        dropout=args.dropout,
        model_size=args.model_size,
    ).to(device)

    print(f"\nModel: {args.model_size.upper()} CNN {'+ Features' if use_features else 'only'}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Training setup
    if args.focal_loss and args.label_smoothing > 0:
        criterion = CombinedLoss(
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
            smoothing=args.label_smoothing
        )
        print(f"  Using Combined Loss (focal alpha={args.focal_alpha}, gamma={args.focal_gamma}, smoothing={args.label_smoothing})")
    elif args.focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"  Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    elif args.label_smoothing > 0:
        criterion = LabelSmoothingBCELoss(smoothing=args.label_smoothing)
        print(f"  Using Label Smoothing BCE (smoothing={args.label_smoothing})")
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"  Weight decay: {args.weight_decay}")

    # Learning rate scheduler
    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        scheduler_step_on = "loss"
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        scheduler_step_on = "epoch"
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.5
        )
        scheduler_step_on = "epoch"
    print(f"  LR scheduler: {args.scheduler}")

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Model selection: best F1 score")
    if args.early_stopping > 0:
        print(f"  Early stopping: {args.early_stopping} epochs patience")
    if args.mixup_alpha > 0:
        print(f"  Mixup enabled (alpha={args.mixup_alpha})")
    if args.max_grad_norm > 0:
        print(f"  Gradient clipping: max_norm={args.max_grad_norm}")

    best_f1 = 0.0
    best_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, use_features,
            mixup_alpha=args.mixup_alpha,
            max_grad_norm=args.max_grad_norm,
        )
        test_f1, test_acc, test_loss = evaluate(model, test_loader, device, use_features)

        # Step scheduler
        if scheduler_step_on == "loss":
            scheduler.step(test_loss)
        else:
            scheduler.step()

        # Select best model by F1 score
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_acc = test_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            marker = " *"
        else:
            epochs_without_improvement += 1
            marker = ""

        print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}, "
              f"test_loss={test_loss:.4f}, test_f1={test_f1:.3f}, test_acc={test_acc:.3f}{marker}")

        # Early stopping
        if args.early_stopping > 0 and epochs_without_improvement >= args.early_stopping:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {args.early_stopping} epochs)")
            break

    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    print(f"\nBest test F1: {best_f1:.3f} (accuracy: {best_acc:.3f})")

    # Collect all predictions for metrics
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, features, labels in test_loader:
            images = images.to(device)
            features = features.to(device) if use_features else None
            labels = labels.to(device)

            outputs = model(images, features)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Per-class metrics
    tp = ((all_preds == 1) & (all_labels == 1)).sum()  # True positive (bad correct)
    tn = ((all_preds == 0) & (all_labels == 0)).sum()  # True negative (good correct)
    fp = ((all_preds == 1) & (all_labels == 0)).sum()  # False positive
    fn = ((all_preds == 0) & (all_labels == 1)).sum()  # False negative

    total_good = int((all_labels == 0).sum())
    total_bad = int((all_labels == 1).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Good    Bad")
    print(f"  Actual Good    {tn:4d}   {fp:4d}")
    print(f"  Actual Bad     {fn:4d}   {tp:4d}")

    print(f"\n  Per-class accuracy:")
    print(f"    Good: {tn}/{total_good} = {tn/max(1,total_good):.3f}")
    print(f"    Bad:  {tp}/{total_bad} = {tp/max(1,total_bad):.3f}")

    print(f"\n  Metrics (for 'bad' posture detection) @ threshold=0.5:")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall:    {recall:.3f}")
    print(f"    F1 Score:  {f1:.3f}")

    # Find optimal threshold (maximize F1 with optional min recall constraint)
    print(f"\n  Finding optimal threshold (maximize F1, min recall={args.min_recall:.0%})...")
    optimal_threshold, optimal_metrics = find_optimal_threshold(
        model, test_loader, device, use_features,
        min_recall=args.min_recall
    )

    if optimal_metrics:
        print(f"\n  Optimal Threshold: {optimal_threshold:.2f}")
        print(f"    Precision: {optimal_metrics['precision']:.3f}")
        print(f"    Recall:    {optimal_metrics['recall']:.3f}")
        print(f"    F1 Score:  {optimal_metrics['f1']:.3f}")
        print(f"    TP={optimal_metrics['tp']}, TN={optimal_metrics['tn']}, "
              f"FP={optimal_metrics['fp']}, FN={optimal_metrics['fn']}")
    else:
        print(f"  Warning: Could not find threshold achieving target recall")
        optimal_threshold = 0.5

    # Save model
    output_path = Path(args.output)
    save_model(model, str(output_path), metadata={
        "best_f1": best_f1,
        "best_accuracy": best_acc,
        "epochs": args.epochs,
        "use_features": use_features,
        "depth_images": args.depth_images,
        "optimal_threshold": optimal_threshold,
        "min_recall": args.min_recall,
        "final_precision": float(optimal_metrics.get('precision', precision)) if optimal_metrics else float(precision),
        "final_recall": float(optimal_metrics.get('recall', recall)) if optimal_metrics else float(recall),
        "final_f1": float(optimal_metrics.get('f1', f1)) if optimal_metrics else float(f1),
        "training_config": {
            "focal_loss": args.focal_loss,
            "focal_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
            "label_smoothing": args.label_smoothing,
            "mixup_alpha": args.mixup_alpha,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "model_size": args.model_size,
        }
    })
    print(f"\nModel saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
