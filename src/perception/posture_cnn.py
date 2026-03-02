"""
Hybrid CNN + geometric features model for posture classification.

Combines:
- Small CNN on skeleton images (224×224 grayscale or depth-encoded)
- MLP on geometric features (7 values including forward_lean_z)

Fuses both branches for final binary classification (good vs bad posture).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from pathlib import Path
import pickle


class TinyCNN(nn.Module):
    """
    Tiny CNN to reduce overfitting on small datasets.

    Much smaller than SkeletonCNN: ~8k params vs ~250k params.
    Uses aggressive stride and pooling to reduce spatial dimensions quickly.

    Input: (B, 1, 224, 224) grayscale or (B, 3, 224, 224) depth-encoded
    Output: (B, 32) embedding
    """

    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()

        # Two conv layers with aggressive downsampling
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.drop1 = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout2d(dropout)

        # Global average pooling → 32-dim
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # (B, 32)
        return x


class SkeletonCNN(nn.Module):
    """
    Small CNN for skeleton image feature extraction.

    Input: (B, 1, 224, 224) grayscale or (B, 3, 224, 224) depth-encoded
    Output: (B, 128) embedding
    """

    def __init__(self, in_channels: int = 1, dropout: float = 0.2):
        super().__init__()

        # Conv layers with batch norm and spatial dropout
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(dropout * 0.5)  # Lower dropout in early layers

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(dropout * 0.5)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout2d(dropout)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.drop4 = nn.Dropout2d(dropout)

        # Global average pooling → 128-dim
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        x = self.drop4(F.relu(self.bn4(self.conv4(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # (B, 128)
        return x


class FeatureMLP(nn.Module):
    """
    Small MLP for geometric feature processing.

    Input: (B, 7) geometric features
    Output: (B, 32) embedding
    """

    def __init__(self, in_features: int = 7, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        return x


class PostureCNNClassifier(nn.Module):
    """
    Hybrid model combining skeleton images and geometric features.

    Architecture:
        Skeleton Image (224×224) → CNN → 128-dim (normal) or 32-dim (tiny)
                                           ↓
                                       Concat → FC → p_bad
                                           ↑
        Geometric Features (7) → MLP → 32-dim
    """

    def __init__(
        self,
        image_channels: int = 1,
        num_features: int = 7,
        use_features: bool = True,
        dropout: float = 0.3,
        model_size: str = "normal",
    ):
        """
        Args:
            image_channels: 1 for grayscale skeleton, 3 for depth-encoded
            num_features: Number of geometric features (default 7)
            use_features: If False, only use CNN (ablation study)
            dropout: Dropout rate for regularization
            model_size: "tiny" (~25k params) or "normal" (~250k params)
        """
        super().__init__()

        self.use_features = use_features
        self.dropout_rate = dropout
        self.model_size = model_size

        # Image branch - select based on model_size
        if model_size == "tiny":
            self.cnn = TinyCNN(in_channels=image_channels, dropout=dropout)
            cnn_out_dim = 32
        else:
            self.cnn = SkeletonCNN(in_channels=image_channels, dropout=dropout * 0.5)
            cnn_out_dim = 128

        # Feature branch
        if use_features:
            self.mlp = FeatureMLP(in_features=num_features, dropout=dropout)
            fusion_dim = cnn_out_dim + 32  # CNN output + MLP output
        else:
            self.mlp = None
            fusion_dim = cnn_out_dim

        # Classification head - smaller for tiny model
        if model_size == "tiny":
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

    def forward(
        self,
        image: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            image: (B, C, 224, 224) skeleton image
            features: (B, 7) geometric features (optional if use_features=False)

        Returns:
            (B, 1) logits (apply sigmoid for probability)
        """
        # Image branch
        img_emb = self.cnn(image)  # (B, 128)

        # Feature branch + fusion
        if self.use_features and features is not None:
            feat_emb = self.mlp(features)  # (B, 32)
            combined = torch.cat([img_emb, feat_emb], dim=1)  # (B, 160)
        else:
            combined = img_emb

        # Classification
        logits = self.classifier(combined)  # (B, 1)
        return logits

    def predict_proba(
        self,
        image: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get probability of bad posture."""
        logits = self.forward(image, features)
        return torch.sigmoid(logits)


def save_model(model: PostureCNNClassifier, path: str, metadata: dict = None):
    """Save model weights and metadata."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "image_channels": model.cnn.conv1.in_channels,
            "num_features": model.mlp.fc1.in_features if model.mlp else 7,
            "use_features": model.use_features,
            "dropout": model.dropout_rate,
            "model_size": getattr(model, "model_size", "normal"),
        },
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


def load_model(path: str, device: str = "cpu") -> Tuple[PostureCNNClassifier, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint["model_config"]
    model = PostureCNNClassifier(
        image_channels=config["image_channels"],
        num_features=config["num_features"],
        use_features=config["use_features"],
        dropout=config.get("dropout", 0.3),
        model_size=config.get("model_size", "normal"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint.get("metadata", {})


