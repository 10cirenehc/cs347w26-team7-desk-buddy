"""
Learned posture classifier with L2-norm threshold fallback.

When a trained model is available (LogisticRegression or MLP), it
predicts ``p_bad`` from calibration-normalised features.  Otherwise,
an L2-norm threshold on the first 5 features (excluding visibility)
provides a usable default before any training data is collected.
"""

import json
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import yaml

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    LogisticRegression = None  # OK — we can still load from JSON weights


@dataclass
class PostureClassification:
    """Output of the posture classifier."""
    p_bad: float        # probability that current posture is bad [0, 1]
    method: str         # "model" or "fallback"


class _NumpyLR:
    """Minimal numpy-only logistic regression predictor.

    Reproduces ``sklearn.linear_model.LogisticRegression.predict_proba``
    for binary classification using only numpy — no sklearn dependency.
    Loaded from the JSON weights exported by ``scripts/export_models.py``.
    """

    def __init__(self, coef: np.ndarray, intercept: np.ndarray, classes: np.ndarray):
        self.coef_ = coef            # (1, n_features)
        self.intercept_ = intercept  # (1,)
        self.classes_ = classes      # (2,)
        self.n_features_in_ = coef.shape[1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logit = X @ self.coef_.T + self.intercept_  # (N, 1)
        p1 = 1.0 / (1.0 + np.exp(-logit))          # sigmoid
        p0 = 1.0 - p1
        return np.hstack([p0, p1])                  # (N, 2)


class PostureClassifier:
    """
    Predict p_bad from calibration-normalised features.

    * If a trained model is loaded: ``LogisticRegression.predict_proba``.
    * Otherwise: L2-norm of features[:5] vs ``threshold_fallback``.

    Supports loading from:
    * ``.pkl``  — sklearn pickle (original format)
    * ``.json`` — portable JSON weights (no sklearn needed, cross-version safe)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold_fallback: float = 2.0,
        config_path: Optional[str] = None,
    ):
        if config_path:
            cfg = self._load_config(config_path).get("posture_classifier", {})
            model_path = cfg.get("model_path", model_path)
            threshold_fallback = cfg.get("threshold_fallback", threshold_fallback)

        self.threshold_fallback = threshold_fallback
        self._model = None  # LogisticRegression or _NumpyLR

        if model_path:
            p = Path(model_path)
            # Prefer JSON if it exists alongside a .pkl path
            json_path = p.with_suffix(".json")
            if json_path.exists():
                self.load(str(json_path))
            elif p.exists():
                self.load(str(p))

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, normalised: np.ndarray) -> PostureClassification:
        """
        Predict posture quality from normalised features.

        Args:
            normalised: (6,) z-score-normalised feature vector from
                        CalibrationManager.normalize().

        Returns:
            PostureClassification with p_bad in [0, 1].
        """
        if self._model is not None:
            x = normalised.reshape(1, -1)
            proba = self._model.predict_proba(x)[0]
            # Convention: class 1 = bad
            p_bad = float(proba[1]) if proba.shape[0] > 1 else float(proba[0])
            return PostureClassification(p_bad=p_bad, method="model")

        # Fallback: L2-norm on the first 5 features (skip avg_visibility).
        norm = float(np.linalg.norm(normalised[:5]))
        p_bad = min(1.0, norm / self.threshold_fallback)
        return PostureClassification(p_bad=p_bad, method="fallback")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train logistic regression on labelled data.

        Args:
            X: (N, 6) normalised feature matrix.
            y: (N,)   labels — 0 = good, 1 = bad.

        Returns:
            Dict with training metrics.
        """
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X, y)
        self._model = model
        acc = float(model.score(X, y))
        return {"accuracy": acc, "n_samples": len(y)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        if self._model is None:
            raise ValueError("No model trained yet")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str) -> None:
        """Load model from .pkl (sklearn) or .json (portable weights)."""
        if path.endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)
            self._model = _NumpyLR(
                coef=np.array(data["coef"], dtype=np.float64),
                intercept=np.array(data["intercept"], dtype=np.float64),
                classes=np.array(data["classes"]),
            )
        else:
            with open(path, "rb") as f:
                self._model = pickle.load(f)

    @property
    def has_model(self) -> bool:
        return self._model is not None
