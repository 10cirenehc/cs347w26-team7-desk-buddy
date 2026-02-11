"""
Learned posture classifier with L2-norm threshold fallback.

When a trained model is available (LogisticRegression or MLP), it
predicts ``p_bad`` from calibration-normalised features.  Otherwise,
an L2-norm threshold on the first 5 features (excluding visibility)
provides a usable default before any training data is collected.
"""

import pickle
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import yaml

from sklearn.linear_model import LogisticRegression


@dataclass
class PostureClassification:
    """Output of the posture classifier."""
    p_bad: float        # probability that current posture is bad [0, 1]
    method: str         # "model" or "fallback"


class PostureClassifier:
    """
    Predict p_bad from calibration-normalised features.

    * If a trained model is loaded: ``LogisticRegression.predict_proba``.
    * Otherwise: L2-norm of features[:5] vs ``threshold_fallback``.
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
        self._model: Optional[LogisticRegression] = None

        if model_path:
            p = Path(model_path)
            if p.exists():
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
        with open(path, "rb") as f:
            self._model = pickle.load(f)

    @property
    def has_model(self) -> bool:
        return self._model is not None
