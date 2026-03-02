"""
ONNX Runtime wrapper for PostureCNN inference.

Extracted from posture_cnn.py so that ONNX loading does NOT pull in torch.
Only requires: onnxruntime, numpy, json, pathlib.
"""

from typing import Optional, Tuple
from pathlib import Path


class ONNXPostureCNN:
    """ONNX Runtime wrapper that exposes the same interface as PostureCNNClassifier.

    Loads the .onnx exported by ``scripts/export_models.py`` and runs inference
    via onnxruntime — no PyTorch dependency needed, works on any Python 3.6+.
    """

    def __init__(self, onnx_path: str, meta_path: Optional[str] = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        input_names = [inp.name for inp in self.session.get_inputs()]
        self.use_features = "features" in input_names

        # Load metadata (channels, threshold, etc.)
        self.metadata: dict = {}
        if meta_path is None:
            meta_path = Path(onnx_path).with_suffix(".meta.json")
        else:
            meta_path = Path(meta_path)
        if meta_path.exists():
            import json
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

    def predict_proba(self, image, features=None):
        """Run inference. Accepts numpy arrays or torch tensors."""
        import numpy as np

        img = image.cpu().numpy() if hasattr(image, "cpu") else np.asarray(image)
        feeds = {"image": img.astype(np.float32)}
        if self.use_features and features is not None:
            feat = features.cpu().numpy() if hasattr(features, "cpu") else np.asarray(features)
            feeds["features"] = feat.astype(np.float32)
        (logits,) = self.session.run(None, feeds)
        # Model outputs raw logits; apply sigmoid
        p_bad = 1.0 / (1.0 + np.exp(-logits))
        return p_bad

    def eval(self):
        """No-op for API compatibility."""
        return self


def load_onnx_model(onnx_path: str) -> Tuple[ONNXPostureCNN, dict]:
    """Load an ONNX-exported CNN model.

    Returns (model, metadata) matching the load_model() signature.
    """
    model = ONNXPostureCNN(onnx_path)
    return model, model.metadata
