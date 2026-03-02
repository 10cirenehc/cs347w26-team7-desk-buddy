"""
Export trained models to portable formats for cross-Python-version deployment.

Solves: pickle files created on Python 3.9+ / newer sklearn can't be loaded
on Jetson (Python 3.8) due to protocol or version mismatches.

Outputs:
  - posture_model.json   (sklearn LR → raw weights, loads with pure numpy)
  - posture_cnn.onnx     (PyTorch CNN → ONNX, loads with onnxruntime)

Usage:
    python scripts/export_models.py
    python scripts/export_models.py --lr-only
    python scripts/export_models.py --cnn-only
"""

import argparse
import json
import sys
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINED_DIR = PROJECT_ROOT / "data" / "trained_models"


def export_lr_to_json(
    pkl_path: Path = TRAINED_DIR / "posture_model.pkl",
    out_path: Path = TRAINED_DIR / "posture_model.json",
):
    """Export sklearn LogisticRegression weights to a JSON file.

    The JSON contains coefficients, intercept, and classes — enough to
    reproduce predict_proba with pure numpy (no sklearn needed).
    """
    import pickle

    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    payload = {
        "model_type": "LogisticRegression",
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "classes": model.classes_.tolist(),
        "n_features_in": int(model.n_features_in_),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[OK] Exported LR weights → {out_path}")
    print(f"     coef shape: {model.coef_.shape}, intercept: {model.intercept_}")


def export_cnn_to_onnx(
    pt_path: Path = TRAINED_DIR / "posture_cnn.pt",
    out_path: Path = TRAINED_DIR / "posture_cnn.onnx",
):
    """Export PyTorch CNN checkpoint to ONNX.

    The ONNX model can be loaded with onnxruntime on any Python version
    and optionally optimized with TensorRT on the Jetson.
    """
    import torch

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.perception.posture_cnn import load_model

    model, metadata = load_model(str(pt_path), device="cpu")
    model.eval()

    in_channels = model.cnn.conv1.in_channels
    use_features = model.use_features
    num_features = model.mlp.fc1.in_features if model.mlp else 7

    # Build dummy inputs
    dummy_image = torch.randn(1, in_channels, 224, 224)
    input_names = ["image"]
    dynamic_axes = {"image": {0: "batch"}, "p_bad": {0: "batch"}}

    if use_features:
        dummy_features = torch.randn(1, num_features)
        args = (dummy_image, dummy_features)
        input_names.append("features")
        dynamic_axes["features"] = {0: "batch"}
    else:
        args = (dummy_image,)

    torch.onnx.export(
        model,
        args,
        str(out_path),
        input_names=input_names,
        output_names=["p_bad"],
        dynamic_axes=dynamic_axes,
        opset_version=11,
    )

    # Save metadata alongside ONNX (threshold, channels, etc.)
    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "image_channels": in_channels,
        "num_features": num_features,
        "use_features": use_features,
        "optimal_threshold": metadata.get("optimal_threshold", 0.55),
        "model_size": metadata.get("model_size", "normal"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Exported CNN → {out_path}")
    print(f"     metadata   → {meta_path}")
    print(f"     channels={in_channels}, features={use_features}, threshold={meta['optimal_threshold']}")


def main():
    parser = argparse.ArgumentParser(description="Export models to portable formats")
    parser.add_argument("--lr-only", action="store_true", help="Only export LR model")
    parser.add_argument("--cnn-only", action="store_true", help="Only export CNN model")
    parser.add_argument("--lr-pkl", type=Path, default=TRAINED_DIR / "posture_model.pkl")
    parser.add_argument("--cnn-pt", type=Path, default=TRAINED_DIR / "posture_cnn.pt")
    parser.add_argument("--output-dir", type=Path, default=TRAINED_DIR)
    args = parser.parse_args()

    do_lr = not args.cnn_only
    do_cnn = not args.lr_only

    if do_lr:
        if args.lr_pkl.exists():
            export_lr_to_json(args.lr_pkl, args.output_dir / "posture_model.json")
        else:
            print(f"[SKIP] LR pickle not found: {args.lr_pkl}")

    if do_cnn:
        if args.cnn_pt.exists():
            export_cnn_to_onnx(args.cnn_pt, args.output_dir / "posture_cnn.onnx")
        else:
            print(f"[SKIP] CNN checkpoint not found: {args.cnn_pt}")


if __name__ == "__main__":
    main()
