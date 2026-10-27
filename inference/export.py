"""
Model export to ONNX and TorchScript.

ONNX enables inference without PyTorch, OpenCV DNN backend, and mobile/edge
deployment.  Includes a verification step checking output diff < 1e-5.
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_onnx(model, img_size: int, onnx_path: str, opset: int = 12):
    """Export PyTorch model to ONNX format.

    Parameters
    ----------
    model : nn.Module (eval mode)
    img_size : int
    onnx_path : str  output path
    opset : int  ONNX opset version
    """
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)

    # Determine output heads dynamically
    with torch.no_grad():
        out = model(dummy)
        has_landmarks = len(out) == 3 if isinstance(out, (tuple, list)) else False

    output_names = ["cls_preds", "reg_preds"]
    dynamic_axes = {
        "input": {0: "batch_size"},
        "cls_preds": {0: "batch_size"},
        "reg_preds": {0: "batch_size"},
    }
    if has_landmarks:
        output_names.append("lmk_preds")
        dynamic_axes["lmk_preds"] = {0: "batch_size"}

    # Export
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=opset,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"✅ ONNX exported to {onnx_path}")
    print(f"   File size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")


def export_torchscript(model, img_size: int, ts_path: str):
    """Export PyTorch model to TorchScript via tracing.

    Parameters
    ----------
    model : nn.Module (eval mode)
    img_size : int
    ts_path : str  output path
    """
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)

    traced = torch.jit.trace(model, dummy)
    traced.save(ts_path)
    print(f"✅ TorchScript exported to {ts_path}")
    print(f"   File size: {os.path.getsize(ts_path) / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_onnx(pytorch_model, onnx_path: str, img_size: int,
                tolerance: float = 1e-5) -> bool:
    """Verify ONNX output matches PyTorch output within tolerance.

    Parameters
    ----------
    pytorch_model : nn.Module
    onnx_path : str
    img_size : int
    tolerance : float

    Returns
    -------
    passed : bool
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  onnxruntime not installed — skipping ONNX verification")
        return False

    pytorch_model.eval()
    test_input = torch.randn(1, 3, img_size, img_size)

    # PyTorch forward
    with torch.no_grad():
        pt_outputs = pytorch_model(test_input)

    # ONNX inference
    session = ort.InferenceSession(onnx_path)
    ort_inputs = {"input": test_input.numpy()}
    ort_outputs = session.run(None, ort_inputs)

    # Compare
    all_close = True
    names = ["cls_preds", "reg_preds", "lmk_preds"]
    for i, (pt_out, ort_out) in enumerate(zip(pt_outputs, ort_outputs)):
        if pt_out is None:
            continue
        pt_np = pt_out.numpy()
        max_diff = np.max(np.abs(pt_np - ort_out))
        name = names[i] if i < len(names) else f"output_{i}"
        status = "✅" if max_diff < tolerance else "❌"
        print(f"  {status} {name}: max diff = {max_diff:.2e}")
        if max_diff >= tolerance:
            all_close = False

    if all_close:
        print("✅ ONNX verification PASSED")
    else:
        print("❌ ONNX verification FAILED — outputs differ beyond tolerance")

    return all_close


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export face detection model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--format", choices=["onnx", "torchscript", "both"],
                        default="both")
    parser.add_argument("--output-dir", default="exports")
    parser.add_argument("--verify", action="store_true",
                        help="Verify ONNX output matches PyTorch")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    img_size = cfg["model"]["image_size"]

    # Build and load model
    from training.train import FaceDetectionModel
    model = FaceDetectionModel(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {state.get('epoch', '?')}")

    # Export
    if args.format in ("onnx", "both"):
        onnx_path = os.path.join(args.output_dir, "face_detector.onnx")
        export_onnx(model, img_size, onnx_path)
        if args.verify:
            verify_onnx(model, onnx_path, img_size)

    if args.format in ("torchscript", "both"):
        ts_path = os.path.join(args.output_dir, "face_detector.pt")
        export_torchscript(model, img_size, ts_path)


if __name__ == "__main__":
    main()
