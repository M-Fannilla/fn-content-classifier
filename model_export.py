import torch
from torch2trt import torch2trt
from pathlib import Path

def export_model(
        model: torch.nn.Module,
        input_shape=(1, 3, 224, 224),
        export_dir="exports",
        model_name="model",
        precision="fp16"
):
    """
    Exports a PyTorch model to ONNX and TensorRT.

    Args:
        model (torch.nn.Module): Trained model.
        input_shape (tuple): Shape of dummy input (N, C, H, W).
        export_dir (str): Directory to save exports.
        model_name (str): Base name for output files.
        precision (str): 'fp16' or 'fp32' for TensorRT.
    """
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    model.eval().to("cuda")

    dummy_input = torch.randn(*input_shape, device="cuda")

    # ===== ONNX EXPORT =====
    onnx_path = f"{export_dir}/{model_name}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    print(f"[✓] Saved ONNX model: {onnx_path}")

    # ===== TENSORRT EXPORT =====
    # Requires: pip install torch2trt nvidia-tensorrt
    model_trt = torch2trt(
        model,
        [dummy_input],
        fp16_mode=(precision == "fp16"),
        log_level=torch2trt.logging.ERROR
    )
    trt_path = f"{export_dir}/{model_name}_{precision}.trt"
    torch.save(model_trt.state_dict(), trt_path)
    print(f"[✓] Saved TensorRT engine weights: {trt_path}")

    return onnx_path, trt_path