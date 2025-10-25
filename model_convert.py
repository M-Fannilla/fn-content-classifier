import torch
from pathlib import Path
import timm
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


def load_custom_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = checkpoint.get("model_name")
    
    if not model_name:
        raise Exception("Model name not found")
        
    labels = checkpoint.get("labels_columns")
    image_size = checkpoint.get("image_size")
    threshold = checkpoint.get("threshold")
    optimal_thresholds = checkpoint.get("optimal_thresholds")

    model = timm.create_model(model_name, pretrained=False, num_classes=len(labels))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    print(f"[✓] Loaded model from {checkpoint_path}")
    print(f"Labels: {len(labels)} | Image size: {image_size}")
    
    # Use optimal thresholds if available, otherwise use static threshold
    if optimal_thresholds is not None:
        print(f"Using optimal thresholds: {optimal_thresholds}")
        return model, labels, image_size, optimal_thresholds
    else:
        print(f"Using static threshold: {threshold}")
        return model, labels, image_size, threshold

def save_labels(model_name: str, labels: dict[int, str]):
    labels_path = Path(f'./onnx/{model_name}_labels.json')
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(labels_path, 'w') as f:
        json.dump(labels, f)
    print(f"[✓] Saved labels to {labels_path}")

def export_model(
        model: torch.nn.Module,
        input_shape=(1, 3, 224, 224),
        model_name="model",
):
    export_dir = './onnx'
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    model.eval().to("cuda")

    dummy_input = torch.randn(*input_shape, device="cuda")

    # ===== ONNX EXPORT =====
    onnx_path = f"{export_dir}/{model_name}.onnx"
    if Path(onnx_path).exists():
        print(f"[✓] ONNX model already exists: {onnx_path}")
        return onnx_path

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

    return onnx_path

def process_image_pil(image_path: str, image_size: int = 224):
    image = Image.open(image_path).convert("RGB")

    # Resize image proportionally and add black padding
    original_width, original_height = image.size

    # Calculate scale factor to fit within target_size while maintaining aspect ratio
    scale = min(image_size / original_width, image_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize image proportionally
    image = image.resize((new_width, new_height), Image.BILINEAR)

    # Create black canvas
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))

    # Calculate position to paste (center the image)
    paste_x = (image_size - new_width) // 2
    paste_y = (image_size - new_height) // 2

    # Paste resized image onto black canvas
    canvas.paste(image, (paste_x, paste_y))

    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(canvas).astype(np.float32) / 255.0

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_array = (image_array - mean) / std

    # Convert HWC to CHW format
    image_array = np.transpose(image_array, (2, 0, 1))

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

def infer_onnx(
    model_path: str,
    labels: list[str],
    img: np.ndarray,
    threshold,
):
    providers = [
        # "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
    ort_sess = ort.InferenceSession(model_path, providers=providers)
    input_name = ort_sess.get_inputs()[0].name

    outputs = ort_sess.run(None, {input_name: img})
    image_probs = outputs[0].flatten()
    image_probs = 1 / (1 + np.exp(-image_probs))

    # Handle both static threshold (float) and dynamic thresholds (array)
    if isinstance(threshold, (int, float)):
        # Static threshold
        labeled_probs = {
            label: round(float(weight), 3)
            for label, weight in zip(labels, image_probs)
            if weight >= threshold
        }
    else:
        # Dynamic thresholds (array)
        labeled_probs = {
            label: round(float(weight), 3)
            for i, (label, weight) in enumerate(zip(labels, image_probs))
            if i < len(threshold) and weight >= threshold[i]
        }

    return labeled_probs


if __name__ == "__main__":
    # model_root = Path('./models')
    # model_root.mkdir(parents=True, exist_ok=True)
    # model_name = 'action_convnextv2_nano_32_384'
    #
    # model, labels, image_size, threshold = load_custom_model(
    #     model_root / f'{model_name}.pth',
    # )
    #
    # print(f"Model: {model_name}")
    # print(f"Labels: {labels}")
    # print(f"Image size: {image_size}")
    # print(f"Threshold: {threshold}")
    #
    # save_labels(model_name, labels)
    #
    # export_path = export_model(
    #     model=model,
    #     input_shape=(1, 3, image_size, image_size),
    #     model_name=model_name,
    # )

    image_size = 384
    action_path = 'onnx/action_convnextv2_nano_32_384.onnx'
    bodypart_path = 'onnx/bodyparts_convnextv2_nano_32_384.onnx'
    action_labels = ["69", "anal_fucking", "ass_licking", "ass_penetration", "fingering", "grabbing_ass", "grabbing_boobs", "grabbing_hair/head", "handjob", "kissing", "masturbation", "pussy_rubbing", "vaginal_fucking", "vaginal_penetration", "vibrating", "wet_genitals", "blowjob", "cum", "pussy_licking"]
    bodyparts_labels = ["asshole", "belly_button", "cum", "hairy_pussy", "labia", "wet_pussy", "tongue", "ass", "balls", "boobs", "brown_pussy", "dick", "feet", "lower_legs", "nipples", "pussy", "pink_pussy", "pussy_closeup", "shaved_pussy"]
    threshold = 0.4

    for model_path, labels in zip([action_path, bodypart_path], [action_labels, bodyparts_labels]):
        print("inferring", model_path)

        for image_path in ['./0.jpg', './1.jpg']:
            img = preprocess_image_cv2(image_path, image_size)
            probs = infer_onnx(
                model_path=model_path,
                img=img,
                labels=labels,
                threshold=threshold,
            )

            print(f"\nCV2 Image: {image_path}")
            for label, prob in probs.items():
                print(f"{label}: {prob}")
            print()

            img = process_image_pil(image_path, image_size)
            probs = infer_onnx(
                model_path=model_path,
                img=img,
                labels=labels,
                threshold=threshold,
            )

            print(f"\nPIL Image: {image_path}")
            for label, prob in probs.items():
                print(f"{label}: {prob}")
            print()

        print("-----")

    # import subprocess
    #
    # tensorrt_path = Path('./tensorrt') / f'{model_name}.trt'
    # tensorrt_path.mkdir(parents=True, exist_ok=True)
    #
    # subprocess.run([
    #     'trtexec',
    #     f'--onnx={export_path}',
    #     '--saveEngine=my_model.trt',
    #     '--workspace=1G',
    #     '--fp16',
    # ])
