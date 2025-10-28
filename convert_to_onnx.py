import torch
from pathlib import Path
import timm
import json

# import custom modules
from model import ClassifierModel
from training.config import Config
from training.trainer import ModelConfig

train_config = Config()

MODEL_ROOT = Path(train_config.models_dir)
ONNX_FOLDER = Path('src/inference/models/onnx')
ONNX_FOLDER.mkdir(parents=True, exist_ok=True)


def load_custom_model(checkpoint_path: str)-> tuple[torch.nn.Module, ModelConfig]:
    model_config = ModelConfig.load_model(checkpoint_path)

    # Create backbone
    backbone = timm.create_model(
        model_config.model_name,
        pretrained=False,  # because we load weights right after
        in_chans=3,
        num_classes=len(model_config.labels),
    )

    model = ClassifierModel(
        backbone=backbone,
        class_freq=model_config.class_frequency,
        tau=model_config.tau_logit_adjust,
    )
    model.load_state_dict(model_config.model_state_dict, strict=False)

    print(f"[✓] Loaded model from {checkpoint_path}")
    print(f"Labels: {len(model_config.labels)} | Image size: {model_config.image_size}")

    return model, model_config

def save_labels(model_name: str, labels: list[str]) -> None:
    labels_path = ONNX_FOLDER / f"{model_name}.json"
    with open(labels_path, 'w') as f:
        json.dump(labels, f)
    print(f"[✓] Saved labels to {labels_path}")

def export_model(
        model: torch.nn.Module,
        image_size: int,
        model_name: str="model",
):
    model.eval().to("cuda")
    input_shape = (1, 3, image_size, image_size)

    onnx_path = ONNX_FOLDER / f"{model_name}.onnx"
    torch.onnx.export(
        model,
        torch.randn(*input_shape, device="cuda"),
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

if __name__ == "__main__":
    model_name = 'wbce_action_convnextv2_tiny_32_384'
    model, model_config = load_custom_model(str(MODEL_ROOT / f'{model_name}.pth'))
    save_labels(model_name, model_config.labels)
    export_path = export_model(
        model=model,
        image_size=model_config.image_size,
        model_name=model_config.model_name,
    )
