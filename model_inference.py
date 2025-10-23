import onnxruntime as ort
import torch
from torch2trt import TRTModule

def infer_onnx(model_path, input_tensor: torch.Tensor):
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)

    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    return torch.tensor(ort_outs[0])


def infer_trt(model_path, input_tensor: torch.Tensor):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(model_path))
    model_trt.eval().to("cuda")

    with torch.no_grad():
        output = model_trt(input_tensor.to("cuda"))
    return output


# if __name__ == "__main__":
#     model_path = "./model.onnx"
#
#     # 1. Create and export
#     model = create_model("convnextv2_base", pretrained=True, num_classes=10)
#     onnx_path, trt_path = export_model(model, input_shape=(1, 3, 224, 224))
#
#     # 2. Run inference (ONNX)
#     x = torch.randn(1, 3, 224, 224)
#     y_onnx = infer_onnx(onnx_path, x)
#
#     # 3. Run inference (TensorRT)
#     y_trt = infer_trt(trt_path, x)
#
#     print("ONNX output shape:", y_onnx.shape)
#     print("TRT output shape:", y_trt.shape)