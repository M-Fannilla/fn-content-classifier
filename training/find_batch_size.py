import torch
import gc
from timm import create_model
import pynvml

from training.config import Config


def _get_gpu_memory(device="cuda"):
    """Get total and free GPU memory in MB."""
    if device != "cuda" or not torch.cuda.is_available():
        return None, None

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total = info.total / (1024 ** 2)
    free = info.free / (1024 ** 2)
    return int(total), int(free)


def _find_max_batch_size(
    model,
    input_size,
    device="cuda",
    start=4,
    amp=True,
):
    """
    Binary search for max batch size that fits in memory.
    Auto-bounds based on GPU memory.
    """
    total_mem, free_mem = _get_gpu_memory(device)
    if total_mem:
        print(f"GPU Memory: total={total_mem}MB, free={free_mem}MB")

        # Rough heuristic: keep ~20% headroom, estimate via input shape
        _, H, W = input_size
        approx_batch_mem = (H * W * 3 * 4) / (1024 ** 2)  # bytes → MB
        est_upper_bs = int((free_mem * 0.8) / approx_batch_mem)
        max_search = max(start * 2, est_upper_bs)
        print(f"Auto-estimated search upper bound: {max_search}")
    else:
        max_search = 1024  # generic fallback CPU/sim mode

    model.to(device)
    torch.cuda.empty_cache()
    gc.collect()

    C, H, W = input_size
    low, high = start, max_search
    max_bs = start

    while low <= high:
        mid = (low + high) // 2

        try:
            dummy = torch.randn(mid, C, H, W, device=device)
            with torch.cuda.amp.autocast(enabled=amp):
                out = model(dummy)
                loss = out.sum()
            loss.backward()

            max_bs = mid
            low = mid + 1

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            high = mid - 1

        finally:
            model.zero_grad(set_to_none=True)
            del dummy
            gc.collect()
            torch.cuda.empty_cache()

    return max_bs


def _suggest_grad_accumulation(max_micro_bs, target_eff_bs=1024):
    return max(1, target_eff_bs // max_micro_bs)


def find_batch_size(
    config: Config,
    num_classes: int,
    device="cuda",
    target_eff_bs=1024,
    amp=True,
) -> tuple[int, int]:
    """Compute optimal micro-batch + grad accumulation."""
    if device != "cuda" or not torch.cuda.is_available():
        print("Non-CUDA device detected; skipping batch size search.")
        return config.batch_size or 32, 1

    print("Finding optimal batch size and gradient accumulation...")
    img_size = config.img_size

    model = create_model(
        config.model_name,
        pretrained=False,  # AMP influences activations not params
        num_classes=num_classes,
    )

    input_size = (3, img_size, img_size)

    max_micro_bs = _find_max_batch_size(
        model=model,
        input_size=input_size,
        device=device,
        start=16,
        amp=amp,
    )
    print(f"Max micro-batch fitting GPU: {max_micro_bs}")

    grad_acc = _suggest_grad_accumulation(
        max_micro_bs,
        target_eff_bs=target_eff_bs,
    )
    print(f"Recommended grad_accumulation: {grad_acc}")

    return max_micro_bs, grad_acc