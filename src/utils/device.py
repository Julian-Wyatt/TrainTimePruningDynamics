import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def gpu_info() -> dict[str, str | int]:
    """Return stable metadata for the currently selected CUDA GPU."""
    if not torch.cuda.is_available():
        return {"device": get_device(), "gpu_name": "none", "gpu_family": "none"}

    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    return {
        "device": "cuda",
        "gpu_index": idx,
        "gpu_name": name,
        "gpu_family": gpu_family(name),
    }


def gpu_family(name: str) -> str:
    """Normalize detailed CUDA device names to a groupable GPU family."""
    normalized = " ".join(str(name).upper().replace("_", " ").replace("-", " ").split())
    families = (
        "B200",
        "B100",
        "H200",
        "H100",
        "A100",
        "L40S",
        "L40",
        "V100",
        "T4",
        "P100",
    )
    for family in families:
        if family in normalized:
            return family

    # Fallback for common GeForce/RTX cards where the model number is the family.
    parts = normalized.split()
    for idx, part in enumerate(parts):
        if part in {"RTX", "GTX"} and idx + 1 < len(parts):
            return f"{part} {parts[idx + 1]}"
    return normalized or "unknown"


def precision_dtype(precision: str) -> torch.dtype:
    return (
        torch.bfloat16 if precision == "bf16"
        else torch.float16 if precision == "fp16"
        else torch.float32
    )


def unwrap_model(model):
    from torch.nn.parallel import DistributedDataParallel
    while True:
        if isinstance(model, DistributedDataParallel):
            model = model.module
            continue
        orig_mod = getattr(model, "_orig_mod", None)
        if orig_mod is not None:
            model = orig_mod
            continue
        return model
