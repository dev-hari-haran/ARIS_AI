import os
import gc
import sys
import warnings
from pathlib import Path
from typing import Tuple, Optional, Any

# Ensure project root cache directory is used for ALL caches
WORKSPACE_ROOT = Path(__file__).resolve().parent
CACHE_DIR = WORKSPACE_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set environment variables for caches to remain strictly inside workspace folder
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["TORCH_HUB"] = str(CACHE_DIR / "torch_hub")
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_UPDATE_CHECK"] = "1"
os.environ["ULTRALYTICS_CONFIG_DIR"] = str(CACHE_DIR / "ultralytics")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Enable CUDA devices
if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "":
    del os.environ["CUDA_VISIBLE_DEVICES"]

import torch

FORCE_CPU = False

# Enable CUDA performance optimizations if GPU available
if torch.cuda.is_available():
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

def clear_gpu_cache() -> None:
    """
    Clears Python garbage collector and CUDA memory cache safely.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def get_vram_info() -> Tuple[float, float]:
    """
    Returns free and total VRAM in MB.
    """
    if torch.cuda.is_available():
        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            free = total - max(reserved, allocated)
            return free, total
        except Exception:
            pass
    return 0.0, 0.0

def check_vram_available(min_free_mb: float = 250.0) -> bool:
    if torch.cuda.is_available():
        free, _ = get_vram_info()
        return free >= min_free_mb
    return False

def get_optimal_device(min_free_vram_mb: float = 250.0, force_cpu: bool = False) -> torch.device:
    """
    Returns CUDA device if GPU available and force_cpu is False, else CPU.
    """
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def is_oom_error(e: Exception) -> bool:
    err_msg = str(e).lower()
    return (
        isinstance(e, torch.cuda.OutOfMemoryError)
        or "out of memory" in err_msg
        or "cuda" in err_msg
        or "memory" in err_msg
        or "alloc" in err_msg
    )

def switch_to_cpu(
    model: Optional[torch.nn.Module] = None,
    criterion: Optional[torch.nn.Module] = None,
    tensors: Optional[list] = None,
    reason: str = "Enforced CPU mode"
) -> Tuple[torch.device, Optional[torch.nn.Module], Optional[torch.nn.Module], list]:
    clear_gpu_cache()
    cpu_device = torch.device("cpu")
    if model is not None:
        try:
            model = model.to(cpu_device)
        except Exception:
            pass
    if criterion is not None:
        try:
            criterion = criterion.to(cpu_device)
        except Exception:
            pass
    moved_tensors = []
    if tensors:
        for t in tensors:
            if isinstance(t, torch.Tensor):
                moved_tensors.append(t.to(cpu_device))
            else:
                moved_tensors.append(t)
    return cpu_device, model, criterion, moved_tensors
