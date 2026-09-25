import os
import json
import glob
import warnings
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import sys
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from device_utils import clear_gpu_cache, get_optimal_device, is_oom_error, switch_to_cpu
except ImportError:
    def clear_gpu_cache():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    def get_optimal_device(min_free_mb=250.0):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    def is_oom_error(e):
        return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
    def switch_to_cpu(model=None, criterion=None, tensors=None, reason="VRAM exceeded"):
        clear_gpu_cache()
        return torch.device("cpu"), model, criterion, []

try:
    from train import RepViTClassifier, CONFIG as TRAIN_CONFIG
except ImportError:
    try:
        from Disease.train import RepViTClassifier, CONFIG as TRAIN_CONFIG
    except ImportError:
        from disease.train import RepViTClassifier, CONFIG as TRAIN_CONFIG

CONFIG = {
    "seed": 42,
    "tta_passes": 5,
    "top_k_checkpoints": 3,
    "checkpoint_dir": "Final/disease/checkpoints",
    "metrics_json_v2": "Final/disease/metrics_v2.json",
    "metrics_json_v1": "Final/disease/metrics.json",
    "val_csv_v2": "Final/disease/data/processed_val_v2.csv",
    "val_csv_v1": "Final/disease/data/processed_val.csv",
    "output_csv_v2": "Final/disease/results_v2.csv",
    "output_csv_v1": "Final/disease/results.csv",
}


def get_tta_transforms(img_size: Tuple[int, int]) -> List[transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tta_list = [
        transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize(img_size), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize(img_size), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize(img_size), transforms.RandomRotation(degrees=(15, 15)), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize(img_size), transforms.ColorJitter(brightness=0.1), transforms.ToTensor(), normalize]),
    ]
    return tta_list[:CONFIG["tta_passes"]]


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    img_path: str,
    tta_transforms: List[transforms.Compose],
    img_size: Tuple[int, int],
    device: torch.device
) -> torch.Tensor:
    try:
        raw_img = Image.open(img_path).convert("RGB")
    except Exception:
        raw_img = Image.fromarray(np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8))

    probs_list = []
    for t_transform in tta_transforms:
        tensor_img = t_transform(raw_img).unsqueeze(0).to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(tensor_img)
            probs = F.softmax(logits, dim=-1)
        probs_list.append(probs.cpu())

    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
    return avg_probs


def run_disease_inference() -> None:
    device = get_optimal_device()
    ckpt_dir = Path(CONFIG["checkpoint_dir"])

    val_csv_path = Path(CONFIG["val_csv_v2"]) if Path(CONFIG["val_csv_v2"]).exists() else Path(CONFIG["val_csv_v1"])
    metrics_path = Path(CONFIG["metrics_json_v2"]) if Path(CONFIG["metrics_json_v2"]).exists() else Path(CONFIG["metrics_json_v1"])

    if not val_csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found at {val_csv_path}")

    val_df = pd.read_csv(val_csv_path)

    ckpt_files = []
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                meta = json.load(f)
                ckpt_files = [Path(p) for p in meta.get("top_checkpoints", []) if Path(p).exists()]
        except Exception:
            pass

    if not ckpt_files:
        v2_ckpts = sorted(list(ckpt_dir.glob("*_v2.pth")))
        all_ckpts = v2_ckpts if v2_ckpts else sorted(list(ckpt_dir.glob("fold_*.pth")))
        ckpt_files = all_ckpts[:CONFIG["top_k_checkpoints"]]

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    first_ckpt = torch.load(ckpt_files[0], map_location=device)
    classes = first_ckpt["classes"]
    img_size = first_ckpt.get("img_size", (192, 192))
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    num_classes = len(classes)

    models = []
    for ckpt_path in tqdm(ckpt_files, desc="Loading Checkpoint Models", unit="model"):
        model = RepViTClassifier(num_classes).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        models.append(model)

    print(f"Loaded Top {len(models)} fold models for 5-Pass TTA Disease inference.")

    tta_transforms = get_tta_transforms(img_size)
    ensemble_probs = []

    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Disease TTA Inference", unit="sample"):
        img_path = row["filepath"]
        sample_fold_probs = []

        try:
            for model in models:
                sample_probs = predict_with_tta(model, img_path, tta_transforms, img_size, device)
                sample_fold_probs.append(sample_probs)
        except Exception as e:
            if is_oom_error(e):
                device, _, _, _ = switch_to_cpu(reason="GPU VRAM exceeded during TTA inference")
                models = [m.to(device) for m in models]
                sample_fold_probs = []
                for model in models:
                    sample_probs = predict_with_tta(model, img_path, tta_transforms, img_size, device)
                    sample_fold_probs.append(sample_probs)
            else:
                raise e

        avg_sample_prob = torch.stack(sample_fold_probs, dim=0).mean(dim=0).squeeze(0).numpy()
        ensemble_probs.append(avg_sample_prob)

        if torch.cuda.is_available():
            clear_gpu_cache()

    ensemble_probs = np.array(ensemble_probs)
    preds = np.argmax(ensemble_probs, axis=-1)
    pred_labels = [classes[p] for p in preds]

    results_df = val_df.copy()
    results_df["predicted_class_id"] = preds
    results_df["predicted_label"] = pred_labels
    results_df["confidence"] = np.max(ensemble_probs, axis=-1)

    out_v2 = Path(CONFIG["output_csv_v2"])
    out_v1 = Path(CONFIG["output_csv_v1"])
    out_v2.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_v2, index=False)
    results_df.to_csv(out_v1, index=False)

    if "class_id" in val_df.columns:
        targets = val_df["class_id"].values
        final_f1 = f1_score(targets, preds, average="macro")
        print(f"\n==========================================")
        print(f"DISEASE INFERENCE (5-TTA TOP-3 ENSEMBLE) F1 MACRO: {final_f1:.4f} [v2]")
        print(f"==========================================")


if __name__ == "__main__":
    run_disease_inference()

