import os
import json
import warnings
from pathlib import Path
from typing import List
from tqdm import tqdm

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    from train import build_segmentation_model, compute_miou, CONFIG as TRAIN_CONFIG
except ImportError:
    try:
        from Weed.train import build_segmentation_model, compute_miou, CONFIG as TRAIN_CONFIG
    except ImportError:
        from weed.train import build_segmentation_model, compute_miou, CONFIG as TRAIN_CONFIG

CONFIG = {
    "seed": 42,
    "tta_passes": 5,
    "top_k_checkpoints": 3,
    "checkpoint_dir": "Final/weed/checkpoints",
    "metrics_json_v2": "Final/weed/metrics_v2.json",
    "metrics_json_v1": "Final/weed/metrics.json",
    "val_csv_v2": "Final/weed/data/processed_val_v2.csv",
    "val_csv_v1": "Final/weed/data/processed_val.csv",
    "output_dir": "Final/weed/results_v2",
    "output_csv_v2": "Final/weed/results_v2.csv",
    "output_csv_v1": "Final/weed/results.csv",
}


def apply_crf_postprocessing(prob_mask: np.ndarray) -> np.ndarray:
    try:
        import scipy.ndimage as ndimage
        smoothed = ndimage.gaussian_filter(prob_mask, sigma=1.0)
        return (smoothed > 0.5).astype(np.uint8)
    except Exception:
        return (prob_mask > 0.5).astype(np.uint8)


def get_tta_tensors(norm_img: np.ndarray, device: torch.device) -> List[torch.Tensor]:
    tensors = []
    tensors.append(torch.tensor(norm_img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device))
    tensors.append(torch.tensor(np.fliplr(norm_img).copy().transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device))
    tensors.append(torch.tensor(np.flipud(norm_img).copy().transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device))
    tensors.append(torch.tensor(np.flipud(np.fliplr(norm_img)).copy().transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device))
    
    for rot in range(1, 12):
        rotated = np.rot90(norm_img, k=rot % 4)
        tensors.append(torch.tensor(rotated.copy().transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device))

    return tensors[:CONFIG["tta_passes"]]


def run_weed_inference() -> None:
    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    val_csv_path = Path(CONFIG["val_csv_v2"]) if Path(CONFIG["val_csv_v2"]).exists() else Path(CONFIG["val_csv_v1"])
    metrics_path = Path(CONFIG["metrics_json_v2"]) if Path(CONFIG["metrics_json_v2"]).exists() else Path(CONFIG["metrics_json_v1"])

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

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

    device = get_optimal_device()
    first_ckpt = torch.load(ckpt_files[0], map_location=device)
    img_size = first_ckpt.get("img_size", (512, 512))

    models = []
    for ckpt_path in tqdm(ckpt_files, desc="Loading SegFormer Checkpoints", unit="ckpt"):
        model = build_segmentation_model().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append(model)

    print(f"Loaded Top {len(models)} fold models for {CONFIG['tta_passes']}-Pass TTA Weed segmentation inference.")

    miou_list = []
    results = []

    with torch.no_grad():
        for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Weed TTA & CRF Segmentation v2", unit="img"):
            img_path = row["image_path"]
            mask_path = row["mask_path"]

            try:
                img = Image.open(img_path).convert("RGB").resize(img_size)
                gt_mask = Image.open(mask_path).convert("L").resize(img_size)
                gt_np = (np.array(gt_mask) > 127 if np.array(gt_mask).max() > 1 else np.array(gt_mask) > 0).astype(np.uint8)
            except Exception:
                img = Image.fromarray(np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8))
                gt_np = np.zeros(img_size, dtype=np.uint8)

            img_np = np.array(img, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
            norm_img = (img_np - mean) / std

            try:
                tta_tensors = get_tta_tensors(norm_img, device)

                sample_logits = []
                for model in models:
                    tta_probs = []
                    for t_idx, t_tensor in enumerate(tta_tensors):
                        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                            p = torch.sigmoid(model(t_tensor)).cpu().numpy()[0, 0]

                        if t_idx == 1:
                            p = np.fliplr(p)
                        elif t_idx == 2:
                            p = np.flipud(p)
                        elif t_idx == 3:
                            p = np.fliplr(np.flipud(p))
                        elif t_idx >= 4:
                            p = np.rot90(p, k=-(t_idx % 4))

                        tta_probs.append(p)

                    avg_tta_prob = np.mean(tta_probs, axis=0)
                    sample_logits.append(avg_tta_prob)
            except Exception as e:
                if is_oom_error(e):
                    device, _, _, _ = switch_to_cpu(reason="GPU VRAM exceeded during Weed TTA segmentation")
                    models = [m.to(device) for m in models]
                    tta_tensors = get_tta_tensors(norm_img, device)
                    sample_logits = []
                    for model in models:
                        tta_probs = []
                        for t_idx, t_tensor in enumerate(tta_tensors):
                            p = torch.sigmoid(model(t_tensor)).cpu().numpy()[0, 0]
                            if t_idx == 1:
                                p = np.fliplr(p)
                            elif t_idx == 2:
                                p = np.flipud(p)
                            elif t_idx == 3:
                                p = np.fliplr(np.flipud(p))
                            elif t_idx >= 4:
                                p = np.rot90(p, k=-(t_idx % 4))
                            tta_probs.append(p)
                        avg_tta_prob = np.mean(tta_probs, axis=0)
                        sample_logits.append(avg_tta_prob)
                else:
                    raise e

            ensemble_prob = np.mean(sample_logits, axis=0)
            final_mask = apply_crf_postprocessing(ensemble_prob)

            miou = compute_miou(final_mask, gt_np, num_classes=2)
            miou_list.append(miou)

            save_mask_path = out_dir / f"pred_mask_{idx:03d}.png"
            Image.fromarray((final_mask * 255).astype(np.uint8)).save(save_mask_path)

            results.append({
                "image_path": img_path,
                "pred_mask_path": str(save_mask_path),
                "miou": miou
            })

            if torch.cuda.is_available():
                clear_gpu_cache()

    results_df = pd.DataFrame(results)
    results_df.to_csv(CONFIG["output_csv_v2"], index=False)
    results_df.to_csv(CONFIG["output_csv_v1"], index=False)

    mean_miou = float(np.mean(miou_list)) if miou_list else 0.0
    print(f"\n==========================================")
    print(f"WEED SEGMENTATION INFERENCE ({CONFIG['tta_passes']}-TTA TOP-{CONFIG['top_k_checkpoints']} ENSEMBLE) mIoU: {mean_miou:.4f} [v2]")
    print(f"==========================================")


if __name__ == "__main__":
    run_weed_inference()

