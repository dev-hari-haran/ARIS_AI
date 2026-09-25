import os
import pickle
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
from torchvision import transforms
from sklearn.metrics import f1_score

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
    from train import NutrientFusionModel, CONFIG as TRAIN_CONFIG
except ImportError:
    try:
        from Nutrient.train import NutrientFusionModel, CONFIG as TRAIN_CONFIG
    except ImportError:
        from nutrient.train import NutrientFusionModel, CONFIG as TRAIN_CONFIG

CONFIG = {
    "seed": 42,
    "checkpoint_dir": "Final/nutrient/checkpoints",
    "val_csv": "Final/nutrient/data/processed_val.csv",
    "output_csv": "Final/nutrient/results.csv",
}


def run_nutrient_inference() -> None:
    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    val_csv_path = Path(CONFIG["val_csv"])

    if not val_csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found at {val_csv_path}")

    val_df = pd.read_csv(val_csv_path)

    with open(ckpt_dir / "xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    device = get_optimal_device()

    npk_cols = [c for c in val_df.columns if c not in ["image_path", "label", "class_id", "stem"]]
    npk_features = val_df[npk_cols].values
    xgb_probs_val = xgb_model.predict_proba(npk_features)

    ckpts = sorted(list(ckpt_dir.glob("fold_*.pth")))
    if not ckpts:
        raise FileNotFoundError(f"No model checkpoints found in {ckpt_dir}")

    first_ckpt = torch.load(ckpts[0], map_location=device)
    classes = first_ckpt["classes"]
    npk_input_dim = first_ckpt["npk_input_dim"]
    num_classes = len(classes)

    transform_val = transforms.Compose([
        transforms.Resize(TRAIN_CONFIG["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    models = []
    for ckpt_path in tqdm(ckpts, desc="Loading Checkpoint Models", unit="model"):
        model = NutrientFusionModel(npk_input_dim, num_classes).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        models.append(model)

    print(f"Loaded {len(models)} fold models for Nutrient inference.")

    sample_probs = []

    with torch.no_grad():
        for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Nutrient Multimodal Inference", unit="sample"):
            img_path = row["image_path"]
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.fromarray(np.zeros((TRAIN_CONFIG["image_size"][0], TRAIN_CONFIG["image_size"][1], 3), dtype=np.uint8))

            try:
                img_tensor = transform_val(img).unsqueeze(0).to(device)
                npk_tensor = torch.tensor(npk_features[idx:idx+1], dtype=torch.float32).to(device)
                xgb_tensor = torch.tensor(xgb_probs_val[idx:idx+1], dtype=torch.float32).to(device)

                fold_logits = []
                for model in models:
                    logits = model(img_tensor, npk_tensor, xgb_tensor)
                    probs = F.softmax(logits, dim=-1)
                    fold_logits.append(probs.cpu())
            except Exception as e:
                if is_oom_error(e):
                    device, _, _, _ = switch_to_cpu(reason="GPU VRAM exceeded during Nutrient inference")
                    models = [m.to(device) for m in models]
                    img_tensor = transform_val(img).unsqueeze(0).to(device)
                    npk_tensor = torch.tensor(npk_features[idx:idx+1], dtype=torch.float32).to(device)
                    xgb_tensor = torch.tensor(xgb_probs_val[idx:idx+1], dtype=torch.float32).to(device)
                    fold_logits = []
                    for model in models:
                        logits = model(img_tensor, npk_tensor, xgb_tensor)
                        probs = F.softmax(logits, dim=-1)
                        fold_logits.append(probs.cpu())
                else:
                    raise e

            avg_prob = torch.stack(fold_logits, dim=0).mean(dim=0).squeeze(0).numpy()
            sample_probs.append(avg_prob)
            if torch.cuda.is_available():
                clear_gpu_cache()

    sample_probs = np.array(sample_probs)
    preds = np.argmax(sample_probs, axis=-1)

    results_df = val_df.copy()
    results_df["predicted_class_id"] = preds
    results_df["predicted_label"] = [classes[p] for p in preds]
    results_df["confidence"] = np.max(sample_probs, axis=-1)

    out_csv = Path(CONFIG["output_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)

    if "class_id" in val_df.columns:
        targets = val_df["class_id"].values
        f1 = f1_score(targets, preds, average="macro")
        print(f"\n==========================================")
        print(f"NUTRIENT FUSION INFERENCE FINAL F1 MACRO: {f1:.4f}")
        print(f"==========================================")
    else:
        print(f"Inference results saved to {out_csv}")


if __name__ == "__main__":
    run_nutrient_inference()
