import os
import json
import warnings
from pathlib import Path
from typing import List
from tqdm import tqdm

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    from train import TemporalFusionTransformer, TimeSeriesDataset, CONFIG as TRAIN_CONFIG
except ImportError:
    try:
        from Irrigation.train import TemporalFusionTransformer, TimeSeriesDataset, CONFIG as TRAIN_CONFIG
    except ImportError:
        from irrigation.train import TemporalFusionTransformer, TimeSeriesDataset, CONFIG as TRAIN_CONFIG

CONFIG = {
    "seed": 42,
    "top_k_checkpoints": 3,
    "checkpoint_dir": "Final/irrigation/checkpoints",
    "metrics_json_v2": "Final/irrigation/metrics_v2.json",
    "metrics_json_v1": "Final/irrigation/metrics.json",
    "val_csv_v2": "Final/irrigation/data/cleaned_val_v2.csv",
    "val_csv_v1": "Final/irrigation/data/cleaned_val.csv",
    "output_csv_v2": "Final/irrigation/results_v2.csv",
    "output_csv_v1": "Final/irrigation/results.csv",
}


def run_irrigation_inference() -> None:
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

    device = get_optimal_device()
    first_ckpt = torch.load(ckpt_files[0], map_location=device)
    feature_cols = first_ckpt["feature_cols"]

    features = val_df[feature_cols].values
    targets = val_df["irrigation_amount"].values

    v_ds = TimeSeriesDataset(features, targets, seq_len=TRAIN_CONFIG["seq_len"])
    v_loader = DataLoader(v_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)

    models = []
    for ckpt_path in tqdm(ckpt_files, desc="Loading TFT Checkpoints", unit="ckpt"):
        model = TemporalFusionTransformer(len(feature_cols), TRAIN_CONFIG["hidden_dim"], TRAIN_CONFIG["n_heads"]).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append(model)

    print(f"Loaded Top {len(models)} fold models for Irrigation TFT inference.")

    fold_preds = []
    actuals = []

    with torch.no_grad():
        for model in tqdm(models, desc="TFT Ensemble Inference", unit="model"):
            preds_list = []
            act_list = []
            try:
                for bx, by in v_loader:
                    bx, by = bx.to(device), by.to(device)
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        out, _ = model(bx)
                    preds_list.extend(out.cpu().numpy())
                    act_list.extend(by.cpu().numpy())
            except Exception as e:
                if is_oom_error(e):
                    device, _, _, _ = switch_to_cpu(reason="GPU VRAM exceeded during TFT inference")
                    models = [m.to(device) for m in models]
                    preds_list = []
                    act_list = []
                    for bx, by in v_loader:
                        bx, by = bx.to(device), by.to(device)
                        out, _ = model(bx)
                        preds_list.extend(out.cpu().numpy())
                        act_list.extend(by.cpu().numpy())
                else:
                    raise e

            fold_preds.append(preds_list)
            actuals = act_list

            if torch.cuda.is_available():
                clear_gpu_cache()

    avg_preds = np.mean(fold_preds, axis=0)

    results_df = val_df.iloc[TRAIN_CONFIG["seq_len"] - 1:].copy()
    results_df["predicted_irrigation_amount"] = avg_preds

    out_csv_v2 = Path(CONFIG["output_csv_v2"])
    out_csv_v1 = Path(CONFIG["output_csv_v1"])
    out_csv_v2.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv_v2, index=False)
    results_df.to_csv(out_csv_v1, index=False)

    rmse = float(np.sqrt(mean_squared_error(actuals, avg_preds)))
    mae = float(mean_absolute_error(actuals, avg_preds))

    print(f"\n==========================================")
    print(f"IRRIGATION INFERENCE RESULTS (TOP-{CONFIG['top_k_checkpoints']} ENSEMBLE) [v2]:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"==========================================")


if __name__ == "__main__":
    run_irrigation_inference()

