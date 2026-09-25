import os
import json
import random
import shutil
import sys
import warnings
from pathlib import Path
from typing import Tuple, List, Dict
from tqdm import tqdm

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

CONFIG = {
    "seed": 42,
    "num_folds": 3,
    "epochs": 30,
    "early_stopping": 7,
    "patience": 7,
    "batch_size": 32,
    "lr": 1e-3,
    "seq_len": 7,
    "hidden_dim": 64,
    "n_heads": 4,
    "checkpoint_prefix": "v2",
    "top_k_checkpoints": 3,
    "data_dir": "Final/irrigation/data",
    "output_dir": "Final/irrigation",
}


def backup_all(task: str):
    src_ckpt = Path(f"Final/{task}/checkpoints")
    dst_ckpt = Path(f"Final/{task}/checkpoints_backup")
    if src_ckpt.exists():
        shutil.copytree(src_ckpt, dst_ckpt, dirs_exist_ok=True)

    for png in Path(f"Final/{task}").glob("*.png"):
        shutil.copy(png, Path(f"Final/{task}/backup_{png.name}"))

    for txt in Path(f"Final/{task}").glob("*.txt"):
        shutil.copy(txt, Path(f"Final/{task}/backup_{txt.name}"))

    for js in Path(f"Final/{task}").glob("*.json"):
        shutil.copy(js, Path(f"Final/{task}/backup_{js.name}"))

    print(f"[BACKUP] {task} -> all files backed up")



def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TimeSeriesDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int = 7):
        self.features = features
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.features) - self.seq_len + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.features[idx : idx + self.seq_len]
        y_target = self.targets[idx + self.seq_len - 1]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.var_weights = nn.Linear(input_dim, input_dim)
        self.projectors = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(input_dim)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.var_weights(x), dim=-1)
        projected = []
        for i, proj in enumerate(self.projectors):
            val = x[:, :, i:i+1]
            projected.append(proj(val) * weights[:, :, i:i+1])

        fused = torch.stack(projected, dim=-1).sum(dim=-1)
        return fused, weights


class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.vsn = VariableSelectionNetwork(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vsn_out, attn_weights = self.vsn(x)
        lstm_out, _ = self.lstm(vsn_out)
        attn_out, attn_map = self.mha(lstm_out, lstm_out, lstm_out)

        gated = torch.sigmoid(self.gate(torch.cat([lstm_out, attn_out], dim=-1))) * attn_out
        final_seq = gated[:, -1, :]
        out = self.head(final_seq).squeeze(-1)
        return out, attn_map


def train_irrigation_model() -> None:
    seed_everything(CONFIG["seed"])
    backup_all("irrigation")

    data_dir = Path(CONFIG["data_dir"])
    output_dir = Path(CONFIG["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_csv_v2 = data_dir / "cleaned_train_v2.csv"
    val_csv_v2 = data_dir / "cleaned_val_v2.csv"

    if train_csv_v2.exists():
        train_df = pd.read_csv(train_csv_v2)
        val_df = pd.read_csv(val_csv_v2)
    else:
        train_df = pd.read_csv(data_dir / "cleaned_train.csv")
        val_df = pd.read_csv(data_dir / "cleaned_val.csv")

    full_df = pd.concat([train_df, val_df], ignore_index=True)

    ignore_cols = ["timestamp", "crop_type", "irrigation_amount"]
    numeric_cols = [c for c in full_df.columns if pd.api.types.is_numeric_dtype(full_df[c])]
    feature_cols = [c for c in numeric_cols if c not in ignore_cols]

    features = full_df[feature_cols].values
    targets = full_df["irrigation_amount"].values

    tss = TimeSeriesSplit(n_splits=CONFIG["num_folds"])
    device = get_optimal_device()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    fold_metrics = []
    overall_actuals, overall_preds = [], []
    last_attn_map = None
    per_fold_rmse = []

    print(f"Starting {CONFIG['num_folds']}-Fold TimeSeriesSplit for Irrigation TFT v2...")

    tss_splits = list(tss.split(features))
    for fold, (t_idx, v_idx) in enumerate(tqdm(tss_splits, desc="Irrigation TimeSeries Folds v2", unit="fold")):
        print(f"\n--- FOLD {fold+1}/{CONFIG['num_folds']} [v2] ---")

        X_train, y_train = features[t_idx], targets[t_idx]
        X_val, y_val = features[v_idx], targets[v_idx]

        t_ds = TimeSeriesDataset(X_train, y_train, seq_len=CONFIG["seq_len"])
        v_ds = TimeSeriesDataset(X_val, y_val, seq_len=CONFIG["seq_len"])

        if len(t_ds) == 0 or len(v_ds) == 0:
            continue

        t_loader = DataLoader(t_ds, batch_size=CONFIG["batch_size"], shuffle=False)
        v_loader = DataLoader(v_ds, batch_size=CONFIG["batch_size"], shuffle=False)

        model = TemporalFusionTransformer(len(feature_cols), CONFIG["hidden_dim"], CONFIG["n_heads"]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)

        best_val_rmse = float("inf")
        best_val_mae = float("inf")
        best_actuals, best_predictions = None, None
        patience_counter = 0

        for epoch in tqdm(range(1, CONFIG["epochs"] + 1), desc=f"Fold {fold+1} Epochs", unit="epoch", leave=False):
            model.train()
            t_loss = 0.0
            for bx, by in tqdm(t_loader, desc="TFT Train Batches", leave=False, unit="batch"):
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out, _ = model(bx)
                    loss = criterion(out, by)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                t_loss += loss.item() * bx.size(0)

                if torch.cuda.is_available():
                    clear_gpu_cache()

            t_loss /= len(t_loader.dataset)

            model.eval()
            val_actuals, val_preds_list = [], []
            with torch.no_grad():
                for bx, by in tqdm(v_loader, desc="TFT Eval Batches", leave=False, unit="batch"):
                    bx, by = bx.to(device), by.to(device)
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        out, attn_map = model(bx)
                    val_actuals.extend(by.cpu().numpy())
                    val_preds_list.extend(out.cpu().numpy())
                    last_attn_map = attn_map.cpu().numpy()

            val_actuals_clean = np.nan_to_num(np.array(val_actuals), nan=0.0)
            val_preds_clean = np.nan_to_num(np.array(val_preds_list), nan=0.0)

            rmse = float(np.sqrt(mean_squared_error(val_actuals_clean, val_preds_clean)))
            mae = float(mean_absolute_error(val_actuals_clean, val_preds_clean))

            print(f"Epoch {epoch:03d}/{CONFIG['epochs']:03d} | Loss: {t_loss:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | Best: {best_val_rmse:.4f} [v2]")

            if rmse < best_val_rmse:
                best_val_rmse = rmse
                best_val_mae = mae
                best_actuals = val_actuals
                best_predictions = val_preds_list
                patience_counter = 0
                torch.save(
                    {"state_dict": model.state_dict(), "feature_cols": feature_cols, "rmse": rmse, "mae": mae, "version": "v2"},
                    ckpt_dir / f"fold_{fold+1}_v2.pth"
                )
            else:
                patience_counter += 1

            if patience_counter >= CONFIG["early_stopping"]:
                print(f"Early stopping triggered at epoch {epoch} (Patience={CONFIG['early_stopping']}).")
                break

        if best_actuals is not None:
            fold_metrics.append({"fold": fold + 1, "rmse": best_val_rmse, "mae": best_val_mae, "ckpt": str(ckpt_dir / f"fold_{fold+1}_v2.pth")})
            per_fold_rmse.append(best_val_rmse)
            overall_actuals.extend(best_actuals)
            overall_preds.extend(best_predictions)

        if torch.cuda.is_available():
            clear_gpu_cache()

    fold_metrics_sorted = sorted(fold_metrics, key=lambda x: x["rmse"])
    top_k = fold_metrics_sorted[:CONFIG["top_k_checkpoints"]]

    mean_rmse = float(np.mean([m["rmse"] for m in top_k])) if top_k else 0.0
    mean_mae = float(np.mean([m["mae"] for m in top_k])) if top_k else 0.0

    print(f"\nIrrigation Training Complete [v2]. Top {CONFIG['top_k_checkpoints']} Mean RMSE: {mean_rmse:.4f} | Mean MAE: {mean_mae:.4f}")

    with open(output_dir / "metrics_v2.txt", "w") as f:
        f.write("=== IRRIGATION METRICS V2 ===\n")
        f.write(f"Top-{CONFIG['top_k_checkpoints']} Mean RMSE: {mean_rmse:.4f}\n")
        f.write(f"Top-{CONFIG['top_k_checkpoints']} Mean MAE: {mean_mae:.4f}\n\n")

    with open(output_dir / "metrics.txt", "w") as f:
        f.write("=== IRRIGATION METRICS ===\n")
        f.write(f"Top-{CONFIG['top_k_checkpoints']} Mean RMSE: {mean_rmse:.4f}\n")
        f.write(f"Top-{CONFIG['top_k_checkpoints']} Mean MAE: {mean_mae:.4f}\n\n")

    plt.figure(figsize=(10, 5))
    plt.plot(overall_actuals[:100], label="Actual Irrigation Amount", color="blue")
    plt.plot(overall_preds[:100], label="Predicted Irrigation Amount v2", color="orange", linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Irrigation Amount")
    plt.title("Irrigation TFT: Actual vs Predicted v2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_pred_v2.png", dpi=300)
    plt.savefig(output_dir / "actual_vs_pred.png", dpi=300)
    plt.close()

    if per_fold_rmse:
        plt.figure(figsize=(6, 4))
        plt.bar(range(1, len(per_fold_rmse) + 1), per_fold_rmse, color="teal")
        plt.xlabel("Fold")
        plt.ylabel("RMSE")
        plt.title("Irrigation Fold Curves v2 (RMSE per Fold)")
        plt.tight_layout()
        plt.savefig(output_dir / "fold_curves_v2.png", dpi=300)
        plt.close()

    if last_attn_map is not None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(last_attn_map[0], cmap="viridis")
        plt.title("TFT Multi-Head Attention Weights v2")
        plt.tight_layout()
        plt.savefig(output_dir / "attention_v2.png", dpi=300)
        plt.savefig(output_dir / "attention.png", dpi=300)
        plt.close()

    metrics_v2_dict = {
        "version": "v2",
        "task": "Irrigation",
        "model": "TFT",
        "metric": "RMSE",
        "rmse": mean_rmse,
        "score": mean_rmse,
        "mae": mean_mae,
        "target_column_used": "irrigation_amount",
        "per_fold_rmse": per_fold_rmse,
        "top_checkpoints": [m["ckpt"] for m in top_k],
    }
    with open(output_dir / "metrics_v2.json", "w") as f:
        json.dump(metrics_v2_dict, f, indent=4)


if __name__ == "__main__":
    train_irrigation_model()

