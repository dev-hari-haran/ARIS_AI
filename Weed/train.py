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
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold

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
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

CONFIG = {
    "seed": 42,
    "num_folds": 3,
    "epochs": 30,
    "early_stopping": 7,
    "patience": 7,
    "batch_size": 8,
    "lr": 1e-3,
    "image_size": (512, 512),
    "fallback_image_size": (256, 256),
    "mixed_precision": True,
    "gradient_checkpointing": True,
    "checkpoint_prefix": "v2",
    "top_k_checkpoints": 3,
    "data_dir": "Final/weed/data",
    "output_dir": "Final/weed",
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


def compute_miou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2
) -> float:
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    if pred.ndim > 1 and pred.shape != target.shape:
        pred = (pred > 0.5).astype(np.uint8)
        target = (target > 0.5).astype(np.uint8)
    else:
        pred = (pred > 0.5).astype(np.uint8) if pred.dtype != np.uint8 else pred
        target = (target > 0.5).astype(np.uint8) if target.dtype != np.uint8 else target

    pred_flat = pred.flatten()
    target_flat = target.flatten()
    ious = []
    for cls in range(num_classes):
        intersection = ((pred_flat == cls) & (target_flat == cls)).sum()
        union = ((pred_flat == cls) | (target_flat == cls)).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def sanity_check_predictions(pred_mask, true_mask):
    if not isinstance(pred_mask, torch.Tensor):
        pred_mask = torch.tensor(pred_mask)
    if not isinstance(true_mask, torch.Tensor):
        true_mask = torch.tensor(true_mask)
    pred_unique = torch.unique(pred_mask).tolist()
    true_unique = torch.unique(true_mask).tolist()
    print(f"Pred unique values: {pred_unique}")
    print(f"True unique values: {true_unique}")
    if len(pred_unique) == 1:
        print("WARNING: Model predicting single class only!")


class WeedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_train: bool = True, img_size: Tuple[int, int] = (512, 512)):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        mask_path = row["mask_path"]

        try:
            img = Image.open(img_path).convert("RGB").resize(self.size)
            mask = Image.open(mask_path).convert("L").resize(self.size)
        except Exception:
            img = Image.fromarray(np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8))
            mask = Image.fromarray(np.zeros(self.size, dtype=np.uint8))

        img_np = np.array(img, dtype=np.float32) / 255.0
        mask_arr = np.array(mask, dtype=np.float32)
        mask_np = (mask_arr > 127 if mask_arr.max() > 1 else mask_arr > 0).astype(np.float32)

        if self.is_train:
            if random.random() > 0.5:
                img_np = np.fliplr(img_np).copy()
                mask_np = np.fliplr(mask_np).copy()
            if random.random() > 0.5:
                img_np = np.flipud(img_np).copy()
                mask_np = np.flipud(mask_np).copy()

        img_tensor = torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
        return img_tensor, mask_tensor


class FallbackUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2), nn.BatchNorm2d(32), nn.ReLU())
        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.dec1(x2)
        out = self.final(x3)
        return out


def build_segmentation_model() -> nn.Module:
    if HAS_SMP:
        try:
            model = smp.Unet(encoder_name="mit_b0", encoder_weights=None, in_channels=3, classes=1)
            if CONFIG.get("gradient_checkpointing", False) and hasattr(model.encoder, "set_grad_checkpointing"):
                model.encoder.set_grad_checkpointing(True)
            return model
        except Exception:
            pass
    return FallbackUNet()


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (pred_sigmoid.sum() + target.sum() + 1.0)
        return bce_loss + dice_loss


def get_device() -> torch.device:
    return get_optimal_device()


def train_weed_model() -> None:
    seed_everything(CONFIG["seed"])
    backup_all("weed")

    data_dir = Path(CONFIG["data_dir"])
    output_dir = Path(CONFIG["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    samples_dir = output_dir / "samples_v2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    train_csv_v2 = data_dir / "processed_train_v2.csv"
    val_csv_v2 = data_dir / "processed_val_v2.csv"

    if train_csv_v2.exists():
        train_df = pd.read_csv(train_csv_v2)
        val_df = pd.read_csv(val_csv_v2)
    else:
        train_df = pd.read_csv(data_dir / "processed_train.csv")
        val_df = pd.read_csv(data_dir / "processed_val.csv")

    full_df = pd.concat([train_df, val_df], ignore_index=True)

    kf = KFold(n_splits=CONFIG["num_folds"], shuffle=True, random_state=CONFIG["seed"])
    device = get_device()
    print(f"Using compute device: {device}")
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and CONFIG["mixed_precision"]))

    curr_img_size = CONFIG["image_size"]
    curr_batch_size = CONFIG["batch_size"]

    fold_metrics = []
    unique_mask_values = set()
    print(f"Starting {CONFIG['num_folds']}-Fold CV for Weed SegFormer v2...")

    kf_splits = list(kf.split(full_df))
    for fold, (t_idx, v_idx) in enumerate(tqdm(kf_splits, desc="Weed 3-Fold SegFormer CV v2", unit="fold")):
        print(f"\n--- FOLD {fold+1}/{CONFIG['num_folds']} [v2] ---")
        train_sub = full_df.iloc[t_idx].reset_index(drop=True)
        val_sub = full_df.iloc[v_idx].reset_index(drop=True)

        t_ds = WeedDataset(train_sub, is_train=True, img_size=curr_img_size)
        v_ds = WeedDataset(val_sub, is_train=False, img_size=curr_img_size)

        t_loader = DataLoader(t_ds, batch_size=curr_batch_size, shuffle=True, num_workers=0)
        v_loader = DataLoader(v_ds, batch_size=curr_batch_size, shuffle=False, num_workers=0)

        try:
            model = build_segmentation_model().to(device)
        except RuntimeError:
            device = torch.device("cpu")
            model = build_segmentation_model().to(device)

        criterion = DiceBCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

        best_val_miou = 0.0
        patience_counter = 0
        train_losses, val_mious = [], []
        sample_batch = None
        first_batch_checked = False

        for epoch in tqdm(range(1, CONFIG["epochs"] + 1), desc=f"Fold {fold+1} Epochs", unit="epoch", leave=False):
            try:
                model.train()
                t_loss = 0.0
                for imgs, masks in tqdm(t_loader, desc="Segmentation Train Batches", leave=False, unit="batch"):
                    imgs, masks = imgs.to(device), masks.to(device)
                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and CONFIG["mixed_precision"])):
                        preds = model(imgs)
                        loss = criterion(preds, masks)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    t_loss += loss.item() * imgs.size(0)

                    if torch.cuda.is_available():
                        clear_gpu_cache()

                t_loss /= len(t_loader.dataset)
                scheduler.step()

                model.eval()
                val_mious_list = []
                with torch.no_grad():
                    for imgs, masks in tqdm(v_loader, desc="Segmentation Eval Batches", leave=False, unit="batch"):
                        imgs, masks = imgs.to(device), masks.to(device)
                        with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and CONFIG["mixed_precision"])):
                            preds = model(imgs)
                            preds_sigmoid = torch.sigmoid(preds)

                        pred_bin = (preds_sigmoid > 0.5).long()
                        true_bin = (masks > 0.5).long()

                        if not first_batch_checked:
                            sanity_check_predictions(pred_bin[0], true_bin[0])
                            first_batch_checked = True

                        for p, g in zip(pred_bin, true_bin):
                            miou_val = compute_miou(p, g, num_classes=2)
                            val_mious_list.append(miou_val)

                        unique_mask_values.update(torch.unique(true_bin).cpu().tolist())

                        if sample_batch is None:
                            sample_batch = (imgs[:3].cpu(), masks[:3].cpu(), preds_sigmoid[:3].cpu())

                v_miou = float(np.mean(val_mious_list)) if val_mious_list else 0.0
                train_losses.append(t_loss)
                val_mious.append(v_miou)

                print(f"Epoch {epoch:02d}/{CONFIG['epochs']:02d} | Loss: {t_loss:.3f} | mIoU: {v_miou:.3f} | Best: {best_val_miou:.3f} [v2]")

                if v_miou > best_val_miou:
                    best_val_miou = v_miou
                    patience_counter = 0
                    torch.save(
                        {"state_dict": model.state_dict(), "miou": v_miou, "img_size": curr_img_size, "version": "v2"},
                        ckpt_dir / f"fold_{fold+1}_v2.pth"
                    )
                else:
                    patience_counter += 1

                if patience_counter >= CONFIG["early_stopping"]:
                    print(f"Early stopping triggered at epoch {epoch} (Patience={CONFIG['early_stopping']}).")
                    break

            except Exception as e:
                if is_oom_error(e):
                    print("\n[WARNING] CUDA OOM in Weed SegFormer!")
                    if device.type == "cuda" and curr_batch_size > 2:
                        print("Retrying with smaller batch size (4) and image size (256)...")
                        clear_gpu_cache()
                        curr_batch_size = max(2, curr_batch_size // 2)
                        curr_img_size = CONFIG["fallback_image_size"]
                        t_ds = WeedDataset(train_sub, is_train=True, img_size=curr_img_size)
                        v_ds = WeedDataset(val_sub, is_train=False, img_size=curr_img_size)
                        t_loader = DataLoader(t_ds, batch_size=curr_batch_size, shuffle=True, num_workers=0)
                        v_loader = DataLoader(v_ds, batch_size=curr_batch_size, shuffle=False, num_workers=0)
                        continue
                    else:
                        device, model, criterion, _ = switch_to_cpu(model, criterion, reason="SegFormer GPU VRAM exceeded")
                        scaler = torch.cuda.amp.GradScaler(enabled=False)
                        continue
                else:
                    raise e

        fold_metrics.append({"fold": fold + 1, "miou": best_val_miou, "ckpt": str(ckpt_dir / f"fold_{fold + 1}_v2.pth")})

        if torch.cuda.is_available():
            clear_gpu_cache()

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(val_mious) + 1), val_mious, label="Val mIoU v2", color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("mIoU")
        plt.title(f"Weed Fold {fold+1} mIoU Curve v2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"fold_{fold+1}_miou_v2.png", dpi=300)
        plt.savefig(output_dir / f"fold_{fold+1}_miou.png", dpi=300)
        plt.close()

        if sample_batch:
            imgs, masks, preds = sample_batch
            fig, axes = plt.subplots(3, 3, figsize=(9, 9))
            for idx in range(min(3, len(imgs))):
                img_np = imgs[idx].numpy().transpose(1, 2, 0)
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)

                axes[idx, 0].imshow(img_np)
                axes[idx, 0].set_title("Input Image")
                axes[idx, 0].axis("off")

                axes[idx, 1].imshow(masks[idx][0], cmap="gray")
                axes[idx, 1].set_title("Ground Truth Mask")
                axes[idx, 1].axis("off")

                axes[idx, 2].imshow(preds[idx][0].numpy() > 0.5, cmap="gray")
                axes[idx, 2].set_title("Predicted Mask v2")
                axes[idx, 2].axis("off")

            plt.tight_layout()
            plt.savefig(samples_dir / f"fold_{fold+1}_samples_v2.png", dpi=300)
            plt.close()

    fold_metrics_sorted = sorted(fold_metrics, key=lambda x: x["miou"], reverse=True)
    top_k = fold_metrics_sorted[:CONFIG["top_k_checkpoints"]]
    mean_miou = float(np.mean([m["miou"] for m in top_k]))

    print(f"\nWeed Segmentation Training Complete [v2]. Top {CONFIG['top_k_checkpoints']} Mean mIoU: {mean_miou:.4f}")

    metrics_v2_dict = {
        "version": "v2",
        "task": "Weed Detection",
        "model": "SegFormer",
        "metric": "mIoU",
        "mean_miou": mean_miou,
        "score": mean_miou,
        "per_fold_miou": [m["miou"] for m in fold_metrics],
        "mask_unique_values": sorted(list(unique_mask_values)),
        "top_checkpoints": [m["ckpt"] for m in top_k],
    }
    with open(output_dir / "metrics_v2.json", "w") as f:
        json.dump(metrics_v2_dict, f, indent=4)


if __name__ == "__main__":
    train_weed_model()

