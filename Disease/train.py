import os
import gc
import json
import math
import random
import shutil
import sys
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Any
from tqdm import tqdm

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
            try:
                clear_gpu_cache()
                return torch.device("cuda")
            except Exception:
                pass
        return torch.device("cpu")

    def is_oom_error(e):
        return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()

    def switch_to_cpu(model=None, criterion=None, tensors=None, reason="VRAM exceeded"):
        clear_gpu_cache()
        dev = torch.device("cpu")
        if model is not None:
            model = model.to(dev)
        if criterion is not None:
            criterion = criterion.to(dev)
        return dev, model, criterion, []

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
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

CONFIG = {
    "image_size": 192,
    "batch_size": 16,
    "fallback_batch_size": 8,
    "num_folds": 3,
    "epochs": 30,
    "early_stopping": 10,
    "mixed_precision": True,
    "num_workers": 0,
    "pin_memory": True,
    "lr": 3e-4,
    "lr_head": 1e-3,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "focal_gamma": 2,
    "seed": 42,
    "checkpoint_prefix": "v2",
    "model_name": "resnet18",
    "top_k_checkpoints": 3,
    "max_train_samples": 3000,
    "data_dir": "Final/disease/data",
    "output_dir": "Final/disease",
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


def safe_forward(model: nn.Module, images: torch.Tensor, retry: bool = True) -> torch.Tensor:
    try:
        return model(images)
    except Exception as e:
        if is_oom_error(e) and retry:
            clear_gpu_cache()
            return model(images)
        raise e


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.weight is not None and self.weight.device != inputs.device:
            self.weight = self.weight.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiseaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: transforms.Compose = None, img_size: Tuple[int, int] = (192, 192)):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = row["filepath"]
        label = int(row["class_id"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.fromarray(np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8))

        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(img_size: Tuple[int, int]) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


class RepViTClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        if HAS_TIMM:
            try:
                self.backbone = timm.create_model(CONFIG["model_name"], pretrained=True, num_classes=0)
            except Exception:
                self.backbone = timm.create_model(CONFIG["model_name"], pretrained=False, num_classes=0)
            embed_dim = getattr(self.backbone, "num_features", 512)
        else:
            from torchvision.models import mobilenet_v3_small
            mobilenet = mobilenet_v3_small(weights=None)
            self.backbone = mobilenet.features
            embed_dim = 576

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(0.4),
            nn.Linear(embed_dim, num_classes)
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        logits = self.classifier(features)
        return logits


def train_epoch_amp(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = safe_forward(model, images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def eval_epoch_amp(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = safe_forward(model, images)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    f1 = f1_score(all_targets, all_preds, average="macro") if len(all_targets) > 0 else 0.0
    return avg_loss, f1, np.array(all_targets), np.array(all_preds)


def train_disease_model() -> None:
    seed_everything(CONFIG["seed"])
    backup_all("disease")

    data_dir = Path(CONFIG["data_dir"])
    output_dir = Path(CONFIG["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Prefer processed_train_v2.csv, fallback to processed_train.csv
    train_csv_v2 = data_dir / "processed_train_v2.csv"
    val_csv_v2 = data_dir / "processed_val_v2.csv"

    if train_csv_v2.exists():
        train_df = pd.read_csv(train_csv_v2)
        val_df = pd.read_csv(val_csv_v2) if val_csv_v2.exists() else pd.DataFrame()
    else:
        train_df = pd.read_csv(data_dir / "processed_train.csv")
        val_df = pd.read_csv(data_dir / "processed_val.csv") if (data_dir / "processed_val.csv").exists() else pd.DataFrame()

    full_df = pd.concat([train_df, val_df], ignore_index=True)

    classes = sorted(full_df["label"].unique().tolist())
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    full_df["class_id"] = full_df["label"].map(class_to_idx)

    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=full_df["class_id"].values
    )
    device = get_optimal_device()
    class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32).to(device)

    curr_img_size = (CONFIG["image_size"], CONFIG["image_size"])
    curr_batch_size = CONFIG["batch_size"]

    train_transform, val_transform = get_transforms(curr_img_size)
    skf = StratifiedKFold(n_splits=CONFIG["num_folds"], shuffle=True, random_state=CONFIG["seed"])

    fold_metrics = []
    overall_targets = []
    overall_preds = []

    print(f"Starting {CONFIG['num_folds']}-Fold CV with Focal Loss [v2]...")

    fold_splits = list(skf.split(full_df, full_df["class_id"]))
    for fold, (train_idx, val_idx) in enumerate(tqdm(fold_splits, desc="Disease 3-Fold CV v2", unit="fold")):
        print(f"\n--- FOLD {fold + 1}/{CONFIG['num_folds']} [v2] ---")

        fold_train_df = full_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = full_df.iloc[val_idx].reset_index(drop=True)

        num_samples = min(CONFIG.get("max_train_samples", 3000), len(fold_train_df))
        sample_weights = torch.tensor(
            [class_weights_arr[cid] for cid in fold_train_df["class_id"].values],
            dtype=torch.float32
        )
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

        train_dataset = DiseaseDataset(fold_train_df, transform=train_transform, img_size=curr_img_size)
        val_dataset = DiseaseDataset(fold_val_df, transform=val_transform, img_size=curr_img_size)

        train_loader = DataLoader(train_dataset, batch_size=curr_batch_size, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=curr_batch_size, shuffle=False, num_workers=0)

        try:
            model = RepViTClassifier(num_classes).to(device)
        except RuntimeError:
            device = torch.device("cpu")
            model = RepViTClassifier(num_classes).to(device)

        criterion = FocalLoss(gamma=CONFIG["focal_gamma"], weight=class_weights_tensor.to(device))
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        total_epochs = CONFIG["epochs"]
        best_val_f1 = 0.0
        best_preds, best_targets = None, None
        patience_counter = 0

        train_losses, val_losses = [], []
        train_f1s, val_f1s = [], []

        # Phase 1: Frozen backbone
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=CONFIG["lr_head"], weight_decay=CONFIG["weight_decay"])

        for epoch in range(1, 2):
            try:
                train_loss = train_epoch_amp(model, train_loader, criterion, optimizer, scaler, device)
                val_loss, val_f1, targets, preds = eval_epoch_amp(model, val_loader, criterion, device)
            except Exception as e:
                if is_oom_error(e):
                    device, model, criterion, _ = switch_to_cpu(model, criterion, reason="GPU VRAM exceeded in Phase 1")
                    scaler = torch.cuda.amp.GradScaler(enabled=False)
                    train_loss = train_epoch_amp(model, train_loader, criterion, optimizer, scaler, device)
                    val_loss, val_f1, targets, preds = eval_epoch_amp(model, val_loader, criterion, device)
                else:
                    raise e

        # Phase 2: Unfrozen end-to-end fine-tuning
        model.unfreeze_backbone()
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": CONFIG["lr_backbone"]},
            {"params": model.classifier.parameters(), "lr": CONFIG["lr"]}
        ], weight_decay=CONFIG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - 1, eta_min=1e-6)

        for epoch in tqdm(range(2, total_epochs + 1), desc=f"Fold {fold + 1} Epochs", unit="epoch", leave=False):
            try:
                train_loss = train_epoch_amp(model, train_loader, criterion, optimizer, scaler, device)
                val_loss, val_f1, targets, preds = eval_epoch_amp(model, val_loader, criterion, device)
            except Exception as e:
                if is_oom_error(e):
                    device, model, criterion, _ = switch_to_cpu(model, criterion, reason="GPU VRAM exceeded in Phase 2")
                    scaler = torch.cuda.amp.GradScaler(enabled=False)
                    train_loss = train_epoch_amp(model, train_loader, criterion, optimizer, scaler, device)
                    val_loss, val_f1, targets, preds = eval_epoch_amp(model, val_loader, criterion, device)
                else:
                    raise e

            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1s.append(val_f1)

            print(f"Epoch {epoch:02d}/{total_epochs:02d} | Loss: {train_loss:.3f} | F1: {val_f1:.3f} | Best: {best_val_f1:.3f} [v2]")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_preds = preds
                best_targets = targets
                patience_counter = 0
                ckpt_v2_name = f"fold_{fold + 1}_v2.pth"
                torch.save(
                    {
                        "fold": fold + 1,
                        "state_dict": model.state_dict(),
                        "classes": classes,
                        "f1_score": val_f1,
                        "img_size": curr_img_size,
                        "version": "v2",
                    },
                    ckpt_dir / ckpt_v2_name
                )
            else:
                patience_counter += 1

            if patience_counter >= CONFIG["early_stopping"]:
                print(f"Early stopping triggered at epoch {epoch} (Patience={CONFIG['early_stopping']}).")
                break

        fold_metrics.append({"fold": fold + 1, "f1": best_val_f1, "ckpt": str(ckpt_dir / f"fold_{fold + 1}_v2.pth")})
        if best_targets is not None and best_preds is not None:
            overall_targets.extend(best_targets)
            overall_preds.extend(best_preds)

        clear_gpu_cache()

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
        plt.title(f"Fold {fold + 1} Loss Curve v2")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(val_f1s) + 1), val_f1s, label="Val F1", color="green")
        plt.title(f"Fold {fold + 1} F1 Curve v2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"fold_{fold + 1}_curves_v2.png", dpi=300)
        plt.savefig(output_dir / f"fold_{fold + 1}_curves.png", dpi=300)
        plt.close()

    fold_metrics_sorted = sorted(fold_metrics, key=lambda x: x["f1"], reverse=True)
    top_k = fold_metrics_sorted[:CONFIG["top_k_checkpoints"]]
    mean_f1 = float(np.mean([m["f1"] for m in top_k]))

    print(f"\nTraining Complete [v2]. Top-{CONFIG['top_k_checkpoints']} Fold Mean F1 Macro: {mean_f1:.4f}")

    if len(overall_targets) > 0 and len(overall_preds) > 0:
        cm = confusion_matrix(overall_targets, overall_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title("Crop Disease Confusion Matrix v2")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix_v2.png", dpi=300)
        plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
        plt.close()

        report_str = classification_report(overall_targets, overall_preds, target_names=classes)
        with open(output_dir / "classification_report_v2.txt", "w") as f:
            f.write("=== CROP DISEASE CLASSIFICATION REPORT V2 ===\n\n")
            f.write(report_str)

    metrics_v2_dict = {
        "version": "v2",
        "task": "Crop Disease",
        "model": "RepViT",
        "metric": "F1 Macro",
        "f1_macro": mean_f1,
        "score": mean_f1,
        "classes_skipped": [],
        "classes_remaining": num_classes,
        "fold_scores": [m["f1"] for m in fold_metrics],
        "top_checkpoints": [m["ckpt"] for m in top_k],
    }
    with open(output_dir / "metrics_v2.json", "w") as f:
        json.dump(metrics_v2_dict, f, indent=4)


if __name__ == "__main__":
    train_disease_model()

