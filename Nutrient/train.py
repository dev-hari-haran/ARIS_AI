import os
import json
import pickle
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
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import xgboost as xgb

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

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

CONFIG = {
    "seed": 42,
    "num_folds": 3,
    "epochs": 30,
    "phase1_epochs": 5,
    "phase2_epochs": 25,
    "patience": 5,
    "batch_size": 8,
    "lr_fusion": 1e-3,
    "lr_backbone": 1e-5,
    "image_size": (224, 224),
    "checkpoint_prefix": "v2",
    "data_dir": "Final/nutrient/data",
    "output_dir": "Final/nutrient",
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


def build_repvit_backbone() -> Tuple[nn.Module, int]:
    try:
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        mobilenet.classifier = nn.Identity()
        return mobilenet, 576
    except Exception:
        from torchvision.models import mobilenet_v3_small
        mobilenet = mobilenet_v3_small(weights=None)
        mobilenet.classifier = nn.Identity()
        return mobilenet, 576


class NutrientDataset(Dataset):
    def __init__(self, df: pd.DataFrame, npk_cols: List[str], xgb_probs: np.ndarray, transform=None):
        self.df = df.reset_index(drop=True)
        self.npk_cols = npk_cols
        self.xgb_probs = xgb_probs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((CONFIG["image_size"][0], CONFIG["image_size"][1], 3), dtype=np.uint8))

        if self.transform:
            img = self.transform(img)

        npk_feat = torch.tensor(row[self.npk_cols].values.astype(np.float32), dtype=torch.float32)
        xgb_p = torch.tensor(self.xgb_probs[idx], dtype=torch.float32)
        label = int(row["class_id"])

        return img, npk_feat, xgb_p, label


class NutrientFusionModel(nn.Module):
    def __init__(self, npk_input_dim: int, num_classes: int):
        super().__init__()
        self.repvit_backbone, embed_dim = build_repvit_backbone()

        self.npk_branch = nn.Sequential(
            nn.Linear(npk_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        fusion_in = embed_dim + 64 + num_classes
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def freeze_backbone(self):
        for param in self.repvit_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.repvit_backbone.parameters():
            param.requires_grad = True

    def forward(self, image: torch.Tensor, npk_features: torch.Tensor, xgb_proba: torch.Tensor) -> torch.Tensor:
        img_emb = self.repvit_backbone(image)
        if img_emb.dim() > 2:
            img_emb = F.adaptive_avg_pool2d(img_emb, (1, 1)).flatten(1)
        npk_emb = self.npk_branch(npk_features)
        fused = torch.cat([img_emb, npk_emb, xgb_proba], dim=1)
        return self.fusion_head(fused)


def get_device() -> torch.device:
    return get_optimal_device()


def train_nutrient_model() -> None:
    seed_everything(CONFIG["seed"])
    backup_all("nutrient")

    data_dir = Path(CONFIG["data_dir"])
    output_dir = Path(CONFIG["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "processed_train.csv").exists():
        try:
            from preprocess import process_nutrient_dataset
        except ImportError:
            try:
                from Nutrient.preprocess import process_nutrient_dataset
            except ImportError:
                from nutrient.preprocess import process_nutrient_dataset
        process_nutrient_dataset()

    train_df = pd.read_csv(data_dir / "processed_train.csv")
    val_df = pd.read_csv(data_dir / "processed_val.csv")
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    classes = sorted(full_df["label"].unique().tolist())
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    full_df["class_id"] = full_df["label"].map(class_to_idx)

    npk_cols = [c for c in full_df.columns if c not in ["image_path", "label", "class_id", "stem"]]
    npk_features = full_df[npk_cols].values
    labels = full_df["class_id"].values

    from sklearn.utils.class_weight import compute_class_weight
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=labels
    )

    print("Training XGBoost tabular classifier on NPK features...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05, random_state=CONFIG["seed"], eval_metric="mlogloss"
    )
    xgb_model.fit(npk_features, labels)

    with open(ckpt_dir / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)

    xgb_probs_full = xgb_model.predict_proba(npk_features)

    transform_train = transforms.Compose([
        transforms.Resize(CONFIG["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(CONFIG["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = get_device()
    print(f"Using compute device: {device}")
    skf = StratifiedKFold(n_splits=CONFIG["num_folds"], shuffle=True, random_state=CONFIG["seed"])

    fold_scores = []
    overall_preds, overall_targets = [], []

    print(f"Starting {CONFIG['num_folds']}-Fold CV for Multimodal Nutrient Fusion v2...")

    skf_splits = list(skf.split(full_df, labels))
    for fold, (t_idx, v_idx) in enumerate(tqdm(skf_splits, desc="Nutrient 3-Fold CV v2", unit="fold")):
        print(f"\n--- FOLD {fold+1}/{CONFIG['num_folds']} [v2] ---")
        train_sub = full_df.iloc[t_idx].reset_index(drop=True)
        val_sub = full_df.iloc[v_idx].reset_index(drop=True)

        t_ds = NutrientDataset(train_sub, npk_cols, xgb_probs_full[t_idx], transform=transform_train)
        v_ds = NutrientDataset(val_sub, npk_cols, xgb_probs_full[v_idx], transform=transform_val)

        t_loader = DataLoader(t_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
        v_loader = DataLoader(v_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

        try:
            model = NutrientFusionModel(len(npk_cols), num_classes).to(device)
        except RuntimeError:
            device = torch.device("cpu")
            model = NutrientFusionModel(len(npk_cols), num_classes).to(device)

        class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32).to(device)
        criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        print("Phase 1: Frozen backbone — Training NPK Branch & Fusion Head...")
        model.freeze_backbone()
        optimizer_p1 = torch.optim.AdamW(
            list(model.npk_branch.parameters()) + list(model.fusion_head.parameters()),
            lr=CONFIG["lr_fusion"]
        )

        for epoch in tqdm(range(1, CONFIG["phase1_epochs"] + 1), desc=f"Fold {fold+1} Phase 1 Epochs", unit="epoch", leave=False):
            model.train()
            t_loss = 0.0
            for img, npk, xgb_p, target in tqdm(t_loader, desc="Phase 1 Batches", leave=False, unit="batch"):
                img, npk, xgb_p, target = img.to(device), npk.to(device), xgb_p.to(device), target.to(device)
                optimizer_p1.zero_grad()
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out = model(img, npk, xgb_p)
                    loss = criterion(out, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer_p1)
                scaler.update()
                t_loss += loss.item() * img.size(0)
                if torch.cuda.is_available():
                    clear_gpu_cache()

        print("Phase 2: Unfrozen backbone — End-to-End Fine-Tuning...")
        model.unfreeze_backbone()
        optimizer_p2 = torch.optim.AdamW([
            {"params": model.repvit_backbone.parameters(), "lr": CONFIG["lr_backbone"]},
            {"params": list(model.npk_branch.parameters()) + list(model.fusion_head.parameters()), "lr": CONFIG["lr_fusion"]}
        ], weight_decay=1e-4)

        best_val_f1 = 0.0
        best_preds, best_targets = None, None
        patience_counter = 0
        train_losses, val_losses = [], []
        val_f1s = []

        for epoch in tqdm(range(1, CONFIG["phase2_epochs"] + 1), desc=f"Fold {fold+1} Phase 2 Epochs", unit="epoch", leave=False):
            model.train()
            t_loss = 0.0
            for img, npk, xgb_p, target in tqdm(t_loader, desc="Phase 2 Train Batches", leave=False, unit="batch"):
                img, npk, xgb_p, target = img.to(device), npk.to(device), xgb_p.to(device), target.to(device)
                optimizer_p2.zero_grad()
                try:
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        out = model(img, npk, xgb_p)
                        loss = criterion(out, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer_p2)
                    scaler.update()
                    t_loss += loss.item() * img.size(0)
                except Exception as e:
                    if is_oom_error(e):
                        device, model, criterion, _ = switch_to_cpu(model, criterion, reason="GPU VRAM exceeded in Nutrient Phase 2 train batch")
                        scaler = torch.cuda.amp.GradScaler(enabled=False)
                        class_weights_tensor = class_weights_tensor.to(device)
                    else:
                        raise e

            t_loss /= len(t_loader.dataset)

            model.eval()
            v_loss = 0.0
            preds_l, targets_l = [], []
            with torch.no_grad():
                for img, npk, xgb_p, target in tqdm(v_loader, desc="Phase 2 Eval Batches", leave=False, unit="batch"):
                    img, npk, xgb_p, target = img.to(device), npk.to(device), xgb_p.to(device), target.to(device)
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        out = model(img, npk, xgb_p)
                        loss = criterion(out, target)
                    v_loss += loss.item() * img.size(0)
                    preds_l.extend(torch.argmax(out, dim=-1).cpu().numpy())
                    targets_l.extend(target.cpu().numpy())
                    if torch.cuda.is_available():
                        clear_gpu_cache()

            v_loss /= len(v_loader.dataset)
            val_f1 = f1_score(targets_l, preds_l, average="macro") if len(targets_l) > 0 else 0.0

            train_losses.append(t_loss)
            val_losses.append(v_loss)
            val_f1s.append(val_f1)

            print(f"Epoch {epoch:02d}/{CONFIG['phase2_epochs']:02d} | Loss: {t_loss:.3f} | Val Loss: {v_loss:.3f} | F1: {val_f1:.3f} | Best: {best_val_f1:.3f} [v2]")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_preds = preds_l
                best_targets = targets_l
                patience_counter = 0
                torch.save(
                    {
                        "fold": fold + 1,
                        "state_dict": model.state_dict(),
                        "classes": classes,
                        "npk_input_dim": len(npk_cols),
                        "f1_score": val_f1,
                        "version": "v2",
                    },
                    ckpt_dir / f"fold_{fold+1}_v2.pth"
                )
                torch.save(
                    {
                        "fold": fold + 1,
                        "state_dict": model.state_dict(),
                        "classes": classes,
                        "npk_input_dim": len(npk_cols),
                        "f1_score": val_f1,
                    },
                    ckpt_dir / f"fold_{fold+1}.pth"
                )
            else:
                patience_counter += 1

            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping triggered at epoch {epoch} (Patience={CONFIG['patience']}).")
                break

        fold_scores.append({"fold": fold + 1, "f1": best_val_f1, "ckpt": str(ckpt_dir / f"fold_{fold+1}_v2.pth")})
        if best_targets is not None:
            overall_targets.extend(best_targets)
            overall_preds.extend(best_preds)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
        plt.title(f"Nutrient Fold {fold+1} Loss v2")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(val_f1s) + 1), val_f1s, label="Val F1", color="purple")
        plt.title(f"Nutrient Fold {fold+1} F1 v2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"fold_{fold+1}_curves_v2.png", dpi=300)
        plt.savefig(output_dir / f"fold_{fold+1}_curves.png", dpi=300)
        plt.close()

    mean_f1 = float(np.mean([s["f1"] for s in fold_scores]))
    print(f"\nNutrient Multimodal Training Complete [v2]. Mean F1 Macro: {mean_f1:.4f}")

    if len(overall_targets) > 0:
        cm = confusion_matrix(overall_targets, overall_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes)
        plt.title("Nutrient Fusion Confusion Matrix v2")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix_v2.png", dpi=300)
        plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
        plt.close()

    metrics_v2_dict = {
        "version": "v2",
        "task": "Nutrient Fusion",
        "model": "RepViT+XGBoost+MLP",
        "metric": "F1 Macro",
        "score": mean_f1,
        "fold_scores": [s["f1"] for s in fold_scores],
        "top_checkpoints": [s["ckpt"] for s in fold_scores],
    }
    with open(output_dir / "metrics_v2.json", "w") as f:
        json.dump(metrics_v2_dict, f, indent=4)

    metrics_v1_dict = {
        "task": "Nutrient Fusion",
        "model": "RepViT+XGBoost+MLP",
        "metric": "F1 Macro",
        "score": mean_f1,
        "fold_scores": [s["f1"] for s in fold_scores],
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_v1_dict, f, indent=4)


if __name__ == "__main__":
    train_nutrient_model()

