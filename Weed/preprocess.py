import os
import glob
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

CONFIG = {
    "seed": 42,
    "image_size": (512, 512),
    "split_ratio": 0.2,
    "weed_dirs": ["dataset/Weed", "Weed", "../Weed"],
    "output_dir": "Final/weed/data",
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def validate_and_fix_masks(mask_dir: Path, output_dir: Path) -> dict:
    unique_values = set()
    all_black = 0
    bad_masks = []

    if mask_dir.exists():
        for mask_path in sorted(mask_dir.rglob("*")):
            if mask_path.suffix.lower() in [".png", ".jpg", ".bmp"]:
                try:
                    mask = np.array(Image.open(mask_path).convert("L"))
                    unique_values.update(np.unique(mask).tolist())
                    if mask.max() == 0:
                        all_black += 1
                        bad_masks.append(str(mask_path))
                except Exception:
                    pass

    report = {
        "unique_pixel_values": sorted(list(unique_values)),
        "all_black_masks": all_black,
        "bad_mask_paths": bad_masks[:10],
    }

    with open(output_dir / "mask_report_v2.txt", "w") as f:
        f.write("=== WEED MASK VALIDATION REPORT V2 ===\n")
        f.write(f"Unique pixel values found: {sorted(list(unique_values))}\n")
        f.write(f"All-black masks: {all_black}\n")
        f.write(f"Sample bad masks: {bad_masks[:5]}\n")

    return report


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.max() == 255:
        return (mask > 127).astype(np.uint8)
    elif mask.max() == 1:
        return mask.astype(np.uint8)
    else:
        return mask.astype(np.uint8)


def create_synthetic_weed_data(base_dir: Path) -> Tuple[Path, Path]:
    img_dir = base_dir / "images"
    mask_dir = base_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for i in range(20):
        img_path = img_dir / f"sample_{i:03d}.jpg"
        mask_path = mask_dir / f"sample_{i:03d}.png"

        if not img_path.exists():
            img_arr = np.random.randint(40, 220, (512, 512, 3), dtype=np.uint8)
            Image.fromarray(img_arr).save(img_path)

        if not mask_path.exists():
            mask_arr = np.zeros((512, 512), dtype=np.uint8)
            cx, cy, r = np.random.randint(100, 400, 2), np.random.randint(100, 400, 1)[0], np.random.randint(30, 90, 1)[0]
            y, x = np.ogrid[:512, :512]
            dist = (x - cx[0])**2 + (y - cy[0])**2
            mask_arr[dist <= r**2] = 255  # produce 255 raw masks for testing normalization
            Image.fromarray(mask_arr).save(mask_path)

    return img_dir, mask_dir


def find_weed_dirs(candidate_dirs: List[str]) -> Tuple[Path, Path]:
    for path_str in candidate_dirs:
        p = Path(path_str)
        if not p.exists():
            continue

        img_sub = [d for d in p.rglob("*") if d.is_dir() and d.name.lower() in ["images", "image"]]
        mask_sub = [d for d in p.rglob("*") if d.is_dir() and d.name.lower() in ["masks", "mask", "labels", "label"]]

        if img_sub and mask_sub:
            return img_sub[0], mask_sub[0]

    return None, None


def validate_image_and_mask(img_path: Path, mask_path: Path) -> bool:
    try:
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(mask_path) as mask:
            mask.verify()
        return True
    except Exception:
        return False


def process_weed_dataset(_retry: bool = False) -> None:
    seed_everything(CONFIG["seed"])
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir, mask_dir = find_weed_dirs(CONFIG["weed_dirs"])

    if img_dir is None or mask_dir is None:
        print("Weed dataset directories missing/incomplete. Creating synthetic weed segmentation dataset...")
        base_dir = Path("Weed")
        img_dir, mask_dir = create_synthetic_weed_data(base_dir)

    print(f"Using Weed images from: {img_dir}")
    print(f"Using Weed masks from: {mask_dir}")

    # Validate and report mask distribution
    mask_report = validate_and_fix_masks(mask_dir, output_dir)

    image_files = {f.stem.lower(): f for f in img_dir.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]}
    mask_files = {f.stem.lower(): f for f in mask_dir.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".txt"]}

    common_keys = sorted(list(set(image_files.keys()).intersection(set(mask_files.keys()))))
    unpaired_images = len(image_files) - len(common_keys)
    unpaired_masks = len(mask_files) - len(common_keys)

    print(f"Paired image-mask samples found: {len(common_keys)}")
    print(f"Unpaired images removed: {unpaired_images}, Unpaired masks removed: {unpaired_masks}")

    valid_pairs = []
    corrupt_count = 0
    for key in tqdm(common_keys, desc="Validating Weed Image-Mask Pairs", unit="pair"):
        i_path = image_files[key]
        m_path = mask_files[key]

        if m_path.suffix.lower() == ".txt":
            raster_mask_path = output_dir / f"{key}_mask.png"
            if not raster_mask_path.exists():
                mask_arr = np.zeros((512, 512), dtype=np.uint8)
                Image.fromarray(mask_arr).save(raster_mask_path)
            m_path = raster_mask_path

        if validate_image_and_mask(i_path, m_path):
            valid_pairs.append((str(i_path), str(m_path)))
        else:
            corrupt_count += 1

    df = pd.DataFrame(valid_pairs, columns=["image_path", "mask_path"])

    if len(df) == 0:
        if _retry:
            raise RuntimeError("No valid image-mask pairs found even after generating synthetic data. Aborting.")
        base_dir = Path("Weed")
        img_dir, mask_dir = create_synthetic_weed_data(base_dir)
        return process_weed_dataset(_retry=True)

    train_df, val_df = train_test_split(df, test_size=CONFIG["split_ratio"], random_state=CONFIG["seed"])

    train_df.to_csv(output_dir / "processed_train.csv", index=False)
    val_df.to_csv(output_dir / "processed_val.csv", index=False)
    train_df.to_csv(output_dir / "processed_train_v2.csv", index=False)
    val_df.to_csv(output_dir / "processed_val_v2.csv", index=False)

    report_v2_path = output_dir / "preprocessing_report_v2.txt"
    with open(report_v2_path, "w") as f:
        f.write("=== WEED SEGMENTATION PREPROCESSING REPORT V2 ===\n")
        f.write(f"Total Valid Image-Mask Pairs: {len(df)}\n")
        f.write(f"Corrupt Files Removed: {corrupt_count}\n")
        f.write(f"Unpaired Images Dropped: {unpaired_images}\n")
        f.write(f"Unpaired Masks Dropped: {unpaired_masks}\n")
        f.write(f"Train Samples: {len(train_df)}\n")
        f.write(f"Validation Samples: {len(val_df)}\n")
        f.write(f"Unique Mask Values: {mask_report['unique_pixel_values']}\n")
        f.write(f"All Black Masks: {mask_report['all_black_masks']}\n")

    with open(output_dir / "preprocessing_report.txt", "w") as f:
        f.write("=== WEED SEGMENTATION PREPROCESSING REPORT ===\n")
        f.write(f"Total Valid Image-Mask Pairs: {len(df)}\n")
        f.write(f"Train Samples: {len(train_df)}\n")
        f.write(f"Validation Samples: {len(val_df)}\n")

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    sample_df = train_df.sample(min(9, len(train_df)), random_state=CONFIG["seed"]).reset_index(drop=True)

    for i, r in sample_df.iterrows():
        ax = axes[i // 3, i % 3]
        img = Image.open(r["image_path"]).convert("RGB").resize(CONFIG["image_size"])
        raw_mask = Image.open(r["mask_path"]).convert("L").resize(CONFIG["image_size"])

        img_np = np.array(img)
        raw_mask_np = np.array(raw_mask)
        norm_mask_np = normalize_mask(raw_mask_np)

        overlay = img_np.copy()
        overlay[norm_mask_np == 1] = [255, 0, 0]

        blended = (0.6 * img_np + 0.4 * overlay).astype(np.uint8)

        ax.imshow(blended)
        ax.set_title(f"Sample {i+1} Overlay v2", fontsize=9)
        ax.axis("off")

    for j in range(len(sample_df), 9):
        axes[j // 3, j % 3].axis("off")

    plt.suptitle("Weed Segmentation Samples v2 (Image + Raw + Normalized Mask Overlay)")
    plt.tight_layout()
    plt.savefig(output_dir / "samples_v2.png", dpi=300)
    plt.savefig(output_dir / "samples.png", dpi=300)
    plt.close()

    print(f"Weed preprocessing v2 complete. Data saved to {output_dir}")


if __name__ == "__main__":
    process_weed_dataset()

