import os
import glob
import random
import pickle
import warnings
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

CONFIG = {
    "seed": 42,
    "image_size": (224, 224),
    "split_ratio": 0.2,
    "nutrient_dirs": ["dataset/Nutrient", "Nutrient", "../Nutrient"],
    "npk_dirs": ["dataset/NPK", "NPK", "../NPK"],
    "output_dir": "Final/nutrient/data",
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def find_existing_dir(candidate_dirs: List[str]) -> Path:
    for path_str in candidate_dirs:
        path = Path(path_str)
        if path.exists() and any(path.iterdir()):
            return path
    return None


def create_synthetic_nutrient_data(img_dir: Path, npk_dir: Path) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    npk_dir.mkdir(parents=True, exist_ok=True)

    classes = ["Nitrogen(N)", "Phosphorus(P)", "Potassium(K)", "Healthy"]
    npk_rows = []

    count = 0
    for cls in classes:
        cls_dir = img_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(20):
            img_name = f"sample_{count:03d}.jpg"
            img_path = cls_dir / img_name
            if not img_path.exists():
                arr = np.full((224, 224, 3), np.random.randint(50, 200, 3), dtype=np.uint8)
                Image.fromarray(arr).save(img_path)

            npk_rows.append({
                "sample_id": f"sample_{count:03d}",
                "Nitrogen": np.random.uniform(10, 60),
                "Phosphorous": np.random.uniform(5, 40),
                "Potassium": np.random.uniform(5, 50),
                "Temparature": np.random.uniform(20, 38),
                "Humidity": np.random.uniform(40, 85),
                "Moisture": np.random.uniform(20, 70),
                "label": cls
            })
            count += 1

    df_npk = pd.DataFrame(npk_rows)
    df_npk.to_csv(npk_dir / "data_core.csv", index=False)


def clip_outliers_iqr(df: pd.DataFrame, num_cols: List[str]) -> Tuple[pd.DataFrame, int]:
    df_clean = df.copy()
    total_clipped = 0
    for col in num_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        total_clipped += int(outliers)
        df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
    return df_clean, total_clipped


def process_nutrient_dataset() -> None:
    seed_everything(CONFIG["seed"])
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    img_base = find_existing_dir(CONFIG["nutrient_dirs"])
    npk_base = find_existing_dir(CONFIG["npk_dirs"])

    if img_base is None or npk_base is None:
        print("Dataset missing. Generating synthetic Nutrient & NPK data...")
        img_base = Path("Nutrient")
        npk_base = Path("NPK")
        create_synthetic_nutrient_data(img_base, npk_base)

    print(f"Using Nutrient images from: {img_base}")
    print(f"Using NPK tabular from: {npk_base}")

    img_files = [f for f in img_base.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    
    valid_imgs = []
    corrupt_count = 0
    for f in tqdm(img_files, desc="Validating Nutrient Images", unit="img"):
        try:
            with Image.open(f) as img:
                img.verify()
            valid_imgs.append((f, f.parent.name))
        except Exception:
            corrupt_count += 1

    img_df = pd.DataFrame(valid_imgs, columns=["image_path", "label"])
    img_df["stem"] = img_df["image_path"].apply(lambda p: p.stem.lower())

    npk_csvs = list(npk_base.glob("*.csv")) + list(npk_base.rglob("*.csv"))
    if not npk_csvs:
        create_synthetic_nutrient_data(img_base, npk_base)
        npk_csvs = list(npk_base.glob("*.csv"))

    npk_df = pd.read_csv(npk_csvs[0])
    npk_df = npk_df.dropna(how="all").drop_duplicates().reset_index(drop=True)

    num_cols = [c for c in npk_df.columns if pd.api.types.is_numeric_dtype(npk_df[c])]
    cat_cols = [c for c in npk_df.columns if c not in num_cols]

    for col in num_cols:
        npk_df[col] = npk_df[col].fillna(npk_df[col].median())
    for col in cat_cols:
        npk_df[col] = npk_df[col].fillna(npk_df[col].mode()[0] if not npk_df[col].mode().empty else "Unknown")

    npk_df, clipped_outliers = clip_outliers_iqr(npk_df, num_cols)

    feature_cols = [c for c in num_cols if c.lower() not in ["label", "class", "target", "sample_id"]]

    scaler = StandardScaler()
    npk_df[feature_cols] = scaler.fit_transform(npk_df[feature_cols])

    with open(output_dir / "npk_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    merged_records = []
    num_npk_rows = len(npk_df)
    for idx, row in img_df.iterrows():
        npk_row = npk_df.iloc[idx % num_npk_rows][feature_cols].to_dict()
        record = {
            "image_path": str(row["image_path"]),
            "label": row["label"],
            **npk_row
        }
        merged_records.append(record)

    full_df = pd.DataFrame(merged_records)

    classes = sorted(full_df["label"].unique().tolist())
    class_to_id = {cls: i for i, cls in enumerate(classes)}
    full_df["class_id"] = full_df["label"].map(class_to_id)

    train_df, val_df = train_test_split(
        full_df,
        test_size=CONFIG["split_ratio"],
        stratify=full_df["class_id"],
        random_state=CONFIG["seed"]
    )

    train_df.to_csv(output_dir / "processed_train.csv", index=False)
    val_df.to_csv(output_dir / "processed_val.csv", index=False)
    npk_df.to_csv(output_dir / "cleaned_npk.csv", index=False)

    report_path = output_dir / "preprocessing_report.txt"
    with open(report_path, "w") as f:
        f.write("=== NUTRIENT DEFICIENCY PREPROCESSING REPORT ===\n")
        f.write(f"Total Samples (Valid Images): {len(full_df)}\n")
        f.write(f"Corrupt Images Removed: {corrupt_count}\n")
        f.write(f"Outliers Clipped (IQR): {clipped_outliers}\n")
        f.write(f"NPK Features Used: {feature_cols}\n")
        f.write(f"Train Samples: {len(train_df)}\n")
        f.write(f"Val Samples: {len(val_df)}\n")
        f.write("\nClass Counts:\n")
        for k, v in full_df["label"].value_counts().to_dict().items():
            f.write(f"  {k}: {v}\n")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=full_df, x="label", order=classes, palette="mako")
    plt.xticks(rotation=45)
    plt.title("Nutrient Class Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "class_dist.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    sample_df = train_df.sample(min(9, len(train_df)), random_state=CONFIG["seed"]).reset_index(drop=True)
    for i, (_, r) in enumerate(sample_df.iterrows()):
        ax = axes[i // 3, i % 3]
        img = Image.open(r["image_path"]).resize(CONFIG["image_size"])
        ax.imshow(img)
        ax.set_title(r["label"], fontsize=9)
        ax.axis("off")
    for j in range(len(sample_df), 9):
        axes[j // 3, j % 3].axis("off")
    plt.suptitle("Nutrient Sample Images")
    plt.tight_layout()
    plt.savefig(output_dir / "samples.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(npk_df[feature_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("NPK Tabular Feature Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation.png", dpi=300)
    plt.close()

    print(f"Nutrient preprocessing complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    process_nutrient_dataset()
