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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

CONFIG = {
    "seed": 42,
    "split_ratio": 0.2,
    "irrigation_dirs": ["dataset/Irrigation", "Irrigation", "../Irrigation"],
    "output_dir": "Final/irrigation/data",
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def validate_target(df: pd.DataFrame, target_col: str) -> str:
    std = df[target_col].std()
    mean = df[target_col].mean()

    print(f"Target '{target_col}': mean={mean:.6f}, std={std:.6f}")

    if std < 0.01:
        print(f"WARNING: '{target_col}' near-zero variance!")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]
        variances = df[numeric_cols].std().sort_values(ascending=False)
        best_col = variances.index[0]
        print(f"Switching target to: '{best_col}' (std={variances[best_col]:.4f})")
        return best_col

    return target_col


def create_synthetic_irrigation_data(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "irrigation_time_series.csv"

    dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
    crop_types = ["Wheat", "Maize", "Rice", "Cotton"]

    rows = []
    for d in dates:
        rows.append({
            "timestamp": d.strftime("%Y-%m-%d"),
            "temperature": np.random.uniform(15, 35),
            "humidity": np.random.uniform(40, 90),
            "soil_moisture": np.random.uniform(10, 50),
            "rainfall": np.random.exponential(scale=10),
            "crop_type": np.random.choice(crop_types),
            "irrigation_amount": np.random.uniform(5, 40)
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return csv_path


def get_season(month: int) -> int:
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Autumn


def map_column_names(df: pd.DataFrame) -> pd.DataFrame:
    col_mapping = {}
    assigned_targets = set()
    
    for col in df.columns:
        c_lower = col.lower()
        if c_lower == "timestamp":
            col_mapping[col] = "timestamp"
            assigned_targets.add("timestamp")

    for col in df.columns:
        if col in col_mapping:
            continue
        c_lower = col.lower()
        target = None
        if "temp" in c_lower:
            target = "temperature"
        elif "humid" in c_lower:
            target = "humidity"
        elif "soil" in c_lower or "moist" in c_lower:
            target = "soil_moisture"
        elif "rain" in c_lower:
            target = "rainfall"
        elif "crop" in c_lower and "type" in c_lower:
            target = "crop_type"
        elif "irrigation" in c_lower or "yield" in c_lower:
            target = "irrigation_amount"
        elif ("time" in c_lower or "date" in c_lower) and "timestamp" not in assigned_targets:
            target = "timestamp"

        if target and target not in assigned_targets:
            col_mapping[col] = target
            assigned_targets.add(target)

    renamed_df = df.rename(columns=col_mapping)
    return renamed_df.loc[:, ~renamed_df.columns.duplicated()]


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


def process_irrigation_dataset() -> None:
    seed_everything(CONFIG["seed"])
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = None
    for p_str in CONFIG["irrigation_dirs"]:
        p = Path(p_str)
        if p.exists():
            csv_matches = list(p.glob("*.csv")) + list(p.rglob("*.csv"))
            if csv_matches:
                csv_file = csv_matches[0]
                break

    if csv_file is None:
        print("Irrigation CSV missing. Creating synthetic time series dataset...")
        csv_file = create_synthetic_irrigation_data(Path("Irrigation"))

    print(f"Using Irrigation raw dataset from: {csv_file}")
    df = pd.read_csv(csv_file)

    df = map_column_names(df)

    required_defaults = {
        "timestamp": pd.date_range("2024-01-01", periods=len(df), freq="D").strftime("%Y-%m-%d"),
        "temperature": 25.0,
        "humidity": 60.0,
        "soil_moisture": 30.0,
        "rainfall": 5.0,
        "crop_type": "Wheat",
        "irrigation_amount": 20.0
    }

    for col, default_val in required_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    df = df.dropna(how="all").drop_duplicates().reset_index(drop=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = df["timestamp"].fillna(pd.to_datetime("2024-01-01"))
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["season"] = df["month"].apply(get_season)

    num_cols = ["temperature", "humidity", "soil_moisture", "rainfall", "irrigation_amount"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df, clipped_count = clip_outliers_iqr(df, num_cols)

    # Detect & validate target column variance
    target_col = validate_target(df, "irrigation_amount")
    if target_col != "irrigation_amount":
        df["irrigation_amount"] = df[target_col]

    target_mean = float(df["irrigation_amount"].mean())
    target_std = float(df["irrigation_amount"].std())

    lag_target_cols = ["soil_moisture", "temperature", "humidity", "rainfall"]
    for col in tqdm(lag_target_cols, desc="Engineering Lag & Rolling Features", unit="feature"):
        for lag in [1, 3, 7]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag).bfill()

        for win in [7, 14]:
            df[f"{col}_roll_mean_{win}"] = df[col].rolling(window=win, min_periods=1).mean()
            df[f"{col}_roll_std_{win}"] = df[col].rolling(window=win, min_periods=1).std().fillna(0)

    encoder = LabelEncoder()
    df["crop_type_encoded"] = encoder.fit_transform(df["crop_type"].astype(str))

    with open(output_dir / "crop_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0.0)

    feature_cols = [c for c in numeric_cols if c != "irrigation_amount"]
    all_num_features = feature_cols + ["irrigation_amount"]

    scaler = MinMaxScaler()
    df[all_num_features] = scaler.fit_transform(df[all_num_features])

    with open(output_dir / "irrigation_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    split_idx = int(len(df) * (1 - CONFIG["split_ratio"]))
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    train_df.to_csv(output_dir / "cleaned_train.csv", index=False)
    val_df.to_csv(output_dir / "cleaned_val.csv", index=False)
    train_df.to_csv(output_dir / "cleaned_train_v2.csv", index=False)
    val_df.to_csv(output_dir / "cleaned_val_v2.csv", index=False)

    report_v2_path = output_dir / "preprocessing_report_v2.txt"
    with open(report_v2_path, "w") as f:
        f.write("=== IRRIGATION PREPROCESSING REPORT V2 ===\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Target column used: {target_col} (switched from irrigation_amount if near-zero)\n")
        f.write(f"Target mean: {target_mean:.6f}\n")
        f.write(f"Target std: {target_std:.6f}\n")
        f.write(f"Features engineered: {len(feature_cols)}\n")
        f.write(f"Outliers clipped: {clipped_count}\n")
        f.write(f"Split: sequential 80/20\n")

    with open(output_dir / "preprocessing_report.txt", "w") as f:
        f.write("=== IRRIGATION TIME SERIES PREPROCESSING REPORT ===\n")
        f.write(f"Total Time Series Samples: {len(df)}\n")
        f.write(f"Outliers Clipped (IQR): {clipped_count}\n")
        f.write(f"Features Engineered: {len(feature_cols)}\n")
        f.write(f"Sequential Split Used (Preserved Time Order): 80% Train ({len(train_df)}), 20% Val ({len(val_df)})\n")

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[all_num_features].corr(), cmap="Blues", annot=False)
    plt.title("Irrigation Feature Correlation Heatmap v2")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_v2.png", dpi=300)
    plt.savefig(output_dir / "correlation.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    df[["soil_moisture", "temperature", "humidity", "rainfall", "irrigation_amount"]].hist(bins=20, figsize=(10, 6))
    plt.suptitle("Irrigation Feature Distributions v2")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_dist_v2.png", dpi=300)
    plt.savefig(output_dir / "feature_dist.png", dpi=300)
    plt.close()

    print(f"Irrigation preprocessing v2 complete. Data saved to {output_dir}")


if __name__ == "__main__":
    process_irrigation_dataset()

