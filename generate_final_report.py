import os
import json
import sys
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import matplotlib.pyplot as plt



def load_best_metrics(task: str) -> dict:
    v2_path = f"Final/{task}/metrics_v2.json"
    v1_path = f"Final/{task}/metrics.json"
    try:
        with open(v2_path) as f:
            data = json.load(f)
            data["source"] = "v2"
            return data
    except Exception:
        pass
    try:
        with open(v1_path) as f:
            data = json.load(f)
            data["source"] = "v1"
            return data
    except Exception:
        return {"score": "N/A", "source": "missing"}


def load_best_graph(paths: list):
    for p in paths:
        try:
            img = plt.imread(p)
            return img, p
        except Exception:
            continue
    return None, None


def generate_judge_report() -> None:
    final_dir = Path("Final")
    final_dir.mkdir(parents=True, exist_ok=True)

    disease_m = load_best_metrics("disease")
    nutrient_m = load_best_metrics("nutrient")
    weed_m = load_best_metrics("weed")
    irrigation_m = load_best_metrics("irrigation")

    disease_graphs = [
        "Final/disease/fold_1_curves_v2.png",
        "Final/disease/fold_1_curves.png",
    ]
    nutrient_graphs = [
        "Final/nutrient/fold_1_curves_v2.png",
        "Final/nutrient/fold_1_curves.png",
    ]
    weed_graphs = [
        "Final/weed/fold_1_miou_v2.png",
        "Final/weed/fold_1_miou.png",
    ]
    irrigation_graphs = [
        "Final/irrigation/actual_vs_pred_v2.png",
        "Final/irrigation/actual_vs_pred.png",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("ARIS AI — COMPETITION EXECUTIVE DASHBOARD", fontsize=18, fontweight="bold", y=0.98)

    tasks_info = [
        ("Crop Disease (RepViT + Focal Loss)", disease_graphs, disease_m, axes[0, 0]),
        ("Nutrient Fusion (RepViT + XGBoost + MLP)", nutrient_graphs, nutrient_m, axes[0, 1]),
        ("Weed Segmentation (SegFormer-B0)", weed_graphs, weed_m, axes[1, 0]),
        ("Irrigation Forecast (TFT)", irrigation_graphs, irrigation_m, axes[1, 1]),
    ]

    for title, graphs, meta, ax in tasks_info:
        img, path_used = load_best_graph(graphs)
        src_tag = f"[{meta.get('source', 'N/A')}]"
        if img is not None:
            ax.imshow(img)
            ax.set_title(f"{title} {src_tag}", fontsize=12, fontweight="bold")
            ax.axis("off")
        else:
            ax.text(0.5, 0.5, f"{title}\n[Graph Pending] {src_tag}", ha="center", va="center", fontsize=14, color="gray")
            ax.set_title(f"{title} {src_tag}", fontsize=12, fontweight="bold")
            ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_png_v2 = Path("Final/judge_report_v2.png")
    output_pdf_v2 = Path("Final/judge_report_v2.pdf")
    output_txt_v2 = Path("Final/summary_table_v2.txt")

    output_png_v1 = Path("Final/judge_report.png")
    output_pdf_v1 = Path("Final/judge_report.pdf")
    output_txt_v1 = Path("Final/summary_table.txt")

    plt.savefig(output_png_v2, dpi=300)
    plt.savefig(output_pdf_v2)

    plt.savefig(output_png_v1, dpi=300)
    plt.savefig(output_pdf_v1)
    plt.close()

    def format_score(score_val):
        if isinstance(score_val, (int, float)):
            return f"{score_val:.4f}"
        return str(score_val)

    d_score = format_score(disease_m.get("score", "N/A"))
    n_score = format_score(nutrient_m.get("score", "N/A"))
    w_score = format_score(weed_m.get("score", "N/A"))
    i_score = format_score(irrigation_m.get("score", "N/A"))

    summary_table_str = f"""
╔══════════════════════════════════════════════════════════════╗
║                 ARIS AI — FINAL RESULTS SUMMARY             ║
╠══════════════════╦════════════════════════╦═════════════════╣
║ Task             ║ Model                  ║ Best Score      ║
╠══════════════════╬════════════════════════╬═════════════════╣
║ Crop Disease     ║ RepViT (Focal Loss)    ║ F1: {d_score:<11} ║
║ Nutrient Fusion  ║ RepViT+XGB+MLP         ║ F1: {n_score:<11} ║
║ Weed Detection   ║ SegFormer-B0           ║ mIoU: {w_score:<9} ║
║ Irrigation       ║ TFT                    ║ RMSE: {i_score:<9} ║
╚══════════════════╩════════════════════════╩═════════════════╝
        Version: v2 (v1 fallback where v2 unavailable)
"""

    print(summary_table_str)

    with open(output_txt_v2, "w", encoding="utf-8") as f:
        f.write(summary_table_str)

    with open(output_txt_v1, "w", encoding="utf-8") as f:
        f.write(summary_table_str)

    print(f"Executive Judge Report Dashboard successfully saved to:")
    print(f"  - PNG: {output_png_v2.resolve()}")
    print(f"  - PDF: {output_pdf_v2.resolve()}")
    print(f"  - TXT: {output_txt_v2.resolve()}")


if __name__ == "__main__":
    generate_judge_report()

