# 🌾 ARIS AI — Agricultural Intelligence System

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ARIS AI** is an end-to-end multimodal artificial intelligence system built for precision agriculture. The platform integrates computer vision, semantic segmentation, tabular feature engineering, and time-series forecasting to automate crop health monitoring, nutrient optimization, weed management, and smart irrigation.

---

## 🌟 Key Architecture & Modules

The platform is structured into **4 core specialized AI sub-systems**:

| Module | Task | Architecture / Model | Evaluation Metric |
| :--- | :--- | :--- | :--- |
| **🌿 Crop Disease** | Disease Identification & Health Diagnosis | **RepViT** (with Focal Loss) | Macro F1-Score |
| **🧪 Nutrient Fusion** | Multimodal Soil & Plant Nutrient Analysis | **RepViT + XGBoost + MLP Fusion** | F1-Score (~0.94) |
| **🌱 Weed Detection** | Precision Weed Instance Segmentation | **SegFormer-B0** | mIoU |
| **💧 Irrigation** | Moisture & Crop Yield Forecasting | **Temporal Fusion Transformer (TFT)** | RMSE |

---

## 🚀 Key Features

- **⚡ Automatic Hardware Acceleration**: Automatically detects and leverages `CUDA` GPUs, Apple Silicon `MPS`, or fallback `CPU` execution with active VRAM clearing (`device_utils.py`).
- **🔄 End-to-End Pipeline Orchestration**: A single command (`python run_pipeline.py`) seamlessly runs preprocessing, model training, inference, and report compilation across all 4 modules.
- **📊 Executive Dashboard Generation**: Generates cross-module metrics, evaluation curves, and PDF/PNG executive dashboards (`generate_final_report.py`).
- **🛡️ Clean Data & Model Management**: Modular design separating raw preprocessing, model training, and inference scripts with strict checkpointing.

---

## 📁 Repository Structure

```directory
ARIS/
├── Disease/                  # Crop Disease Identification Module
│   ├── preprocess.py         # Data cleaning & class balancing
│   ├── train.py              # RepViT training with Focal Loss
│   └── inference.py          # Prediction & evaluation script
├── Nutrient/                 # Multimodal Nutrient Fusion Module
│   ├── preprocess.py         # Tabular (NPK) & image preprocessing
│   ├── train.py              # XGBoost + RepViT + MLP fusion model
│   └── inference.py          # Model inference & evaluation
├── Weed/                     # Weed Detection & Segmentation Module
│   ├── preprocess.py         # Mask preprocessing & dataset formatting
│   ├── train.py              # SegFormer-B0 semantic segmentation training
│   └── inference.py          # Mask prediction & mIoU evaluation
├── Irrigation/               # Smart Irrigation Forecasting Module
│   ├── preprocess.py         # Yield & moisture time-series preprocessing
│   ├── train.py              # Temporal Fusion Transformer (TFT) training
│   └── inference.py          # Predictive forecasting & RMSE evaluation
├── Final/                    # Generated Metrics, Plots & Executive Reports
│   ├── summary_table.txt     # Formatted CLI summary table
│   └── judge_report.pdf      # Visual executive report dashboard
├── device_utils.py           # Device detection (CUDA/MPS/CPU) & VRAM management
├── run_pipeline.py           # Master pipeline orchestrator script
├── generate_final_report.py  # Report & dashboard generation script
├── requirements.txt          # Dependencies list
└── .gitignore                # Optimized Git rules for ML repos
```

---

## ⚙️ Installation & Prerequisites

### 1. Clone the Repository
```bash
git clone https://github.com/dev-hari-haran/ARIS_AI.git
cd ARIS_AI
```

### 2. Set Up Environment & Install Dependencies
```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## 🏃 Usage & Execution

### Run Full Pipeline
To execute preprocessing, training, inference, and report generation for all modules in sequence:
```bash
python run_pipeline.py
```

### Run Specific Modules Independently
You can also run preprocessing, training, or inference for any specific module:
```bash
# Example: Crop Disease Module
python Disease/preprocess.py
python Disease/train.py
python Disease/inference.py

# Example: Generate Executive Dashboard Report
python generate_final_report.py
```

---

## 📊 Summary Output Example

```text
╔══════════════════════════════════════════════════════════════╗
║                 ARIS AI — FINAL RESULTS SUMMARY             ║
╠══════════════════╦════════════════════════╦═════════════════╣
║ Task             ║ Model                  ║ Best Score      ║
╠══════════════════╬════════════════════════╬═════════════════╣
║ Crop Disease     ║ RepViT (Focal Loss)    ║ F1: 0.9250      ║
║ Nutrient Fusion  ║ RepViT+XGB+MLP         ║ F1: 0.9410      ║
║ Weed Detection   ║ SegFormer-B0           ║ mIoU: 0.8840    ║
║ Irrigation       ║ TFT                    ║ RMSE: 0.1420    ║
╚══════════════════╩════════════════════════╩═════════════════╝
```

---

## 🤝 Contributing & Contact

Created as part of the **ARIS (Agricultural Intelligence System)** project by [Hariharan R](https://github.com/dev-hari-haran).

Feel free to open an issue or submit a pull request for feature requests or enhancements!