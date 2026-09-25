import os
import sys
import time
import subprocess
import warnings
from pathlib import Path
from tqdm import tqdm

from device_utils import clear_gpu_cache, get_optimal_device
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

STEPS = [
    # Preprocessing — skip if v2 data exists
    ("Disease Preprocessing v2",    "Disease/preprocess.py",    False),
    ("Nutrient Preprocessing v2",   "Nutrient/preprocess.py",   True),
    ("Weed Preprocessing v2",       "Weed/preprocess.py",       False),
    ("Irrigation Preprocessing v2", "Irrigation/preprocess.py", False),

    # Training — v2 only
    ("Disease Training v2",         "Disease/train.py",         False),
    ("Nutrient Training v2",        "Nutrient/train.py",        True),   # skip=working model F1=0.94
    ("Weed Training v2",            "Weed/train.py",            False),
    ("Irrigation Training v2",      "Irrigation/train.py",      False),

    # Inference — v2 outputs
    ("Disease Inference v2",        "Disease/inference.py",     False),
    ("Nutrient Inference v2",       "Nutrient/inference.py",    True),   # skip=working model F1=0.94
    ("Weed Inference v2",           "Weed/inference.py",        False),
    ("Irrigation Inference v2",     "Irrigation/inference.py",  False),

    # Final report — loads best of v1/v2
    ("Final Report v2",             "generate_final_report.py", False),
]


def run_full_pipeline() -> None:
    clear_gpu_cache()
    dev = get_optimal_device()
    print("=" * 60)
    print(f" ARIS AI: STARTING V2 PIPELINE EXECUTION (Device: {dev})")
    print("=" * 60)
    
    start_total_time = time.time()

    pipeline_pbar = tqdm(STEPS, desc="Overall Pipeline Progress", unit="stage", dynamic_ncols=True)
    for title, script_path, skip_stage in pipeline_pbar:
        if skip_stage:
            print(f"[SKIP] {title} -> existing working model score preserved")
            continue

        clear_gpu_cache()
        pipeline_pbar.set_description(f"Running {title}")
        print(f"\n[RUN]  Running {title}...")
        step_start = time.time()
        
        cmd = [sys.executable, script_path]
        try:
            res = subprocess.run(cmd, check=True, text=True)
            elapsed = time.time() - step_start
            print(f"[OK] {title} completed successfully in {elapsed:.2f}s")
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] ERROR: {title} failed with exit code {e.returncode}")
            sys.exit(e.returncode)

    total_elapsed = time.time() - start_total_time
    print("\n" + "=" * 60)
    print(f" ALL V2 PIPELINE STAGES COMPLETED SUCCESSFULLY IN {total_elapsed:.2f} SECONDS")
    print("=" * 60)



if __name__ == "__main__":
    run_full_pipeline()

