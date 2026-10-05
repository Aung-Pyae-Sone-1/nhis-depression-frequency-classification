"""Run the complete project in the correct order.

The nested cross-validation step can take several minutes.
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEPS = [
    "1_cleaning.py",
    "2_split.py",
    "3_eda.py",
    "4_tuning.py",
    "5_cv_evaluation.py",
    "6_test_set.py",
    "7_visualizations.py",
]


def main():
    for script in STEPS:
        print(f"\nRunning {script}...", flush=True)
        # check=True stops immediately if a step fails.
        subprocess.run([sys.executable, str(ROOT / script)], check=True, cwd=ROOT)
    print("\nPipeline complete. Results are in the output folder.")


if __name__ == "__main__":
    main()
