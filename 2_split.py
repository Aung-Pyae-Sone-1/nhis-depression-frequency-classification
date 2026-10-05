"""Step 3: reserve 20% of the cleaned rows as an untouched test set."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "output/.mpl_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from project_common import DATA_FILE, ROOT, SEED, TARGET


def main():
    data = pd.read_csv(DATA_FILE)

    # Stratify keeps similar target-class percentages in both datasets.
    train, test = train_test_split(
        data, test_size=0.20, random_state=SEED, stratify=data[TARGET]
    )
    folder = ROOT / "output/split_output"
    folder.mkdir(parents=True, exist_ok=True)
    train.to_csv(folder / "train_data.csv", index=False)
    test.to_csv(folder / "test_data.csv", index=False)

    # This table makes the split easy to explain in the presentation.
    rows = []
    for name, frame in [("Training", train), ("Test", test)]:
        counts = frame[TARGET].value_counts(normalize=True).sort_index()
        for target, percentage in counts.items():
            rows.append({"dataset": name, "target": target,
                         "percentage": percentage * 100})
    distribution = pd.DataFrame(rows)
    distribution.to_csv(folder / "class_distribution.csv", index=False)

    # Similar bars confirm that stratification preserved class proportions.
    table = distribution.pivot(index="target", columns="dataset", values="percentage")
    positions = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(positions - 0.2, table["Training"], width=0.4, label="Training (80%)")
    ax.bar(positions + 0.2, table["Test"], width=0.4, label="Test (20%)")
    ax.set_xticks(positions, ["Daily", "Weekly", "Monthly",
                             "Few times/year", "Never"], rotation=10)
    ax.set(ylabel="Percentage of dataset",
           title="Stratified train-test class distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "train_test_class_distribution.png", dpi=160)
    plt.close(fig)

    print(f"Training rows: {len(train):,} (80%)")
    print(f"Untouched test rows: {len(test):,} (20%)")


if __name__ == "__main__":
    main()
