"""Step 2: describe the cleaned data and make simple exploratory charts."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "output/.mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / "output/.cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from project_common import LABELS, ROOT, TARGET, TRAIN_FILE


def main():
    df = pd.read_csv(TRAIN_FILE)
    folder = ROOT / "output/eda_output"
    folder.mkdir(parents=True, exist_ok=True)

    # Show how many people gave each depression-frequency answer.
    counts = df[TARGET].value_counts().reindex(LABELS).fillna(0).astype(int)
    counts.rename(index=LABELS).rename("people").to_csv(folder / "target_counts.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([LABELS[x] for x in counts.index], counts.values)
    ax.set(ylabel="Number of respondents", title="Depression frequency in the training set")
    fig.tight_layout()
    fig.savefig(folder / "target_counts.png", dpi=160)
    plt.close(fig)

    # Show missing or invalid responses before complete-case rows were removed.
    missing = pd.read_csv(
        ROOT / "output/cleaning_output/missing_values_before_complete_case.csv",
        index_col=0,
    ).iloc[:, 0].drop(TARGET).sort_values()
    missing.rename("missing_or_invalid_rows").to_csv(folder / "missing_predictors.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(missing.index, missing.values)
    ax.set(xlabel="Rows with a missing or invalid response",
           title="Data quality before complete-case filtering")
    fig.tight_layout()
    fig.savefig(folder / "missing_predictors.png", dpi=160)
    plt.close(fig)

    # Pearson correlation ranges from -1 to +1. Positive values mean two
    # coded values tend to rise together; negative values move oppositely.
    # Most inputs are category codes, so use this only as an exploratory view.
    correlation = df.corr(method="pearson", numeric_only=True)
    correlation.to_csv(folder / "pearson_correlation_matrix.csv")

    # Save the target column separately so predictors are easy to rank.
    target_correlation = (
        correlation[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .rename("pearson_correlation_with_DEPFREQ_A")
    )
    target_correlation.to_csv(folder / "target_pearson_correlations.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)), correlation.columns,
                  rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(correlation.index)), correlation.index, fontsize=8)
    ax.set_title("Training-set correlation matrix of coded survey values")

    # Print each coefficient inside its cell. White text is easier to read on
    # dark cells, while black text is clearer on light cells.
    for row in range(len(correlation.index)):
        for column in range(len(correlation.columns)):
            value = correlation.iloc[row, column]
            text_color = "white" if abs(value) >= 0.55 else "black"
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    color=text_color, fontsize=6)

    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(folder / "pearson_correlation_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2f6f9f" if value >= 0 else "#c44e52" for value in target_correlation]
    ax.barh(target_correlation.index[::-1], target_correlation.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Pearson correlation with DEPFREQ_A",
           title="Training-set correlations with depression frequency")
    fig.tight_layout()
    fig.savefig(folder / "target_pearson_correlations.png", dpi=180)
    plt.close(fig)
    print(f"Saved EDA tables and charts to {folder}")


if __name__ == "__main__":
    main()
