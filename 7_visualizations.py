"""Step 7: create presentation charts from CV and final test results."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "output/.mpl_cache"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_common import LABELS, ROOT


def main():
    cv_folder = ROOT / "output/cv_evaluation_output"
    test_folder = ROOT / "output/test_set_output"
    folder = ROOT / "output/visualizations_output"
    folder.mkdir(parents=True, exist_ok=True)
    selected = (cv_folder / "selected_model.txt").read_text().strip()

    # Chart 1: training-only CV selects the model before the test set is opened.
    comparison = pd.read_csv(cv_folder / "model_comparison.csv").sort_values("macro_f1_mean")
    colors = ["#1976a3" if name == selected else "#9ba6ac" for name in comparison["model"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(comparison["model"], comparison["macro_f1_mean"], color=colors)
    ax.set(xlabel="Mean Macro F1 across 5 training folds",
           title="Training cross-validation model comparison", xlim=(0, 0.6))
    ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(folder / "model_comparison.png", dpi=180)
    plt.close(fig)

    # Chart 2: final test metrics for the one model selected by CV.
    metrics = pd.read_csv(test_folder / "final_metrics.csv").iloc[0]
    metric_names = ["macro_f1", "balanced_accuracy", "accuracy",
                    "macro_precision", "macro_recall"]
    labels = ["Macro F1", "Balanced accuracy", "Accuracy",
              "Macro precision", "Macro recall"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, metrics[metric_names].astype(float), color="#1976a3")
    ax.set(ylabel="Score on untouched test set",
           title=f"{selected}: final test results", ylim=(0, 0.75))
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(folder / "final_test_metrics.png", dpi=180)
    plt.close(fig)

    # Chart 3: precision, recall, and F1 for every target class.
    classes = pd.read_csv(test_folder / "per_class_metrics.csv")
    positions = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, metric in enumerate(["precision", "recall", "f1"]):
        ax.bar(positions + (index - 1) * 0.25, classes[metric],
               width=0.25, label=metric.title())
    ax.set_xticks(positions, classes["class"], rotation=10)
    ax.set(ylabel="Score on untouched test set",
           title=f"{selected}: performance for each class", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "per_class_metrics.png", dpi=180)
    plt.close(fig)

    # Chart 4: rows are actual classes and columns are predicted classes.
    matrix = pd.read_csv(test_folder / "confusion_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            ax.text(column, row, f"{value:.1%}", ha="center", va="center",
                    color="white" if value > 0.5 else "black")
    ax.set_xticks(range(len(LABELS)), LABELS.values(), rotation=25, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS.values())
    ax.set(xlabel="Predicted class", ylabel="Actual class",
           title=f"{selected}: normalized test confusion matrix")
    fig.colorbar(image, ax=ax, label="Share of actual class")
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    # Chart 5: model importance. CatBoost has direct predictor importance;
    # encoded models use their expanded input columns when available.
    model = joblib.load(test_folder / "selected_model.joblib")
    estimator = model.named_steps["model"]
    fitted = getattr(estimator, "model_", estimator)
    values = getattr(fitted, "feature_importances_", None)
    if values is not None:
        if selected == "CatBoostClassifier":
            names = model.feature_names_in_
        else:
            names = model.named_steps["prepare"].get_feature_names_out()
            names = [name.split("__", 1)[-1] for name in names]
        importance = pd.Series(values, index=names).sort_values().tail(15)
        importance.rename("importance").to_csv(folder / "feature_importance.csv")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(importance.index, importance.values, color="#1976a3")
        ax.set(xlabel="Model feature importance",
               title=f"{selected}: most useful predictors")
        fig.tight_layout()
        fig.savefig(folder / "feature_importance.png", dpi=180)
        plt.close(fig)

    # Chart 6: compare every model's mean F1 for all five target classes.
    per_class = pd.read_csv(cv_folder / "per_class_comparison.csv")
    class_order = list(LABELS.values())
    model_order = comparison.sort_values("macro_f1_mean", ascending=False)["model"].tolist()
    positions = np.arange(len(class_order))
    bar_width = 0.16
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, name in enumerate(model_order):
        model_scores = (per_class[per_class["model"] == name]
                        .set_index("class").reindex(class_order)["f1"])
        ax.bar(positions + (index - 2) * bar_width, model_scores,
               width=bar_width, label=name)
    ax.set_xticks(positions, class_order)
    ax.set(ylabel="Mean F1 across 5 outer folds",
           title="Per-class F1 comparison across all models", ylim=(0, 0.9))
    ax.legend(fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "five_model_per_class_f1.png", dpi=180)
    plt.close(fig)

    print(f"Saved final charts to {folder}")


if __name__ == "__main__":
    main()
