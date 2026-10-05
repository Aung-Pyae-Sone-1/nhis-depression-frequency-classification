"""Step 5: compare models with nested CV using only the training set."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "output/.mpl_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from project_common import (LABELS, MODEL_NAMES, ROOT, SEED, load_train_test,
                            make_pipeline, parameter_grid)


METRICS = ["macro_f1", "balanced_accuracy", "accuracy",
           "macro_precision", "macro_recall"]


def main():
    X_train, y_train, _, _ = load_train_test()
    folder = ROOT / "output/cv_evaluation_output"
    folder.mkdir(parents=True, exist_ok=True)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    fold_rows, class_rows = [], []

    for name in MODEL_NAMES:
        for fold, (fit_ids, validation_ids) in enumerate(
                outer_cv.split(X_train, y_train), start=1):
            # Tune only inside this outer fold. Its validation rows cannot
            # influence parameter selection, so the fold score stays honest.
            search = GridSearchCV(
                make_pipeline(name), parameter_grid(name), scoring="f1_macro",
                cv=inner_cv, n_jobs=1, refit=True
            )
            search.fit(X_train.iloc[fit_ids], y_train.iloc[fit_ids])
            actual = y_train.iloc[validation_ids]
            predicted = search.predict(X_train.iloc[validation_ids])
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual, predicted, labels=list(LABELS), zero_division=0
            )
            fold_rows.append({
                "model": name, "fold": fold, "macro_f1": np.mean(f1),
                "balanced_accuracy": balanced_accuracy_score(actual, predicted),
                "accuracy": accuracy_score(actual, predicted),
                "macro_precision": np.mean(precision),
                "macro_recall": np.mean(recall),
            })
            for index, label in enumerate(LABELS.values()):
                class_rows.append({"model": name, "fold": fold, "class": label,
                                   "precision": precision[index],
                                   "recall": recall[index], "f1": f1[index]})
            print(f"{name}, fold {fold}: Macro F1={np.mean(f1):.3f}")

    folds = pd.DataFrame(fold_rows)
    classes = pd.DataFrame(class_rows)
    summary = folds.groupby("model")[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    folds.to_csv(folder / "fold_scores.csv", index=False)
    summary.to_csv(folder / "model_comparison.csv")
    classes.to_csv(folder / "per_class_scores.csv", index=False)
    classes.groupby(["model", "class"])[["precision", "recall", "f1"]].mean().to_csv(
        folder / "per_class_comparison.csv"
    )

    # The winner is selected using training data only. Step 6 reads this file.
    selected_model = summary["macro_f1_mean"].idxmax()
    (folder / "selected_model.txt").write_text(selected_model)

    means = folds.groupby("model")["macro_f1"].mean().sort_values()
    colors = ["#1976a3" if name == means.idxmax() else "#9ba6ac"
              for name in means.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(means.index, means.values, color=colors)
    ax.set(xlabel="Mean Macro F1 across 5 outer folds",
           title="Nested cross-validation using training data only", xlim=(0, 0.56))
    ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(folder / "training_cv_model_comparison.png", dpi=160)
    plt.close(fig)
    print(f"Selected model: {selected_model}")


if __name__ == "__main__":
    main()
