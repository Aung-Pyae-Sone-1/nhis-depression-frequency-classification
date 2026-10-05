"""Step 6: evaluate the CV-selected model once on the untouched test set."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)

from project_common import LABELS, ROOT, load_train_test, make_pipeline


def calculate_scores(actual, predicted):
    """Calculate overall and class-level classification scores."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=list(LABELS), zero_division=0
    )
    overall = {
        "macro_f1": np.mean(f1),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "accuracy": accuracy_score(actual, predicted),
        "macro_precision": np.mean(precision),
        "macro_recall": np.mean(recall),
    }
    return overall, precision, recall, f1


def main():
    X_train, y_train, X_test, y_test = load_train_test()
    selected = (ROOT / "output/cv_evaluation_output/selected_model.txt").read_text().strip()
    settings = json.loads((ROOT / "output/tuning_output/best_parameters.json").read_text())
    folder = ROOT / "output/test_set_output"
    folder.mkdir(parents=True, exist_ok=True)

    # The test set is used here for the first and only model evaluation.
    model = make_pipeline(selected).set_params(**settings[selected])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    overall, precision, recall, f1 = calculate_scores(y_test, predicted)
    pd.DataFrame([{"model": selected, **overall}]).to_csv(
        folder / "final_metrics.csv", index=False
    )
    pd.DataFrame({
        "class": list(LABELS.values()), "precision": precision,
        "recall": recall, "f1": f1,
    }).to_csv(folder / "per_class_metrics.csv", index=False)
    pd.DataFrame({"actual": y_test.to_numpy(), "predicted": predicted}).to_csv(
        folder / "test_predictions.csv", index=False
    )
    matrix = confusion_matrix(y_test, predicted, labels=list(LABELS), normalize="true")
    pd.DataFrame(matrix, index=LABELS.values(), columns=LABELS.values()).to_csv(
        folder / "confusion_matrix.csv"
    )
    joblib.dump(model, folder / "selected_model.joblib")

    # A simple baseline shows whether the trained model adds useful information.
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_scores, _, _, _ = calculate_scores(y_test, dummy.predict(X_test))
    pd.DataFrame([{"model": "DummyMostFrequent", **dummy_scores}]).to_csv(
        folder / "dummy_metrics.csv", index=False
    )
    print(f"Selected model: {selected}")
    print(f"Untouched-test Macro F1: {overall['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
