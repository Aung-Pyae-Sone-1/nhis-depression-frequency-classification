"""Step 4: tune every model using only the 80% training dataset."""

import json

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from project_common import (MODEL_NAMES, ROOT, SEED, load_train_test,
                            make_pipeline, parameter_grid)


def main():
    X_train, y_train, _, _ = load_train_test()
    folder = ROOT / "output/tuning_output"
    folder.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_settings, rows = {}, []

    for name in MODEL_NAMES:
        # The untouched 20% test set is never passed to GridSearchCV.
        search = GridSearchCV(
            make_pipeline(name), parameter_grid(name), scoring="f1_macro",
            cv=cv, n_jobs=1, refit=True
        )
        search.fit(X_train, y_train)
        best_settings[name] = search.best_params_
        rows.append({"model": name, "training_cv_macro_f1": search.best_score_,
                     "best_parameters": str(search.best_params_)})
        pd.DataFrame(search.cv_results_).to_csv(
            folder / f"{name}_search_details.csv", index=False
        )
        print(f"{name}: Macro F1={search.best_score_:.3f}")

    (folder / "best_parameters.json").write_text(
        json.dumps(best_settings, indent=2)
    )
    pd.DataFrame(rows).to_csv(folder / "tuning_results.csv", index=False)


if __name__ == "__main__":
    main()
