"""Small shared settings so every step uses the same data and models."""

from pathlib import Path
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "output/cleaning_output/cleaned_data.csv"
TRAIN_FILE = ROOT / "output/split_output/train_data.csv"
TEST_FILE = ROOT / "output/split_output/test_data.csv"
TARGET = "DEPFREQ_A"
AGE = ["AGEP_A"]
CATEGORIES = [
    "SEX_A", "MARITAL_A", "ANXEV_A", "ANXFREQ_A", "NERVOUS_A",
    "RESTLESS_A", "LSATIS4_A", "PHSTAT_A", "SLPREST_A", "SUPPORT_A",
]
LABELS = {1: "Daily", 2: "Weekly", 3: "Monthly", 4: "Few times/year", 5: "Never"}
SEED = 42


class SurveyXGBClassifier(ClassifierMixin, BaseEstimator):
    """Use this target's NHIS labels 1–5; weight training rows only."""

    def __init__(self, max_depth=3, learning_rate=0.1,
                 n_estimators=200, weight_mode="none"):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.weight_mode = weight_mode

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # This dataset's target has exactly classes 1–5. XGBoost uses 0–4.
        weights = (compute_sample_weight("balanced", y)
                   if self.weight_mode != "none" else None)
        if self.weight_mode == "sqrt":
            weights = np.sqrt(weights)
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, tree_method="hist",
            objective="multi:softprob", random_state=SEED, n_jobs=2,
        )
        self.model_.fit(X, np.asarray(y) - 1, sample_weight=weights)
        return self

    def predict(self, X):
        return self.model_.predict(X) + 1

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class SurveyCatBoostClassifier(ClassifierMixin, BaseEstimator):
    """Fit CatBoost with age numeric and the remaining columns categorical."""

    def __init__(self, depth=6, learning_rate=0.1, iterations=200,
                 l2_leaf_reg=3, weight_mode="none"):
        self.depth = depth
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_leaf_reg = l2_leaf_reg
        self.weight_mode = weight_mode

    def _prepare(self, X):
        # The first column is age. CatBoost receives all other columns as text
        # categories, which prevents survey codes from being treated as amounts.
        prepared = pd.DataFrame(X).copy()
        prepared.iloc[:, 0] = pd.to_numeric(prepared.iloc[:, 0])
        for column in prepared.columns[1:]:
            prepared[column] = prepared[column].astype(str)
        return prepared

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        weights = (compute_sample_weight("balanced", y)
                   if self.weight_mode != "none" else None)
        if self.weight_mode == "sqrt":
            weights = np.sqrt(weights)
        self.model_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, l2_leaf_reg=self.l2_leaf_reg,
            loss_function="MultiClass", random_seed=SEED,
            thread_count=2, verbose=False, allow_writing_files=False,
        )
        self.model_.fit(self._prepare(X), y,
                        cat_features=list(range(1, X.shape[1])),
                        sample_weight=weights)
        return self

    def predict(self, X):
        return self.model_.predict(self._prepare(X)).ravel().astype(int)

    def predict_proba(self, X):
        return self.model_.predict_proba(self._prepare(X))


def load_data():
    """Return predictors and the five-class target."""
    df = pd.read_csv(DATA_FILE)
    return df.drop(columns=TARGET), df[TARGET]


def load_train_test():
    """Return the saved training and untouched test datasets."""
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    return (train.drop(columns=TARGET), train[TARGET],
            test.drop(columns=TARGET), test[TARGET])


def make_pipeline(model_name):
    """Encode the cleaned predictors, then fit the chosen model."""
    age_steps = Pipeline([
        ("scale", StandardScaler()),
    ])
    category_steps = Pipeline([
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prepare = ColumnTransformer([
        ("age", age_steps, AGE),
        ("categories", category_steps, CATEGORIES),
    ])
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100, random_state=SEED, n_jobs=2
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=100, random_state=SEED
        ),
        "XGBClassifier": SurveyXGBClassifier(),
        "CatBoostClassifier": SurveyCatBoostClassifier(),
    }
    if model_name == "CatBoostClassifier":
        # CatBoost handles categories itself, so do not one-hot encode them.
        prepare = ColumnTransformer([
            ("age", "passthrough", AGE),
            ("categories", "passthrough", CATEGORIES),
        ])
    return Pipeline([("prepare", prepare), ("model", models[model_name])])


def parameter_grid(model_name, catboost_stage="final"):
    """Return the final CV grid or an optional CatBoost exploration grid."""
    if model_name == "CatBoostClassifier" and catboost_stage == "stage1":
        return {
            "model__depth": [4, 6],
            "model__learning_rate": [0.05, 0.1],
            "model__weight_mode": ["sqrt", "balanced"],
        }
    if model_name == "CatBoostClassifier" and catboost_stage == "stage2":
        return {
            "model__depth": [3, 4],
            "model__iterations": [200, 300],
            "model__learning_rate": [0.1],
            "model__weight_mode": ["sqrt"],
        }
    return {
        "LogisticRegression": {
            "model__C": [0.1, 1], "model__class_weight": [None, "balanced"]
        },
        "RandomForestClassifier": {
            "model__max_depth": [12, None],
            "model__class_weight": [None, "balanced_subsample"],
        },
        "HistGradientBoostingClassifier": {
            "model__max_leaf_nodes": [15, 31],
            "model__class_weight": [None, "balanced"],
        },
        "XGBClassifier": {
            "model__max_depth": [2, 3],
            "model__learning_rate": [0.05, 0.1],
            "model__n_estimators": [200, 400],
            "model__weight_mode": ["none", "sqrt", "balanced"],
        },
        "CatBoostClassifier": {
            "model__depth": [3, 4],
            "model__iterations": [300],
            "model__learning_rate": [0.1],
            "model__weight_mode": ["sqrt"],
        },
    }[model_name]


MODEL_NAMES = [
    "LogisticRegression", "RandomForestClassifier",
    "HistGradientBoostingClassifier", "XGBClassifier",
    "CatBoostClassifier",
]
