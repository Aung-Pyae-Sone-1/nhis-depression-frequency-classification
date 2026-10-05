"""Step 1: keep the chosen columns and mark unanswered questions as missing."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PREDICTORS = [
    # Demographic factors
    "AGEP_A", "SEX_A", "MARITAL_A",
    # Physical health and psychosocial factors
    "ANXEV_A", "ANXFREQ_A", "NERVOUS_A", "RESTLESS_A", "LSATIS4_A",
    "PHSTAT_A", "SLPREST_A", "SUPPORT_A",
]
TARGET = "DEPFREQ_A"


def main():
    # Read only the variables needed for this project. The target goes last.
    df = pd.read_csv(ROOT / "doc/adult24.csv", usecols=PREDICTORS + [TARGET])
    df = df[PREDICTORS + [TARGET]].copy()
    original_rows = len(df)

    # NHIS uses special numbers for refused, unknown, and unanswered responses.
    # Age 85 means "85 or older" and is a valid answer.
    missing_codes = {
        "AGEP_A": [97, 98, 99],
        "SEX_A": [7, 9],
        "MARITAL_A": [7, 8, 9],
        "ANXEV_A": [7, 9],
        "ANXFREQ_A": [7, 8, 9],
        "NERVOUS_A": [7, 8, 9],
        "RESTLESS_A": [7, 8, 9],
        "LSATIS4_A": [7, 9],
        "PHSTAT_A": [7, 9],
        "SLPREST_A": [7, 8, 9],
        "SUPPORT_A": [7, 8, 9],
        TARGET: [7, 8, 9],
    }
    for column, codes in missing_codes.items():
        df[column] = df[column].replace(codes, pd.NA)

    # Enforce the valid ranges listed in the NHIS codebook. Any unexpected
    # code is treated as missing before the row-selection rule is applied.
    valid_values = {
        "SEX_A": [1, 2],
        "MARITAL_A": [1, 2, 3],
        "ANXEV_A": [1, 2],
        "ANXFREQ_A": [1, 2, 3, 4, 5],
        "NERVOUS_A": [1, 2, 3, 4, 5],
        "RESTLESS_A": [1, 2, 3, 4, 5],
        "LSATIS4_A": [1, 2, 3, 4],
        "PHSTAT_A": [1, 2, 3, 4, 5],
        "SLPREST_A": [1, 2, 3, 4],
        "SUPPORT_A": [1, 2, 3, 4, 5],
        TARGET: [1, 2, 3, 4, 5],
    }
    df["AGEP_A"] = df["AGEP_A"].where(df["AGEP_A"].between(18, 85), pd.NA)
    for column, values in valid_values.items():
        df[column] = df[column].where(df[column].isin(values), pd.NA)

    # Record data loss before complete-case filtering. This makes the cleaning
    # decision visible instead of showing only the zero missing values afterward.
    missing_before = df.isna().sum().rename("missing_or_invalid_rows")
    incomplete_rows = df[PREDICTORS + [TARGET]].isna().any(axis=1).sum()

    # Keep only complete rows with valid answers for every predictor and target.
    df = df.dropna(subset=PREDICTORS + [TARGET])
    df[TARGET] = df[TARGET].astype(int)

    folder = ROOT / "output/cleaning_output"
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / "cleaned_data.csv", index=False)
    missing_before.to_csv(folder / "missing_values_before_complete_case.csv")
    pd.DataFrame([
        {"stage": "Raw NHIS sample adult file", "rows": original_rows},
        {"stage": "Rows removed by complete-case rule", "rows": incomplete_rows},
        {"stage": "Final complete-case dataset", "rows": len(df)},
    ]).to_csv(folder / "row_selection_audit.csv", index=False)
    print(f"Rows: {original_rows:,} before, {len(df):,} after cleaning")
    print("Every selected predictor and the target contains a valid response.")
    print(f"Saved {folder / 'cleaned_data.csv'}")


if __name__ == "__main__":
    main()
