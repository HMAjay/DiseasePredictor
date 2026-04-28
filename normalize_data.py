import pandas as pd
from pathlib import Path
import re


BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "raw" / "dataset.csv"
SEV_PATH = BASE / "data" / "raw" / "Symptom-severity.csv"


def clean_token(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s)
    s = s.strip().lower()
    # normalize spaces
    s = re.sub(r"\s+", " ", s)
    # fix stray spaces around underscores
    s = s.replace(" _", "_").replace("_ ", "_")
    # replace remaining spaces with underscore for consistent tokens
    s = s.replace(" ", "_")
    # remove accidental commas
    s = s.replace(",", "")
    return s


def normalize_dataset():
    df = pd.read_csv(DATA_PATH)
    symptom_cols = [c for c in df.columns if "Symptom" in c]

    for col in symptom_cols:
        df[col] = df[col].apply(clean_token)

    df.to_csv(DATA_PATH, index=False)
    print(f"Normalized dataset saved to: {DATA_PATH}")


def normalize_severity():
    df = pd.read_csv(SEV_PATH)
    if "Symptom" in df.columns:
        df["Symptom"] = df["Symptom"].apply(clean_token)
    df.to_csv(SEV_PATH, index=False)
    print(f"Normalized severity file saved to: {SEV_PATH}")


def main():
    normalize_dataset()
    normalize_severity()


if __name__ == '__main__':
    main()
