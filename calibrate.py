import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "dataset.csv"
SEVERITY_PATH = BASE_DIR / "data" / "raw" / "Symptom-severity.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    severity_df = pd.read_csv(SEVERITY_PATH)
    severity_df.columns = severity_df.columns.str.strip()

    severity_dict = {
        str(row["Symptom"]).strip().lower().replace("_", " "): int(row["weight"])
        for _, row in severity_df.iterrows()
    }

    symptom_cols = [col for col in df.columns if "Symptom" in col]
    for col in symptom_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("_", " ", regex=False)
        )

    all_symptoms = set()
    for col in symptom_cols:
        all_symptoms.update(df[col].replace("nan", np.nan).dropna().unique())
    all_symptoms = sorted(all_symptoms)

    def encode_row(row):
        symptoms = set(row.dropna())
        vector = []
        for s in all_symptoms:
            if s in symptoms:
                vector.append(severity_dict.get(s, 1))
            else:
                vector.append(0)
        return vector

    X = pd.DataFrame(
        [encode_row(row) for _, row in df[symptom_cols].iterrows()],
        columns=all_symptoms,
    )

    y = df["Disease"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le


def multiclass_brier_score(y_true, probs, n_classes):
    # y_true: integer labels
    # probs: (n_samples, n_classes)
    from sklearn.metrics import mean_squared_error

    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(y_true)), y_true] = 1
    return mean_squared_error(y_onehot, probs)


def main():
    X, y, le = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    rf_path = MODEL_DIR / "rf_model.pkl"
    if rf_path.exists():
        print("Loading existing RandomForest from models/")
        rf = joblib.load(rf_path)
    else:
        rf = RandomForestClassifier(n_estimators=200, random_state=42)

    print("Fitting base RandomForest on training split...")
    rf.fit(X_train, y_train)

    print("Calibrating probabilities with isotonic calibration (CV=5)...")
    calib = CalibratedClassifierCV(rf, cv=5, method="isotonic")
    calib.fit(X_train, y_train)

    probs = calib.predict_proba(X_test)
    logloss = log_loss(y_test, probs)
    brier = multiclass_brier_score(y_test, probs, probs.shape[1])

    print(f"Log-loss on test set: {logloss:.4f}")
    print(f"Multiclass Brier score (MSE): {brier:.4f}")

    out_path = MODEL_DIR / "rf_calibrated.pkl"
    joblib.dump(calib, out_path)
    joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
    print(f"Saved calibrated model to: {out_path}")


if __name__ == '__main__':
    main()
