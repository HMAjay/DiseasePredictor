import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "raw" / "dataset.csv"
SEVERITY_PATH = BASE_DIR / "data" / "raw" / "Symptom-severity.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ---------------- LOAD ----------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

severity_df = pd.read_csv(SEVERITY_PATH)
severity_df.columns = severity_df.columns.str.strip()

# ---------------- SEVERITY MAP ----------------
severity_dict = {
    str(row["Symptom"]).strip().lower().replace("_", " "): int(row["weight"])
    for _, row in severity_df.iterrows()
}

# ---------------- CLEAN DATA ----------------
symptom_cols = [col for col in df.columns if "Symptom" in col]

for col in symptom_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("_", " ", regex=False)
    )

# ---------------- FEATURE SPACE ----------------
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].replace("nan", np.nan).dropna().unique())

all_symptoms = sorted(all_symptoms)

# ---------------- ENCODE ----------------
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
    columns=all_symptoms
)

y = df["Disease"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---------------- TRAIN TEST ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# ---------------- MODELS ----------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
svm = SVC(probability=True)
knn = KNeighborsClassifier()
nb = GaussianNB()
xgb = XGBClassifier(eval_metric="mlogloss")

models = {
    "RandomForest": rf,
    "SVM": svm,
    "KNN": knn,
    "NaiveBayes": nb,
    "XGBoost": xgb
}

# ---------------- EVALUATE ----------------
model_scores = {}

print("\n📊 Model Accuracy:\n")

for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    acc = accuracy_score(y_test, preds)
    model_scores[name] = acc

    print(f"{name}: {acc:.4f}")

# ---------------- BEST MODEL ----------------
best_model_name = max(model_scores, key=model_scores.get)
print(f"\n🏆 Best Model: {best_model_name}")

# ---------------- ENSEMBLE ----------------
ensemble = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("svm", svm),
        ("knn", knn),
        ("nb", nb),
        ("xgb", xgb),
    ],
    voting="soft"
)

ensemble.fit(X, y_encoded)

# retrain RF for explainability
rf.fit(X, y_encoded)

# ---------------- SAVE ----------------
joblib.dump(ensemble, MODEL_DIR / "model.pkl")
joblib.dump(rf, MODEL_DIR / "rf_model.pkl")
joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
joblib.dump(all_symptoms, MODEL_DIR / "symptoms.pkl")
joblib.dump(model_scores, MODEL_DIR / "model_scores.pkl")
joblib.dump(best_model_name, MODEL_DIR / "best_model.pkl")

print("\n✅ Training Complete!")