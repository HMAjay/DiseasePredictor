import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/raw/dataset.csv")

# Clean columns
df.columns = df.columns.str.strip()

# Identify columns
symptom_cols = [col for col in df.columns if "Symptom" in col]

# Clean symptom values
for col in symptom_cols:
    df[col] = df[col].str.strip().str.lower().str.replace("_", " ")
# Get all unique symptoms
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().unique())

all_symptoms = sorted(all_symptoms)

# Convert to one-hot
def encode(row):
    symptoms = set(row.dropna())
    return [1 if s in symptoms else 0 for s in all_symptoms]

X = df[symptom_cols].apply(encode, axis=1, result_type="expand")
X.columns = all_symptoms

y = df["Disease"]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier(n_estimators=200)
model.fit(X, y_encoded)

# Save
joblib.dump(model, "models/model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(all_symptoms, "models/symptoms.pkl")

print("Fixed and trained correctly")