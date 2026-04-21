import joblib
symptoms = joblib.load("models/symptoms.pkl")
print(symptoms[:20])