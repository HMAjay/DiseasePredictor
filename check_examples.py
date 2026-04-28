import sys
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))

from utils import predict_and_explain, get_best_matches, post_filter_predictions

MODEL_DIR = ROOT / "models"

model = joblib.load(MODEL_DIR / "model.pkl")
rf_model = None
calib = MODEL_DIR / "rf_calibrated.pkl"
if calib.exists():
    rf_model = joblib.load(calib)
else:
    rf_model = joblib.load(MODEL_DIR / "rf_model.pkl")
label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
symptoms_list = joblib.load(MODEL_DIR / "symptoms.pkl")

examples = [
    "I have a high fever, severe headache, nausea and abdominal pain.",
    "I've been having continuous sneezing, itchy watery eyes and chills."
]

def nl_to_severity(text, symptoms_list):
    words = text.lower().replace(",", " ").replace(".", " ").split()
    selected = []
    for w in words:
        if len(w) > 3:
            matches = get_best_matches(w, symptoms_list, limit=1)
            if matches:
                selected.append(matches[0])
    selected = list(dict.fromkeys(selected))
    # default severity 2 for matched symptoms
    return {s: 2 for s in selected}

for ex in examples:
    sev = nl_to_severity(ex, symptoms_list)
    results, explanation = predict_and_explain(sev, symptoms_list, model, rf_model, label_encoder)
    # post-filter using detected symptoms
    detected = list(sev.keys())
    results = post_filter_predictions(results, detected)
    print("\nInput:", ex)
    print("Top predictions:")
    for d, p in results:
        print(f" - {d}: {p*100:.2f}%")
    print("Explanation (top features):")
    for s, score in explanation:
        print(f" - {s}: {score:.4f}")
