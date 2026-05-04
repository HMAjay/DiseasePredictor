import pandas as pd
from rapidfuzz import process, fuzz
import joblib
from pathlib import Path
import re

def predict_and_explain(symptom_severity, all_symptoms, model, rf_model, label_encoder):

    symptom_severity = {
        k.lower().strip(): v for k, v in symptom_severity.items()
    }

    input_vector = [symptom_severity.get(s, 0) for s in all_symptoms]
    input_df = pd.DataFrame([input_vector], columns=all_symptoms)

    probs = model.predict_proba(input_df)[0]
    classes = label_encoder.inverse_transform(range(len(probs)))

    results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

    # explainability: try to get feature importances from the provided rf_model.
    # If rf_model is a calibrated wrapper it won't expose importances, so
    # fall back to the saved uncalibrated RandomForest at models/rf_model.pkl.
    importances = getattr(rf_model, "feature_importances_", None)
    if importances is None:
        try:
            models_dir = Path(__file__).resolve().parent.parent / "models"
            rf_uncal = joblib.load(models_dir / "rf_model.pkl")
            importances = rf_uncal.feature_importances_
        except Exception:
            # As a last resort, set uniform importances to avoid crashing.
            importances = [1.0 / len(all_symptoms)] * len(all_symptoms)

    explanation = []
    for i, val in enumerate(input_vector):
        if val > 0:
            explanation.append((all_symptoms[i], val * importances[i]))

    explanation = sorted(explanation, key=lambda x: x[1], reverse=True)

    return results[:3], explanation[:5]


def extract_symptoms_from_text(text: str, all_symptoms: list, min_word_len: int = 3, fuzz_threshold: int = 90):
    """Extract symptom tokens from free text.

    Strategy:
    - Normalize text, split into words.
    - Try greedy longest-match against `all_symptoms` (which are tokenized with underscores).
    - Prefer exact matches; fall back to rapidfuzz fuzzy match with threshold.
    """
    if not text:
        return []

    txt = text.lower()
    txt = re.sub(r"[^a-z0-9 _]", " ", txt)
    words = [w for w in txt.split() if len(w) >= min_word_len]
    matched = []

    # try n-grams from longest to shortest
    max_n = min(4, len(words))
    i = 0
    while i < len(words):
        found = False
        for n in range(max_n, 0, -1):
            if i + n > len(words):
                continue
            phrase = "_".join(words[i:i+n])
            # exact match
            if phrase in all_symptoms:
                matched.append(phrase)
                i += n
                found = True
                break
            # fuzzy match
            best = process.extractOne(phrase, all_symptoms, scorer=fuzz.WRatio)
            if best and best[1] >= fuzz_threshold:
                matched.append(best[0])
                i += n
                found = True
                break
        if not found:
            i += 1

    # deduplicate keeping order
    seen = set()
    out = []
    for m in matched:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def post_filter_predictions(results, detected_symptoms, dataset_path=None, demote_factor: float = 0.3):
    """Apply simple rule-based post-filtering to prediction probabilities.

    - Demotes dermatology/skin diseases if no explicit skin symptom was detected.
    - `results` is list[(disease, prob)]. Returns reordered list.
    """
    if not results:
        return results

    # Define skin symptom tokens
    skin_tokens = set([
        'skin_rash', 'itching', 'blackheads', 'pus_filled_pimples', 'scurring',
        'dischromic_patches', 'nodal_skin_eruptions', 'acne', 'scaly_skin'
    ])

    # Load disease->symptom mapping from cleaned dataset if available
    disease_skin = set()
    disease_liver = set()
    try:
        from pathlib import Path
        import pandas as pd
        dp = Path(dataset_path) if dataset_path else Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'dataset_clean.csv'
        if dp.exists():
            df = pd.read_csv(dp)
            symptom_cols = [c for c in df.columns if 'Symptom' in c]
            for _, row in df.iterrows():
                toks = {str(row[c]).strip().lower() for c in symptom_cols if pd.notna(row[c])}
                if toks & skin_tokens:
                    disease_skin.add(str(row['Disease']).strip())
                # detect hepatic/jaundice diseases
                liver_tokens = set(['yellowish_skin', 'yellowing_of_eyes', 'yellow_urine', 'dark_urine', 'jaundice', 'yellowing_of_eyes'])
                if toks & liver_tokens:
                    disease_liver.add(str(row['Disease']).strip())
    except Exception:
        disease_skin = set()
        disease_liver = set()

    # Determine if user provided any skin symptom or liver/jaundice symptom
    user_has_skin = any(s in skin_tokens for s in detected_symptoms)
    liver_tokens_input = set(['yellowish_skin', 'yellowing_of_eyes', 'yellow_urine', 'dark_urine', 'jaundice'])
    user_has_liver = any(s in liver_tokens_input for s in detected_symptoms)

    new_results = []
    # Boost liver diseases if user reported liver symptoms; otherwise demote skin diseases
    boost_factor = 1.5
    for disease, prob in results:
        label = disease.strip()
        if user_has_liver and label in disease_liver:
            prob = prob * boost_factor
        elif not user_has_skin and label in disease_skin:
            prob = prob * demote_factor
        new_results.append((disease, prob))

    # Normalize probabilities to sum to 1 to keep them meaningful
    total = sum(p for _, p in new_results)
    if total > 0:
        new_results = [(d, p / total) for d, p in new_results]

    new_results = sorted(new_results, key=lambda x: x[1], reverse=True)
    return new_results

def get_best_matches(query, all_symptoms, limit=5):
    """Fuzzy search for symptoms."""
    matches = process.extract(
        query, 
        all_symptoms, 
        scorer=fuzz.WRatio, 
        limit=limit
    )
    return [match[0] for match in matches if match[1] > 60]
