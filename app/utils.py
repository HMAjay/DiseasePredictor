import pandas as pd

def predict_and_explain(symptom_severity, all_symptoms, model, rf_model, label_encoder):

    symptom_severity = {
        k.lower().strip(): v for k, v in symptom_severity.items()
    }

    input_vector = [symptom_severity.get(s, 0) for s in all_symptoms]
    input_df = pd.DataFrame([input_vector], columns=all_symptoms)

    probs = model.predict_proba(input_df)[0]
    classes = label_encoder.inverse_transform(range(len(probs)))

    results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

    # explainability
    importances = rf_model.feature_importances_

    explanation = []
    for i, val in enumerate(input_vector):
        if val > 0:
            explanation.append((all_symptoms[i], val * importances[i]))

    explanation = sorted(explanation, key=lambda x: x[1], reverse=True)

    return results[:3], explanation[:5]