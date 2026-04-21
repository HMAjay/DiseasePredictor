import pandas as pd


def predict(selected_symptoms, all_symptoms, model, label_encoder):

    # Normalize input symptoms
    selected_symptoms = [s.lower().strip() for s in selected_symptoms]

    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]

    input_df = pd.DataFrame([input_vector], columns=all_symptoms)

    probabilities = model.predict_proba(input_df)[0]

    class_labels = label_encoder.inverse_transform(range(len(probabilities)))

    results = sorted(
        zip(class_labels, probabilities),
        key=lambda x: x[1],
        reverse=True
    )
    print("Selected:", selected_symptoms)
    print("Matched features:", [s for s in selected_symptoms if s in all_symptoms])
    return results[:3]