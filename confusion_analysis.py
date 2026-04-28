import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data' / 'raw' / 'dataset_clean.csv'

def load_X_y():
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()
    symptom_cols = [c for c in df.columns if 'Symptom' in c]

    # build all_symptoms
    all_symptoms = sorted({str(s).strip() for c in symptom_cols for s in df[c].dropna().unique()})

    def encode_row(row):
        syms = set([str(row[c]).strip() for c in symptom_cols if pd.notna(row[c])])
        return [1 if s in syms else 0 for s in all_symptoms]

    X = pd.DataFrame([encode_row(r) for _, r in df.iterrows()], columns=all_symptoms)
    y = df['Disease'].astype(str).str.strip()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le

def main():
    X, y, le = load_X_y()
    # simple train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # load model
    import joblib
    model = joblib.load(BASE / 'models' / 'model.pkl')

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = le.inverse_transform(range(len(le.classes_)))

    # find top confusions (off-diagonal largest counts)
    cm2 = cm.copy()
    np.fill_diagonal(cm2, 0)
    flat = []
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            if cm2[i,j] > 0:
                flat.append((cm2[i,j], labels[i], labels[j]))
    flat.sort(reverse=True)

    print('Top confusions (count, true_label, predicted_label):')
    for cnt, t, p in flat[:20]:
        print(cnt, t, '->', p)

if __name__ == '__main__':
    main()
