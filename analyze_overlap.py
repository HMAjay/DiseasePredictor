import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "raw" / "dataset.csv"

df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()
symptom_cols = [c for c in df.columns if "Symptom" in c]

def normalize_token(s):
    if pd.isna(s):
        return None
    return str(s).strip().lower().replace(' ', '_')

for col in symptom_cols:
    df[col] = df[col].apply(normalize_token)

def disease_symptoms(disease):
    subset = df[df['Disease'].str.strip().str.lower() == disease.lower()]
    tokens = set()
    for col in symptom_cols:
        tokens.update(subset[col].dropna().unique())
    return subset, sorted([t for t in tokens if t])

examples = {
    'example1': ['high_fever', 'headache', 'nausea', 'abdominal_pain'],
    'example2': ['continuous_sneezing', 'watering_from_eyes', 'chills', 'itchy_eyes']
}

def match_counts(symptoms):
    rows = []
    for idx, row in df.iterrows():
        row_symptoms = set(row[symptom_cols].dropna().values)
        match = len(set(symptoms) & row_symptoms)
        if match > 0:
            rows.append((row['Disease'], match, row_symptoms))
    out = pd.DataFrame(rows, columns=['Disease', 'matches', 'row_symptoms'])
    summary = out.groupby('Disease')['matches'].agg(['count', 'sum']).sort_values('sum', ascending=False)
    return summary, out

for name, syms in examples.items():
    print(f"\n=== {name} symptoms: {syms}")
    summary, out = match_counts(syms)
    print(summary.head(10))
    # show top 5 matching rows
    top = out.sort_values('matches', ascending=False).head(5)
    print('\nTop matching rows (disease, matches, symptoms):')
    for _, r in top.iterrows():
        print(r['Disease'], r['matches'], sorted(list(r['row_symptoms']))[:10])

# Show unique symptoms for Acne
subset, acne_syms = disease_symptoms('Acne')
print('\n\nUnique symptom tokens for Acne (sample 50):')
print(acne_syms[:50])
print('\nNumber of Acne rows:', len(subset))
