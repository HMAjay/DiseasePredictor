import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / 'data' / 'raw' / 'dataset_clean.csv')
cols = [c for c in df.columns if 'Symptom' in c]
skin_tokens = set(['skin_rash','itching','blackheads','pus_filled_pimples','scurring','dischromic_patches','nodal_skin_eruptions','acne','scaly_skin','pustules','rash'])
disease_skin = set()
for _, r in df.iterrows():
    toks = {str(r[c]).strip().lower() for c in cols if pd.notna(r[c])}
    if toks & skin_tokens:
        disease_skin.add(str(r['Disease']).strip())

print('Found skin diseases (sample):', list(sorted(disease_skin))[:30])
print('Total skin diseases count:', len(disease_skin))
print('Is Acne present?:', 'Acne' in disease_skin or 'acne' in disease_skin)
