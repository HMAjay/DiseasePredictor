import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "raw" / "dataset.csv"
OUT_PATH = BASE / "data" / "raw" / "dataset_clean.csv"


def canonical_symptom_set(row, symptom_cols):
    toks = [str(row[c]).strip().lower() for c in symptom_cols if pd.notna(row[c]) and str(row[c]).strip()!='']
    toks = [t.replace(' ', '_') for t in toks]
    return tuple(sorted(set(toks)))


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    symptom_cols = [c for c in df.columns if 'Symptom' in c]

    # Build canonical symptom set for each row
    df['symptom_set'] = df.apply(lambda r: canonical_symptom_set(r, symptom_cols), axis=1)

    # Exact duplicate rows
    before = len(df)
    df_no_dup = df.drop_duplicates()
    after_dup = len(df_no_dup)

    # For each unique symptom_set, see how many distinct diseases map to it
    mapping = df_no_dup.groupby('symptom_set')['Disease'].agg(lambda s: list(s))

    contradictions = {}
    for sset, diseases in mapping.items():
        uniq = sorted(set([d.strip() for d in diseases if pd.notna(d)]))
        if len(uniq) > 1:
            contradictions[sset] = uniq

    # Resolve contradictions by majority vote (safe heuristic) and report
    resolved = []
    for sset, diseases in contradictions.items():
        # count occurrences in original df
        cnt = Counter([d.strip() for d in df[df['symptom_set']==sset]['Disease']])
        most_common = cnt.most_common(1)[0][0]
        resolved.append((sset, uniq, most_common))

    # Apply resolution: keep rows with the majority label for each sset
    df_clean_rows = []
    for sset, group in df_no_dup.groupby('symptom_set'):
        if sset in contradictions:
            cnt = Counter([d.strip() for d in df[df['symptom_set']==sset]['Disease']])
            most_common = cnt.most_common(1)[0][0]
            # create a single row with that disease and the symptom columns filled
            # find a source row to copy symptom values
            source = df[df['symptom_set']==sset].iloc[0]
            new = source.copy()
            new['Disease'] = most_common
            df_clean_rows.append(new)
        else:
            # keep one representative row for this symptom_set
            source = df[df['symptom_set']==sset].iloc[0]
            df_clean_rows.append(source)

    df_clean = pd.DataFrame(df_clean_rows).drop(columns=['symptom_set'])

    df_clean.to_csv(OUT_PATH, index=False)

    # Report
    print(f"Original rows: {before}")
    print(f"After drop_duplicates: {after_dup}")
    print(f"Cleaned rows (unique symptom sets): {len(df_clean)}")
    print(f"Contradictory symptom sets found: {len(contradictions)}")
    if len(contradictions) > 0:
        print("Sample contradictions (symptom_set -> diseases):")
        i = 0
        for sset, diseases in list(contradictions.items())[:10]:
            print(sset, "->", diseases)
            i += 1

    print(f"Clean dataset written to: {OUT_PATH}")


if __name__ == '__main__':
    main()
