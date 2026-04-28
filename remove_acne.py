import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW = BASE / 'data' / 'raw' / 'dataset.csv'
CLEAN = BASE / 'data' / 'raw' / 'dataset_clean.csv'


def remove_acne(path: Path):
    if not path.exists():
        return False
    df = pd.read_csv(path)
    before = len(df)
    df = df[~df['Disease'].astype(str).str.strip().str.lower().eq('acne')]
    after = len(df)
    df.to_csv(path, index=False)
    print(f"Updated {path.name}: {before} -> {after} rows (removed Acne)")
    return True


def main():
    removed = False
    for p in [RAW, CLEAN]:
        if p.exists():
            ok = remove_acne(p)
            removed = removed or ok
        else:
            print(f"File not found: {p}")
    if not removed:
        print("No files updated (files missing?).")


if __name__ == '__main__':
    main()
