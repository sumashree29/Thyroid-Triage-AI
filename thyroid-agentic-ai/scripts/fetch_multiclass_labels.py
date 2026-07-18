import pandas as pd
from pathlib import Path

def main():
    print("Fetching thyroid0387.data for multi-class targets...")
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/thyroid0387.data'
    df = pd.read_csv(url, header=None)
    
    # Extract labels
    raw_labels = df.iloc[:, -1].astype(str).str.split('.\|').str[0]
    
    # 3-tier lookup table
    tier_lookup = {
        '-': 0, 'K': 0, 'I': 0, 'J': 0, 'S': 0, 'T': 0, 'L': 0,
        'G': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1,
        'A': 2, 'B': 2, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'H': 2
    }
    
    targets = []
    r_indices = []
    
    for idx, raw_lbl in enumerate(raw_labels):
        # Extract letters before the brackets
        letters = raw_lbl.split('[')[0] if '[' in raw_lbl else raw_lbl
        
        if 'R' in letters:
            r_indices.append(idx)
            targets.append(None)
            continue
            
        max_tier = -1
        for char in letters:
            if char not in tier_lookup:
                raise ValueError(f"Unrecognized diagnostic code: '{char}' in '{letters}' at row {idx}")
            max_tier = max(max_tier, tier_lookup[char])
            
        targets.append(max_tier)
        
    df = df.drop(columns=[29])
    df['target'] = targets
    
    # Define columns
    cols = [
        'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
        'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
        'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
        'tsh_measured', 'tsh', 't3_measured', 't3', 'tt4_measured', 'tt4', 't4u_measured',
        't4u', 'fti_measured', 'fti', 'tbg_measured', 'tbg', 'referral_source', 'target'
    ]
    df.columns = cols
    
    # Drop rows where target is -1
    df = df[df['target'] != -1].copy()
    
    # Split into R class and regular dataset
    r_df = df.iloc[r_indices].copy()
    r_df = r_df.drop(columns=['target'])
    
    valid_df = df.dropna(subset=['target']).copy()
    valid_df['target'] = valid_df['target'].astype(int)
    
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    valid_path = out_dir / "Thyroid_Data_multiclass.csv"
    r_path = out_dir / "real_confounders.csv"
    
    valid_df.to_csv(valid_path, index=False)
    r_df.to_csv(r_path, index=False)
    
    print(f"Saved {len(valid_df)} rows to {valid_path}")
    print(f"Saved {len(r_df)} real confounder rows to {r_path}")
    print("Multi-class target distribution:")
    print(valid_df['target'].value_counts())

if __name__ == "__main__":
    main()
