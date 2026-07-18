import pandas as pd
import numpy as np
import random
from pathlib import Path

def main():
    x_path = Path("data/processed/X_test.csv")
    y_path = Path("data/processed/y_test.csv")
    if not x_path.exists():
        x_path = Path("../data/processed/X_test.csv")
        y_path = Path("../data/processed/y_test.csv")
        
    df = pd.read_csv(x_path, na_values='?')
    y = pd.read_csv(y_path)
    df['target'] = y['target']
    
    # We want genuine normal cases (Low Risk, target == 0)
    # The original condition used TSH, but now we can just use the true label = 0.
    normals = df[(df['target'] == 0) & df['fti'].notna() & df['t3'].notna()].copy()
    
    N = 200
    if len(normals) < N:
        N = len(normals)
    
    synthetic_cases = normals.sample(n=N, random_state=42).copy()
    
    # Perturb FT4 (using FTI as proxy) and T3 upward while holding TSH constant
    # (magnitudes loosely based on reported biotin-interference effect sizes)
    # *Note: These are for illustrative purposes and constitute a limitation of this study.*
    np.random.seed(42)
    synthetic_cases['fti'] *= np.random.uniform(1.3, 1.8, size=N)
    synthetic_cases['t3'] *= np.random.uniform(1.2, 1.6, size=N)
    
    # Label these rows "synthetic_interference" separately from the true diagnosis label
    synthetic_cases['interference_label'] = 'synthetic_interference'
    
    # Save to data/processed
    out_dir = Path("data/processed")
    if not out_dir.exists():
         out_dir = Path("../data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_interference_cases.csv"
    
    synthetic_cases.to_csv(out_path, index=False)
    print(f"Injected synthetic interference into {N} cases and saved to {out_path}")

if __name__ == "__main__":
    main()
