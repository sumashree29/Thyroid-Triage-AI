import sys
import pickle
import numpy as np
import pandas as pd
import math
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agents.risk_scoring import RiskScoringAgent

def main():
    # 1. Load data/processed/X_test.csv, y_test.csv
    x_test_path = Path("data/processed/X_test.csv")
    y_test_path = Path("data/processed/y_test.csv")
    
    if not x_test_path.exists() or not y_test_path.exists():
        print("X_test.csv or y_test.csv not found in data/processed/")
        # fallback: run train_model to generate splits if missing? 
        # (assuming they exist per user comment)
        return
        
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    
    # 2. Split in half (fixed random_state for reproducibility) → calibration_set, final_test_set
    X_calib, X_final, y_calib, y_final = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    
    # Save final_test_set so evaluate_system.py uses ONLY this half
    X_final.to_csv("data/processed/X_final_test.csv", index=False)
    y_final.to_frame(name='target').to_csv("data/processed/y_final_test.csv", index=False)
    print(f"Saved X_final_test and y_final_test with {len(X_final)} samples.")
    
    # 3. Run existing risk_scorer on calibration_set (no retraining)
    # The risk scorer uses a preprocessor. If X_test is already preprocessed,
    # we need the raw model. Let's load the model directly for simplicity,
    # or just use the model inside RiskScoringAgent if it expects preprocessed vs raw.
    # Wait, train_model.py says:
    #   X_test_t = self.preprocessor.transform(self.X_test)
    #   m.fit(X_train_t, self.y_train)
    # So X_test in data/processed might not be saved! Wait, the user said:
    # "I found: data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv — already split"
    # If they are already split, we assume they are the preprocessed features or raw?
    # Let's check X_test contents. I viewed it earlier: it has floats, some negatives, one-hot encoded looking columns.
    # It is PREPROCESSED!
    # So we can't pass it to RiskScoringAgent.score_patient() because that expects raw patient data!
    # We must load `models/model.pkl` directly and run `predict_proba`.
    with open("models/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("models/risk_classifier.pkl", "rb") as f:
        model = pickle.load(f)
        
    X_calib_t = encoder.transform(X_calib)
    
    # convert to dense if necessary
    def _to_dense(x):
        if hasattr(x, 'toarray'): return x.toarray()
        return x
    
    X_calib_dense = _to_dense(X_calib_t)
    probs = model.predict_proba(X_calib_dense)
    
    # 4. Compute nonconformity scores + q_hat
    n = len(y_calib)
    nonconformity_scores = []
    
    y_calib_arr = y_calib.values
    for i in range(n):
        # 1 minus the probability the model assigned to the correct class
        true_class = int(y_calib_arr[i])
        s_i = 1.0 - probs[i, true_class]
        nonconformity_scores.append(s_i)
        
    nonconformity_scores = np.sort(nonconformity_scores)
    
    alpha = 0.05  # for 95% coverage
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    # Cap q_level at 1.0
    if q_level > 1.0:
        q_level = 1.0
        
    # empirical quantile
    q_hat = np.quantile(nonconformity_scores, q_level, method='higher')
    
    print(f"Calibration size: {n}")
    print(f"alpha: {alpha}")
    print(f"q_hat (nonconformity threshold): {q_hat:.4f}")
    
    # 5. Save to models/conformal_threshold.pkl
    threshold_data = {
        "q_hat": float(q_hat),
        "alpha": alpha,
        "calibration_size": n
    }
    
    out_path = Path("models/conformal_threshold.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(threshold_data, f)
        
    print(f"Saved threshold to {out_path}")

if __name__ == "__main__":
    main()
