import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agents.risk_scoring import RiskScoringAgent
from agents.confounder import ConfounderAgent
from core.conformal import ConformalWrapper

def main():
    # Load Models and Agents
    model_path = "models/risk_classifier.pkl"
    with open(model_path, "rb") as f:
        raw_model = pickle.load(f)
    
    with open("models/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    
    risk_scorer = RiskScoringAgent(model_path, "models/encoder.pkl")
    confounder = ConfounderAgent()
    conformal = ConformalWrapper("models/conformal_threshold.pkl")
    
    results = []
    
    # 1. Evaluate on X_final_test (Preprocessed) for Conformal Coverage
    print("Evaluating on X_final_test (Raw data being processed)...")
    try:
        X_final = pd.read_csv("data/processed/X_final_test.csv")
        y_final = pd.read_csv("data/processed/y_final_test.csv").squeeze()
        
        X_final_t = encoder.transform(X_final)
        
        def _to_dense(x):
            if hasattr(x, 'toarray'): return x.toarray()
            return x
        
        X_dense = _to_dense(X_final_t)
        probs = raw_model.predict_proba(X_dense)
        preds = raw_model.predict(X_dense)
        
        # Conformal wrapper
        coverage_count = 0
        set_sizes = []
        empty_sets = 0
        
        for i in range(len(y_final)):
            # Handle multi-class probabilities correctly
            prob_vec = {c: float(probs[i, c]) for c in range(probs.shape[1])}
            c_res = conformal.get_prediction_set(prob_vec)
            
            p_set = c_res["prediction_set"]
            # Check if true class is in prediction set
            labels = {0: "Low_Risk", 1: "Medium_Risk", 2: "High_Risk"}
            true_str = labels.get(int(y_final.iloc[i]), str(y_final.iloc[i]))
            
            if true_str in p_set:
                coverage_count += 1
            set_sizes.append(c_res["set_size"])
            if c_res.get("empty_set", False):
                empty_sets += 1
                
        base_a_acc = np.mean(preds == y_final)
        emp_coverage = coverage_count / len(y_final)
        avg_set_size = np.mean(set_sizes)
        
        print("\n--- RESULTS ---")
        print(f"Baseline A Accuracy (X_final_test): {base_a_acc:.2%}")
        print(f"Conformal Empirical Coverage (X_final_test): {emp_coverage:.2%} (Target: ~95%)")
        print(f"Conformal Average Set Size (X_final_test): {avg_set_size:.2f}")
        print(f"Empty Sets Generated (X_final_test): {empty_sets} / {len(y_final)}")
        
        sizes_arr = np.array(set_sizes)
        print(f"Sets of size 0 (if no fallback): {np.sum(sizes_arr == 0)} / {len(y_final)}")
        print(f"Sets of size 2+: {np.sum(sizes_arr > 1)} / {len(y_final)}")
        
    except Exception as e:
        print(f"Failed step 1: {e}")
        
    # 2. Evaluate on synthetic interference cases
    print("\nEvaluating on synthetic interference cases (Raw data)...")
    try:
        synth = pd.read_csv("data/processed/synthetic_interference_cases.csv")
        flags_caught = 0
        for i, row in synth.iterrows():
            flags = confounder.detect(row.to_dict())
            if len(flags) > 0:
                flags_caught += 1
                
        synth_recall = flags_caught / len(synth)
        print(f"Confounder Recall on Synthetic Cases: {synth_recall:.2%}")
    except Exception as e:
        print(f"Failed step 2: {e}")
        
    # 3. Evaluate on REAL discordant cases
    print("\nEvaluating on real discordant cases (Class R)...")
    try:
        real_conf = pd.read_csv("data/raw/real_confounders.csv", na_values='?')
        flags_caught_real = 0
        for i, row in real_conf.iterrows():
            flags = confounder.detect(row.to_dict())
            if len(flags) > 0:
                flags_caught_real += 1
                
        if len(real_conf) > 0:
            real_recall = flags_caught_real / len(real_conf)
            print(f"Confounder Recall on Real Discordant Cases (Class R): {real_recall:.2%} ({flags_caught_real}/{len(real_conf)})")
        else:
            print("No real discordant cases found.")
    except Exception as e:
        print(f"Failed step 3: {e}")

if __name__ == "__main__":
    main()
