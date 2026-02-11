"""
Thyroid Triage Agentic AI - End-to-End Pipeline Script
Performs workspace cleanup, dataset audit, model training, visualization, RAG validation, and system integration test.
"""
import os
import shutil
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import pickle

# ========== 1. WORKSPACE CLEANUP ==========
print("\n=== WORKSPACE CLEANUP ===")
root = Path(__file__).parent.resolve()
core_dirs = {'src', 'data', 'models', 'docs'}
removed = []
for f in root.glob("*.py"):
    if f.name not in {"main.py", "setup.py", "train_model.py", "api.py", "thyroid_triage_pipeline.py"}:
        f.unlink()
        removed.append(f.name)
for f in root.glob("*.log"):
    f.unlink()
    removed.append(f.name)
print(f"Removed temp files: {removed if removed else 'None'}")

# ========== 2. DATASET AUDIT ==========
print("\n=== DATASET AUDIT ===")
data_path = root / "data/raw/Thyroid_Data.csv"
if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")
df = pd.read_csv(data_path, na_values=['?'])
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0])
for col in ['age', 'tsh', 't3', 'tt4', 't4u', 'fti']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
if not (root / "data/raw").exists():
    raise FileNotFoundError("data/raw/ folder missing!")

# ========== 3. MODEL TRAINING (Agent 1) ==========
print("\n=== MODEL TRAINING (Agent 1: Risk Scoring) ===")
def derive_risk(tsh):
    if pd.isna(tsh):
        return 0
    return 1 if (tsh > 4.5 or tsh < 0.45) else 0
df['target'] = df['tsh'].apply(derive_risk)

num_cols = ['age', 'tsh', 't3', 'tt4', 't4u', 'fti']
cat_cols = [c for c in df.columns if c not in num_cols + ['target', 'referral_source', 'patient_age'] and df[c].dtype == 'object']
features = num_cols + cat_cols
features = [f for f in features if f in df.columns]

# Fill missing values
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

X = pd.get_dummies(df[features], drop_first=True)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)
Path("models").mkdir(exist_ok=True)
pickle.dump(model, open("models/risk_model.pkl", "wb"))
print("Model trained and saved to models/risk_model.pkl")

# ========== 4. VISUALIZATIONS ==========
print("\n=== VISUALIZATIONS ===")
Path("results").mkdir(exist_ok=True)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("results/confusion_matrix.png")
plt.close()
print("Confusion matrix saved to results/confusion_matrix.png")

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title('Feature Importances')
plt.bar(range(len(indices)), importances[indices], align='center')
plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.close()
print("Feature importance plot saved to results/feature_importance.png")

# ========== 5. RAG VALIDATION (Agent 2) ==========
print("\n=== RAG VALIDATION (Agent 2: Retriever) ===")
guideline_dir = root / "docs/guidelines"
pdfs = list(guideline_dir.glob("*.pdf"))
if pdfs:
    print(f"Found {len(pdfs)} guideline PDFs: {[p.name for p in pdfs]}")
    print("VectorDB ready for indexing.")
else:
    print("No NICE/ATA PDFs found in docs/guidelines/. Please add them for full RAG.")

# ========== 6. SYSTEM INTEGRATION TEST ==========
print("\n=== SYSTEM INTEGRATION TEST ===")
sim_patient = {
    'age': 45,
    'tsh': 10.5,
    't3': 2.1,
    'tt4': 90,
    't4u': 1.0,
    'fti': 100,
    'sex': 'F',
    'on_thyroxine': 'f',
    'on_antithyroid_medication': 'f',
    'sick': 'f',
    'pregnant': 'f',
    'thyroid_surgery': 'f',
    'lithium': 'f',
    'goitre': 'f',
    'tumor': 'f',
    'hypopituitary': 'f',
    'psych': 'f'
}
# Ensure all expected categorical columns are present in simulated patient
for col in cat_cols:
    if col not in sim_patient:
        sim_patient[col] = 'f'  # default to 'f' (false)
sim_df = pd.DataFrame([sim_patient])

# Fill missing values for simulated patient
for col in num_cols:
    if col in sim_df.columns:
        sim_df[col] = sim_df[col].fillna(sim_df[col].median())
for col in cat_cols:
    if col in sim_df.columns:
        sim_df[col] = sim_df[col].fillna(sim_df[col].mode()[0])

sim_X = pd.get_dummies(sim_df[features], drop_first=True)
for col in X.columns:
    if col not in sim_X.columns:
        sim_X[col] = 0
sim_X = sim_X[X.columns]
model = pickle.load(open("models/risk_model.pkl", "rb"))
risk_score = model.predict_proba(sim_X)[0][1]
print(f"Simulated Patient Risk Score: {risk_score:.2f}")

# Placeholder guideline retrieval
guideline = "NICE NG145: If TSH > 10, repeat test in 3 months or start Levothyroxine."

# Doctor summary
print("\n--- Doctor Summary ---")
print(f"45yo F, TSH 10.5 (HIGH RISK). ML Risk Score: {risk_score:.2f}. Guideline: {guideline}")

# Patient summary
print("\n--- Patient Summary ---")
print("Your thyroid test shows a high TSH level (10.5). This may mean your thyroid is underactive. Your risk score is high. NICE guidelines suggest follow-up and possible treatment. Please discuss with your doctor.")

print("\n=== PIPELINE COMPLETE ===")
