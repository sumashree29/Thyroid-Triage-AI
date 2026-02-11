"""
Model Inspection Script
View details about your trained Random Forest model
"""
import pickle
import pandas as pd
import numpy as np

# Load the model
print("=" * 60)
print("LOADING TRAINED MODEL")
print("=" * 60)

model = pickle.load(open('models/risk_classifier.pkl', 'rb'))
encoder = pickle.load(open('models/encoder.pkl', 'rb'))

# Model Information
print("\n1. MODEL TYPE:")
print(f"   {type(model).__name__}")

print("\n2. MODEL PARAMETERS:")
print(f"   Number of Trees: {model.n_estimators}")
print(f"   Max Depth: {model.max_depth}")
print(f"   Min Samples Split: {model.min_samples_split}")
print(f"   Number of Features: {model.n_features_in_}")

print("\n3. CLASSES:")
print(f"   Class 0: Normal thyroid (low risk)")
print(f"   Class 1: Thyroid dysfunction (high risk)")

# Feature Importance
print("\n4. FEATURE IMPORTANCE:")
print("   Which features matter most for predictions:")

# Get feature names from encoder
try:
    feature_names = encoder.get_feature_names_out()
except:
    feature_names = ['age', 'sex', 'tsh', 't3', 'tt4', 't4u', 'fti']

if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_importance = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (feature, importance) in enumerate(feature_importance[:10], 1):
        bar = "█" * int(importance * 50)
        print(f"   {i}. {feature:15s} {importance:.4f} {bar}")

# Test the model
print("\n5. QUICK TEST:")
test_data = pd.DataFrame([{
    'age': 52,
    'sex': 'F',
    'tsh': 8.5,
    't3': 1.2,
    'tt4': 65,
    't4u': 0.85,
    'fti': 55
}])

print("   Input: TSH=8.5 (hypothyroidism)")
prediction = model.predict(encoder.transform(test_data))[0]
probabilities = model.predict_proba(encoder.transform(test_data))[0]

print(f"   Prediction: Class {prediction}")
print(f"   Probability of Normal: {probabilities[0]:.1%}")
print(f"   Probability of Dysfunction: {probabilities[1]:.1%}")
print(f"   Risk Score: {probabilities[1]:.1%}")

print("\n6. MODEL FILE SIZE:")
import os
size_mb = os.path.getsize('models/risk_classifier.pkl') / (1024 * 1024)
print(f"   {size_mb:.2f} MB")

print("\n" + "=" * 60)
print("MODEL INSPECTION COMPLETE")
print("=" * 60)
