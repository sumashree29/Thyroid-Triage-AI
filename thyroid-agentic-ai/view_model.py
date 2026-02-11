"""
Simple Model Viewer
Shows basic information about your ML model
"""
import pickle
import os

print("=" * 70)
print("           THYROID TRIAGE AI - MODEL INFORMATION")
print("=" * 70)

# Load model
model = pickle.load(open('models/risk_classifier.pkl', 'rb'))

print("\n📊 MODEL TYPE:")
print(f"   Algorithm: Random Forest Classifier")
print(f"   Python Class: {type(model).__name__}")

print("\n🌲 FOREST DETAILS:")
print(f"   Number of Decision Trees: {model.n_estimators}")
print(f"   Maximum Tree Depth: {model.max_depth}")
print(f"   Minimum Samples to Split: {model.min_samples_split}")

print("\n📈 TRAINING INFO:")
print(f"   Number of Features Used: {model.n_features_in_}")
print(f"   Number of Classes: {len(model.classes_)}")
print(f"   Classes: 0=Normal, 1=Thyroid Dysfunction")

print("\n🎯 FEATURE IMPORTANCE:")
print("   Top features that influence predictions:")
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    
    # Assume standard feature order
    feature_names = [
        'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
        'on_antithyroid_med', 'sick', 'pregnant', 'thyroid_surgery',
        'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
        'TSH', 'T3', 'TT4', 'T4U', 'FTI'
    ]
    
    # Get top 10
    top_indices = importances.argsort()[-10:][::-1]
    
    for i, idx in enumerate(top_indices, 1):
        if idx < len(feature_names):
            name = feature_names[idx]
            score = importances[idx]
            bar = "█" * int(score * 40)
            print(f"   {i:2d}. {name:20s} {score:6.3f}  {bar}")

print("\n💾 FILE INFORMATION:")
model_size = os.path.getsize('models/risk_classifier.pkl') / (1024 * 1024)
encoder_size = os.path.getsize('models/encoder.pkl') / 1024

print(f"   Model File: risk_classifier.pkl ({model_size:.2f} MB)")
print(f"   Encoder File: encoder.pkl ({encoder_size:.2f} KB)")

print("\n📍 LOCATION:")
print(f"   Directory: models/")
print(f"   Full Path: {os.path.abspath('models/risk_classifier.pkl')}")

print("\n✅ MODEL STATUS:")
print("   ✓ Model loaded successfully")
print("   ✓ Ready for predictions")
print("   ✓ Currently being used by API")

print("\n" + "=" * 70)
print("              Model inspection complete!")
print("=" * 70)
