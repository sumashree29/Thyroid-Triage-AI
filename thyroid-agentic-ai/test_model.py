"""
Test the ML model to see what it's actually predicting
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.risk_scoring import RiskScoringAgent
import pandas as pd

# Initialize agent
agent = RiskScoringAgent(
    model_path="models/risk_classifier.pkl",
    encoder_path="models/encoder.pkl"
)

# Test case 1: High TSH (hypothyroid)
test_data_1 = pd.DataFrame([{
    'age': 52,
    'sex': 'F',
    'tsh': 8.5,
    't3': 1.2,
    'tt4': 65,
    't4u': 0.85,
    'fti': 55
}])

print("=" * 60)
print("TEST 1: High TSH = 8.5 (Should be HIGH RISK ~70-80%)")
print("=" * 60)
result1 = agent.score_patient(test_data_1)
print(f"Risk Score: {result1.risk_score:.1%}")
print(f"Confidence: {result1.confidence:.1%}")
print(f"Risk Class: {result1.risk_class}")
print(f"Flags: {result1.uncertainty_flags}")
print()

# Test case 2: Normal TSH
test_data_2 = pd.DataFrame([{
    'age': 35,
    'sex': 'M',
    'tsh': 2.0,
    't3': 1.8,
    'tt4': 105,
    't4u': 0.95,
    'fti': 110
}])

print("=" * 60)
print("TEST 2: Normal TSH = 2.0 (Should be LOW RISK ~10-20%)")
print("=" * 60)
result2 = agent.score_patient(test_data_2)
print(f"Risk Score: {result2.risk_score:.1%}")
print(f"Confidence: {result2.confidence:.1%}")
print(f"Risk Class: {result2.risk_class}")
print(f"Flags: {result2.uncertainty_flags}")
print()

# Test case 3: Low TSH (hyperthyroid)
test_data_3 = pd.DataFrame([{
    'age': 28,
    'sex': 'F',
    'tsh': 0.15,
    't3': 2.5,
    'tt4': 140,
    't4u': 1.1,
    'fti': 127
}])

print("=" * 60)
print("TEST 3: Low TSH = 0.15 (Should be HIGH RISK ~75-90%)")
print("=" * 60)
result3 = agent.score_patient(test_data_3)
print(f"Risk Score: {result3.risk_score:.1%}")
print(f"Confidence: {result3.confidence:.1%}")
print(f"Risk Class: {result3.risk_class}")
print(f"Flags: {result3.uncertainty_flags}")
