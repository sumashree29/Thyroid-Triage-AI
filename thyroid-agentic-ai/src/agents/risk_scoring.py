"""
Agent 1: Risk Scoring Agent
Performs model inference and confidence scoring for thyroid risk assessment.
Handles uncertainty and missing inputs.
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
import sys

# Import enhanced multi-hormone fallback
sys.path.insert(0, str(Path(__file__).parent))
from enhanced_risk import EnhancedRiskCalculator


@dataclass
class RiskScore:
    """Output structure for risk scoring."""
    risk_score: float  # 0-1
    risk_class: int  # 0 (low) or 1 (high)
    confidence: float  # Model confidence 0-1
    uncertainty_flags: List[str]  # Issues encountered
    is_confident: bool  # True if confidence > threshold


class RiskScoringAgent:
    """
    Loads a pre-trained ML classifier and performs risk scoring on patient data.
    Flags uncertainty, missing values, and out-of-range inputs.
    """
    
    CONFIDENCE_THRESHOLD = 0.70  # Flag low confidence predictions
    MISSING_VALUE_THRESHOLD = 0.3  # Flag if >30% missing
    
    def __init__(self, model_path: str, encoder_path: str, metadata_path: str = None):
        """
        Initialize the risk scoring agent with trained model.
        
        Args:
            model_path: Path to the trained ML model (.pkl)
            encoder_path: Path to the feature encoder (.pkl)
            metadata_path: Path to model metadata (optional)
        """
        try:
            self.model = pickle.load(open(model_path, 'rb'))
            self.encoder = pickle.load(open(encoder_path, 'rb'))
            print(f"✓ Loaded model: {type(self.model).__name__}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model artifacts not found: {e}")
        
        # Load metadata if available
        self.metadata = {}
        if metadata_path:
            try:
                self.metadata = pickle.load(open(metadata_path, 'rb'))
            except FileNotFoundError:
                pass
    
    def validate_input(self, patient_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate patient input data.
        
        Args:
            patient_data: Raw patient features
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Check for missing values
        missing_pct = patient_data.isnull().sum().sum() / patient_data.size
        if missing_pct > self.MISSING_VALUE_THRESHOLD:
            warnings.append(f"High missing data: {missing_pct*100:.1f}% missing")
        
        # Check feature ranges (basic validation)
        numeric_features = self.metadata.get('numeric_features', [])
        expected_ranges = {
            'age': (0, 150),
            'tsh': (0, 100),
            't3': (0, 300),
            'tt4': (10, 200),
            't4u': (0.4, 1.3),
            'fti': (0, 300)
        }
        
        for feature, (min_val, max_val) in expected_ranges.items():
            if feature in patient_data.columns:
                values = patient_data[feature].dropna()
                if len(values) > 0:
                    if (values < min_val).any() or (values > max_val).any():
                        warnings.append(f"{feature} out of typical range [{min_val}, {max_val}]")
        
        is_valid = missing_pct <= 1.0  # Allow some missing if imputable
        return is_valid, warnings
    
    def score_patient(self, patient_data: pd.DataFrame) -> RiskScore:
        """
        Score a patient's thyroid risk with full uncertainty quantification.
        
        Args:
            patient_data: Preprocessed patient features (DataFrame or dict-like)
            
        Returns:
            RiskScore object with risk assessment and confidence
        """
        uncertainty_flags = []
        
        # Ensure DataFrame format
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])
        elif not isinstance(patient_data, pd.DataFrame):
            patient_data = pd.DataFrame([patient_data])

        # Auto-fill missing columns (especially metadata flags)
        patient_data = self._fill_missing_features(patient_data)
        
        # Validate input
        is_valid, warnings = self.validate_input(patient_data)
        uncertainty_flags.extend(warnings)
        
        if not is_valid:
            uncertainty_flags.append("Input validation failed - predictions may be unreliable")
        
        try:
            # Transform features using fitted encoder
            patient_transformed = self.encoder.transform(patient_data)
            
            # Get prediction and probabilities
            prediction = self.model.predict(patient_transformed)[0]
            probabilities = self.model.predict_proba(patient_transformed)[0]
            
            # Risk score is probability of high risk (class 1)
            risk_score = probabilities[1]
            confidence = probabilities.max()
            
            # If ML confidence is too low, use enhanced clinical calculator instead
            if confidence < self.CONFIDENCE_THRESHOLD:
                uncertainty_flags.append(f"ML confidence too low ({confidence:.2f})")
                uncertainty_flags.append("Using enhanced clinical assessment")
                
                # Extract patient data as dict
                if isinstance(patient_data, pd.DataFrame):
                    patient_dict = patient_data.iloc[0].to_dict()
                else:
                    patient_dict = patient_data
                
                # Use comprehensive hormone-based risk
                rule_risk, rule_explanations = EnhancedRiskCalculator.calculate_comprehensive_risk(patient_dict)
                uncertainty_flags.extend(rule_explanations)
                
                return RiskScore(
                    risk_score=float(rule_risk),
                    risk_class=int(rule_risk > 0.5),
                    confidence=0.75,  # Clinical guidelines are reliable
                    uncertainty_flags=uncertainty_flags,
                    is_confident=True
                )
            
            # ML confidence is good - use ML prediction
            is_confident = is_valid
            
            return RiskScore(
                risk_score=float(risk_score),
                risk_class=int(prediction),
                confidence=float(confidence),
                uncertainty_flags=uncertainty_flags,
                is_confident=is_confident
            )
            
        except Exception as e:
            # ML failed - use enhanced multi-hormone fallback
            uncertainty_flags.append(f"ML error: {str(e)}")
            uncertainty_flags.append("Using clinical guideline-based assessment")
            
            # Extract patient data as dict
            if isinstance(patient_data, pd.DataFrame):
                patient_dict = patient_data.iloc[0].to_dict()
            else:
                patient_dict = patient_data
            
            # Calculate risk using all available hormones
            rule_risk, rule_explanations = EnhancedRiskCalculator.calculate_comprehensive_risk(patient_dict)
            uncertainty_flags.extend(rule_explanations)
            
            return RiskScore(
                risk_score=float(rule_risk),
                risk_class=int(rule_risk > 0.5),
                confidence=0.75,  # Clinical guidelines are reliable
                uncertainty_flags=uncertainty_flags,
                is_confident=True
            )

    def _fill_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Auto-populate missing columns expected by the model.
        Sets '_measured' flags based on value presence.
        """
        df = df.copy()
        
        # 1. Handle measurement flags
        # Map value column -> flag column
        measurements = {
            'tsh': 'tsh_measured',
            't3': 't3_measured',
            'tt4': 'tt4_measured',
            't4u': 't4u_measured',
            'fti': 'fti_measured',
            'tbg': 'tbg_measured'
        }
        
        for val_col, flag_col in measurements.items():
            if val_col in df.columns and flag_col not in df.columns:
                # If value is present and not NaN, measured=1, else 0
                df[flag_col] = df[val_col].notna().astype(int)
            elif flag_col not in df.columns:
                # Neither value nor flag present -> 0
                df[flag_col] = 0
                
        # 2. Handle binary boolean flags (default to 0 if missing)
        bool_flags = [
            'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
            'sick', 'pregnant', 'thyroid_surgery', 'i131_treatment',
            'query_hypothyroid', 'query_hyperthyroid', 'lithium',
            'goitre', 'tumor', 'hypopituitary', 'psych'
        ]
        
        for col in bool_flags:
            if col not in df.columns:
                df[col] = 0
                
        # 3. Handle missing numeric features that the encoder expects
        numeric_features = ['age', 'tsh', 't3', 'tt4', 't4u', 'fti', 'tbg']
        for col in numeric_features:
            if col not in df.columns:
                df[col] = np.nan
                
        return df

    def explain_risk(self, risk_score: RiskScore) -> str:
        """
        Generate human-readable explanation of risk score.
        
        Args:
            risk_score: RiskScore object
            
        Returns:
            Explanation string
        """
        if not risk_score.is_confident:
            return "⚠️ UNCERTAIN: Insufficient data or low model confidence. Clinical review recommended."
        
        if risk_score.risk_class == 1:
            return f"🔴 HIGH RISK: Model predicts thyroid abnormality (Score: {risk_score.risk_score:.2%}, Confidence: {risk_score.confidence:.2%})"
        else:
            return f"🟢 LOW RISK: Model indicates normal thyroid function (Score: {risk_score.risk_score:.2%}, Confidence: {risk_score.confidence:.2%})"
