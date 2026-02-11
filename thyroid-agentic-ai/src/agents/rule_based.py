"""
Simple Rule-Based Thyroid Risk Calculator
Used when ML model has low confidence - provides interpretable fallback.
"""
import pandas as pd
from typing import Tuple


class RuleBasedRiskCalculator:
    """
    TSH-based risk calculator using clinical guidelines.
    Normal TSH: 0.45 - 4.5 mIU/L
    """
    
    # Clinical thresholds
    TSH_NORMAL_MIN = 0.45
    TSH_NORMAL_MAX = 4.5
    
    TSH_SEVERE_LOW = 0.1    # Severe hyperthyroidism
    TSH_SEVERE_HIGH = 10.0  # Severe hypothyroidism
    
    TSH_MODERATE_LOW = 0.3
    TSH_MODERATE_HIGH = 7.0
    
    @classmethod
    def calculate_risk(cls, tsh: float) -> Tuple[float, str]:
        """
        Calculate risk score and explanation based on TSH level.
        
        Returns:
            (risk_score, explanation)
        """
        if pd.isna(tsh) or tsh is None:
            return 0.5, "TSH not available - cannot assess"
        
        # Severe abnormalities
        if tsh < cls.TSH_SEVERE_LOW:
            risk = 0.95
            explanation = f"Severe hyperthyroidism (TSH {tsh:.2f} << {cls.TSH_SEVERE_LOW})"
        elif tsh > cls.TSH_SEVERE_HIGH:
            risk = 0.90
            explanation = f"Severe hypothyroidism (TSH {tsh:.2f} >> {cls.TSH_SEVERE_HIGH})"
        
        # Moderate abnormalities  
        elif tsh < cls.TSH_NORMAL_MIN:
            if tsh < cls.TSH_MODERATE_LOW:
                risk = 0.75
                explanation = f"Moderate hyperthyroidism (TSH {tsh:.2f} < {cls.TSH_NORMAL_MIN})"
            else:
                risk = 0.60
                explanation = f"Mild hyperthyroidism (TSH {tsh:.2f} slightly low)"
        
        elif tsh > cls.TSH_NORMAL_MAX:
            if tsh > cls.TSH_MODERATE_HIGH:
                risk = 0.80
                explanation = f"Moderate hypothyroidism (TSH {tsh:.2f} > {cls.TSH_MODERATE_HIGH})"
            else:
                risk = 0.65
                explanation = f"Mild hypothyroidism (TSH {tsh:.2f} slightly high)"
        
        # Normal range
        else:
            # Even within range, calculate relative position
            mid_point = (cls.TSH_NORMAL_MIN + cls.TSH_NORMAL_MAX) / 2
            if tsh < mid_point:
                risk = 0.15
            else:
                risk = 0.20
            explanation = f"Normal thyroid function (TSH {tsh:.2f} in range {cls.TSH_NORMAL_MIN}-{cls.TSH_NORMAL_MAX})"
        
        return risk, explanation
