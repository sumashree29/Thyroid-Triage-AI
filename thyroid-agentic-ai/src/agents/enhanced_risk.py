"""Enhanced Multi-Hormone Risk Calculator with Pattern Recognition"""
import pandas as pd
import numpy as np
from typing import Tuple, List


class EnhancedRiskCalculator:
    """Comprehensive thyroid risk assessment using pattern recognition."""
    
    TSH_NORMAL_MIN = 0.45
    TSH_NORMAL_MAX = 4.5
    T3_NORMAL_MIN = 0.8
    T3_NORMAL_MAX = 2.0
    TT4_NORMAL_MIN = 70
    TT4_NORMAL_MAX = 150
    FTI_NORMAL_MIN = 70
    FTI_NORMAL_MAX = 150
    
    @classmethod
    def calculate_comprehensive_risk(cls, patient_data: dict) -> Tuple[float, List[str]]:
        """Calculate risk using pattern recognition."""
        tsh = patient_data.get('tsh')
        t3 = patient_data.get('t3')
        tt4 = patient_data.get('tt4')
        fti = patient_data.get('fti')
        
        explanations = []
        
        if tsh is not None and not pd.isna(tsh):
            if tsh > cls.TSH_NORMAL_MAX:
                base_risk = 0.70 if tsh < 10 else 0.90
                explanations.append(f"TSH elevated ({tsh:.2f}) - hypothyroid pattern")
                pattern_score = base_risk
                
                if tt4 is not None and not pd.isna(tt4):
                    if tt4 < 80:
                        pattern_score = min(0.95, pattern_score + 0.10)
                        explanations.append(f"T4 low ({tt4:.1f}) - confirms hypothyroidism")
                    else:
                        explanations.append(f"T4 normal ({tt4:.1f})")
                
                if fti is not None and not pd.isna(fti):
                    if fti < 70:
                        pattern_score = min(0.95, pattern_score + 0.08)
                        explanations.append(f"FTI low ({fti:.1f}) - supports hypothyroidism")
                    else:
                        explanations.append(f"FTI normal ({fti:.1f})")
                
                if t3 is not None and not pd.isna(t3):
                    if t3 < cls.T3_NORMAL_MIN:
                        pattern_score = min(0.95, pattern_score + 0.05)
                        explanations.append(f"T3 low ({t3:.2f}) - confirms hypothyroidism")
                    else:
                        explanations.append(f"T3 normal ({t3:.2f}) - early stage")
                
                return pattern_score, explanations
            
            elif tsh < cls.TSH_NORMAL_MIN:
                base_risk = 0.70 if tsh > 0.1 else 0.95
                explanations.append(f"TSH suppressed ({tsh:.2f}) - hyperthyroid pattern")
                pattern_score = base_risk
                
                if tt4 is not None and not pd.isna(tt4):
                    if tt4 > 130:
                        pattern_score = min(0.95, pattern_score + 0.10)
                        explanations.append(f"T4 elevated ({tt4:.1f}) - confirms hyperthyroidism")
                    else:
                        explanations.append(f"T4 normal ({tt4:.1f})")
                
                if fti is not None and not pd.isna(fti):
                    if fti > 130:
                        pattern_score = min(0.95, pattern_score + 0.08)
                        explanations.append(f"FTI elevated ({fti:.1f}) - supports hyperthyroidism")
                    else:
                        explanations.append(f"FTI normal ({fti:.1f})")
                
                if t3 is not None and not pd.isna(t3):
                    if t3 > cls.T3_NORMAL_MAX:
                        pattern_score = min(0.95, pattern_score + 0.05)
                        explanations.append(f"T3 elevated ({t3:.2f}) - confirms hyperthyroidism")
                    else:
                        explanations.append(f"T3 normal ({t3:.2f})")
                
                return pattern_score, explanations
            
            else:
                risk = 0.15
                explanations.append(f"TSH normal ({tsh:.2f}) - euthyroid (normal)")
                
                issues = 0
                if tt4 is not None and not pd.isna(tt4):
                    if tt4 < cls.TT4_NORMAL_MIN or tt4 > cls.TT4_NORMAL_MAX:
                        issues += 1
                        risk += 0.15
                        explanations.append(f"T4 abnormal ({tt4:.1f}) - subclinical dysfunction")
                
                if t3 is not None and not pd.isna(t3):
                    if t3 < cls.T3_NORMAL_MIN or t3 > cls.T3_NORMAL_MAX:
                        issues += 1
                        risk += 0.10  
                        explanations.append(f"T3 abnormal ({t3:.2f}) - subclinical dysfunction")
                
                return min(0.50, risk), explanations
        
        else:
            risk_factors = []
            
            if t3 is not None and not pd.isna(t3):
                if t3 < cls.T3_NORMAL_MIN or t3 > cls.T3_NORMAL_MAX:
                    risk_factors.append(0.60)
                    explanations.append(f"T3 abnormal ({t3:.2f})")
                else:
                    risk_factors.append(0.10)
            
            if tt4 is not None and not pd.isna(tt4):
                if tt4 < cls.TT4_NORMAL_MIN or tt4 > cls.TT4_NORMAL_MAX:
                    risk_factors.append(0.65)
                    explanations.append(f"T4 abnormal ({tt4:.1f})")
                else:
                    risk_factors.append(0.10)
            
            if fti is not None and not pd.isna(fti):
                if fti < cls.FTI_NORMAL_MIN or fti > cls.FTI_NORMAL_MAX:
                    risk_factors.append(0.55)
                    explanations.append(f"FTI abnormal ({fti:.1f})")
                else:
                    risk_factors.append(0.10)
            
            if risk_factors:
                return np.mean(risk_factors), explanations
            else:
                return 0.50, ["Insufficient hormone data"]
