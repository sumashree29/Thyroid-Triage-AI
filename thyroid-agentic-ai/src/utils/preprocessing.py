"""
Data Preprocessing Pipeline
Handles cleaning, transformation, and validation of patient data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


class PreprocessingPipeline:
    """
    Standardizes patient input data for model inference.
    """
    
    def __init__(self, encoder=None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            encoder: Fitted sklearn ColumnTransformer or similar
        """
        self.encoder = encoder
    
    def clean_patient_data(self, raw_data: Dict) -> pd.DataFrame:
        """
        Clean raw patient input.
        
        Args:
            raw_data: Raw patient parameters
            
        Returns:
            Cleaned DataFrame ready for model
        """
        df = pd.DataFrame([raw_data])
        
        # Handle missing values
        df = df.fillna(method='ffill')
        
        # Remove invalid characters (like '?')
        df = df.replace('?', np.nan)
        
        return df
    
    def validate_ranges(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate patient parameters are within reasonable ranges.
        
        Args:
            data: Patient data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validation rules
        valid_ranges = {
            'age': (0, 130),
            'tsh': (0, 100),
            't3': (0, 300),
            'tt4': (10, 200),
            't4u': (0.4, 1.3),
            'fti': (0, 300)
        }
        
        for col, (min_val, max_val) in valid_ranges.items():
            if col in data.columns:
                if not ((data[col] >= min_val) & (data[col] <= max_val)).all():
                    return False, f"{col} out of range [{min_val}, {max_val}]"
        
        return True, "Valid"
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Apply fitted encoder to transform data.
        
        Args:
            data: Cleaned patient data
            
        Returns:
            Transformed feature array
        """
        if self.encoder is None:
            raise ValueError("Encoder not fitted")
        
        return self.encoder.transform(data)
