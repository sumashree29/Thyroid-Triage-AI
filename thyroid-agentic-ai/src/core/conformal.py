import pickle
from pathlib import Path

class ConformalWrapper:
    def __init__(self, threshold_path="models/conformal_threshold.pkl"):
        self.q_hat = 0.0
        self.alpha = 0.05
        self.calibration_size = 0
        
        path = Path(threshold_path)
        if not path.exists():
            path = Path("../models/conformal_threshold.pkl")
            
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.q_hat = data.get("q_hat", 0.0)
                self.alpha = data.get("alpha", 0.05)
                self.calibration_size = data.get("calibration_size", 0)
                
    def get_prediction_set(self, prob_vector: dict) -> dict:
        """
        prob_vector: {0: prob_low_risk, 1: prob_high_risk}
        Returns conformal prediction set and metadata.
        """
        if not prob_vector:
            return None
            
        prediction_set = []
        labels = {0: "Low_Risk", 1: "Medium_Risk", 2: "High_Risk"}
        
        point_prediction = None
        max_prob = -1
        
        for class_idx, prob in prob_vector.items():
            if prob > max_prob:
                max_prob = prob
                point_prediction = labels.get(class_idx, str(class_idx))
                
            # Include class y if p_hat(y|x) >= 1 - q_hat
            # equivalently: 1 - p_hat <= q_hat
            if prob >= (1.0 - self.q_hat):
                prediction_set.append(labels.get(class_idx, str(class_idx)))
                
        empty_set = False
        # Fallback if set is empty (rare, but mathematically guaranteed if q_hat < 0.5 in binary, or borderline in multi-class)
        if not prediction_set and point_prediction:
            prediction_set.append(point_prediction)
            empty_set = True
            
        return {
            "prediction_set": prediction_set,
            "coverage_level": 1.0 - self.alpha,
            "set_size": len(prediction_set),
            "point_prediction": point_prediction,
            "empty_set": empty_set
        }
