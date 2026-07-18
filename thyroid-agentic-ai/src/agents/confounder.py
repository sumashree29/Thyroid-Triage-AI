"""
Agent 5: Confounder Agent
Rule-based screen for known thyroid-immunoassay interference patterns.
"""

class ConfounderAgent:
    """
    Rule-based screen for known thyroid-immunoassay interference patterns.
    Logic operationalizes: Favresse et al., "Interferences With Thyroid
    Function Immunoassays: Clinical Implications and Detection Algorithm",
    Endocrine Reviews, 2018.
    """

    TSH_LOW, TSH_HIGH = 0.45, 4.5
    FT4_LOW, FT4_HIGH = 70.0, 150.0       # FTI proxy
    T3_LOW, T3_HIGH = 0.8, 3.0

    def detect(self, patient: dict) -> list:
        flags = []
        
        # safely handle case insensitivity / missing values
        tsh = patient.get("tsh", patient.get("TSH", None))
        ft4 = patient.get("fti", patient.get("FTI", None))
        t3 = patient.get("t3", patient.get("T3", None))
        illness_context = patient.get("illness_flag", patient.get("sick", False))

        # We can only run these rules if we have TSH, FT4, and T3 (or at least TSH and FT4 for some rules)
        # If any essential values are missing, skip rules requiring them.
        
        # Rule 1 — biotin/assay interference
        if tsh is not None and ft4 is not None and t3 is not None:
            if self.TSH_LOW <= tsh <= self.TSH_HIGH and (ft4 > self.FT4_HIGH or t3 > self.T3_HIGH):
                flags.append({
                    "interference_type": "biotin_or_assay_interference",
                    "confidence": self._confidence(ft4, t3),
                    "recommended_follow_up": "Repeat FT4/T3 on non-biotin-based assay platform; confirm normal TSH is genuine.",
                })

        # Rule 2 — macro-TSH
        if tsh is not None and ft4 is not None:
            if self.TSH_HIGH < tsh <= 10.0 and self.FT4_LOW <= ft4 <= self.FT4_HIGH:
                flags.append({
                    "interference_type": "possible_macro_TSH",
                    "confidence": "medium",
                    "recommended_follow_up": "Consider PEG precipitation test to rule out macro-TSH before treating as hypothyroid.",
                })

        # Rule 3 — incoherent panel (heterophile antibody signature)
        if tsh is not None and ft4 is not None:
            if (tsh > self.TSH_HIGH and ft4 > self.FT4_HIGH) or (tsh < self.TSH_LOW and ft4 < self.FT4_LOW):
                flags.append({
                    "interference_type": "incoherent_TSH_FT4_pattern",
                    "confidence": "high",
                    "recommended_follow_up": "TSH and FT4 moving in the same direction is physiologically incoherent — repeat on alternate assay platform.",
                })

        # Rule 4 — non-thyroidal illness / sick euthyroid
        if tsh is not None and t3 is not None:
            if illness_context and t3 < self.T3_LOW and self.TSH_LOW <= tsh <= 10.0:
                flags.append({
                    "interference_type": "possible_non_thyroidal_illness",
                    "confidence": "medium",
                    "recommended_follow_up": "Consider deferring thyroid treatment until recovery from acute illness; recheck panel then.",
                })

        return flags

    def _confidence(self, ft4: float, t3: float) -> str:
        # simple magnitude-based bucketing
        deviation = max(ft4 - self.FT4_HIGH, t3 - self.T3_HIGH, 0)
        if deviation > 1.0:
            return "high"
        elif deviation > 0.3:
            return "medium"
        return "low"
