"""
Agent 3: Reasoning Agent
Links model predictions with clinical evidence for interpretable reasoning.
Provides transparent explanations of clinical decisions.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import json


@dataclass
class ReasoningOutput:
    """Structured reasoning output."""
    risk_score: float
    confidence: float
    clinical_impression: str
    evidence_citations: List[str]
    key_findings: List[str]
    recommendations: List[str]
    uncertainty_notes: List[str]
    triage_category: str


class ReasonerAgent:
    """
    Performs clinical reasoning by combining model predictions with retrieved evidence.
    Generates transparent, interpretable explanations for risk assessments.
    """
    
    def __init__(self, reasoning_rules: Dict = None):
        """
        Initialize the reasoning agent.
        
        Args:
            reasoning_rules: Custom reasoning rules/logic
        """
        self.reasoning_rules = reasoning_rules or self._default_reasoning_rules()
    
    def _default_reasoning_rules(self) -> Dict:
        """Define default clinical reasoning rules."""
        return {
            'high_risk': {
                'threshold': 0.70,
                'actions': [
                    'Urgent endocrinologist referral',
                    'Start treatment within 2 weeks',
                    'Monitor TSH every 4-6 weeks',
                    'Check for complications'
                ]
            },
            'moderate_risk': {
                'threshold': 0.40,
                'actions': [
                    'Schedule endocrinology appointment',
                    'Monitor TSH in 4-6 weeks',
                    'Symptomatic management',
                    'Lifestyle modifications'
                ]
            },
            'low_risk': {
                'threshold': 0.0,
                'actions': [
                    'Routine screening (annual)',
                    'Lifestyle support',
                    'Patient education',
                    'No immediate treatment'
                ]
            }
        }
    
    def reason(
        self,
        risk_score: float,
        confidence: float,
        guidelines: List[Dict],
        patient_data: Dict,
        uncertainty_flags: List[str] = None,
        confounder_flags: List[Dict] = None,
        conformal_set: Dict = None
    ) -> ReasoningOutput:
        """
        Generate clinical reasoning combining model and evidence.
        
        Args:
            risk_score: ML model risk prediction (0-1)
            confidence: Model confidence score (0-1)
            guidelines: Retrieved clinical guidelines
            patient_data: Patient clinical parameters
            uncertainty_flags: Uncertainty flags from risk scorer
            confounder_flags: Flags from ConfounderAgent
            conformal_set: Output from ConformalWrapper
            
        Returns:
            ReasoningOutput with evidence-backed reasoning
        """
        uncertainty_flags = uncertainty_flags or []
        
        # Determine risk category
        triage_category = self._categorize_risk(risk_score)
        
        # Generate clinical impression
        clinical_impression = self._generate_impression(risk_score, patient_data, triage_category, confounder_flags)
        
        # Extract key findings from patient data
        key_findings = self._extract_findings(patient_data, risk_score)
        
        # Get recommended actions based on risk level
        recommendations = self.reasoning_rules.get(triage_category, {}).get('actions', [])
        
        # Format evidence citations
        evidence_citations = self._format_citations(guidelines)
        
        # Consolidate uncertainty notes
        uncertainty_notes = self._prepare_uncertainty_notes(uncertainty_flags, confidence, conformal_set)
        
        return ReasoningOutput(
            risk_score=risk_score,
            confidence=confidence,
            clinical_impression=clinical_impression,
            evidence_citations=evidence_citations,
            key_findings=key_findings,
            recommendations=recommendations,
            uncertainty_notes=uncertainty_notes,
            triage_category=triage_category
        )
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk into triage levels."""
        if risk_score >= 0.70:
            return 'high_risk'
        elif risk_score >= 0.40:
            return 'moderate_risk'
        else:
            return 'low_risk'
    
    def _generate_impression(self, risk_score: float, patient_data: Dict, category: str, confounder_flags: List[Dict] = None) -> str:
        """Generate clinical impression from data."""
        
        tsh = patient_data.get('tsh', None)
        age = patient_data.get('age', None)
        
        impressions = []
        
        # TSH-based impression
        if tsh is not None:
            if tsh > 4.5:
                impressions.append("Elevated TSH suggests hypothyroidism risk")
            elif tsh < 0.45:
                impressions.append("Suppressed TSH suggests hyperthyroidism risk")
            else:
                impressions.append("TSH within normal range")
        
        # Age-related considerations
        if age is not None:
            if age > 60:
                impressions.append("Advanced age increases thyroid disorder prevalence")
            elif age < 30:
                impressions.append("Younger age suggests possible Graves' disease")
        
        # Risk categorization
        if category == 'high_risk':
            impressions.append("⚠️ High risk identified - urgent evaluation recommended")
        elif category == 'moderate_risk':
            impressions.append("Moderate risk - close monitoring and possible treatment")
        else:
            impressions.append("Low risk - routine monitoring appropriate")
            
        if confounder_flags:
            for flag in confounder_flags:
                impressions.append(f"Note: {flag.get('recommended_follow_up', '')}")
        
        return ". ".join(impressions) + "."
    
    def _extract_findings(self, patient_data: Dict, risk_score: float) -> List[str]:
        """Extract key clinical findings from patient data."""
        findings = []
        
        # Thyroid function markers
        markers = {
            'tsh': ('TSH', (0.45, 4.5)),
            't3': ('T3', (60, 180)),
            'tt4': ('Total T4', (50, 150)),
            't4u': ('T4 Uptake', (0.24, 0.39)),
            'fti': ('Free T4 Index', (1.2, 4.9))
        }
        
        for marker_key, (marker_name, normal_range) in markers.items():
            value = patient_data.get(marker_key)
            if value is not None:
                low, high = normal_range
                if value < low:
                    findings.append(f"⬇️ {marker_name} LOW ({value:.2f}, normal: {low}-{high})")
                elif value > high:
                    findings.append(f"⬆️ {marker_name} HIGH ({value:.2f}, normal: {low}-{high})")
                else:
                    findings.append(f"✓ {marker_name} normal ({value:.2f})")
        
        # Demographics
        if 'sex' in patient_data:
            findings.append(f"Gender: {patient_data['sex']}")
        
        return findings
    
    def _format_citations(self, guidelines: List[Dict]) -> List[str]:
        """Format guidelines as citations."""
        citations = []
        
        for guide in guidelines[:5]:  # Top 5 citations
            source = guide.get('source', 'Unknown')
            category = guide.get('category', 'General')
            content_preview = guide.get('content', '')[:80]
            
            citation = f"[{source}] {category}: {content_preview}..."
            citations.append(citation)
        
        return citations
    
    def _prepare_uncertainty_notes(self, uncertainty_flags: List[str], confidence: float, conformal_set: Dict = None) -> List[str]:
        """Prepare notes about prediction uncertainty."""
        notes = []
        
        if confidence < 0.60:
            notes.append("⚠️ Low model confidence - clinical judgment essential")
        
        notes.extend(uncertainty_flags)
        
        if conformal_set and conformal_set.get("set_size", 1) > 1:
            notes.append(f"Model confidence is limited for this patient — "
                         f"risk category could be any of {conformal_set['prediction_set']} "
                         f"at {conformal_set['coverage_level']*100:.0f}% confidence.")
        
        if not notes:
            notes.append("✓ Reasonable confidence in prediction")
        
        return notes
    
    def explain_prediction(self, reasoning: ReasoningOutput) -> str:
        """
        Create comprehensive explanation of prediction.
        
        Args:
            reasoning: Output from reason()
            
        Returns:
            Detailed explanation string
        """
        explanation = f"""
=== CLINICAL REASONING SUMMARY ===

IMPRESSION:
{reasoning.clinical_impression}

KEY FINDINGS:
{chr(10).join(f"  • {f}" for f in reasoning.key_findings)}

EVIDENCE BASIS:
{chr(10).join(f"  • {c}" for c in reasoning.evidence_citations)}

RECOMMENDED ACTIONS:
{chr(10).join(f"  {i+1}. {r}" for i, r in enumerate(reasoning.recommendations))}

CONFIDENCE ASSESSMENT:
Model Confidence: {reasoning.confidence:.1%}
{chr(10).join(f"  • {note}" for note in reasoning.uncertainty_notes)}

TRIAGE LEVEL: {reasoning.triage_category.upper().replace('_', ' ')}
RISK SCORE: {reasoning.risk_score:.1%}
"""
        return explanation.strip()
