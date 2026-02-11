"""
Agent 4: Summary Agent
Generates audience-specific outputs (doctors vs patients) with appropriate detail levels.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class SummaryOutput:
    """Output structure for summarized results."""
    doctor_report: str
    patient_summary: str
    triage_level: str
    next_steps: str


class SummarizerAgent:
    """
    Tailors triage output for specific audiences with appropriate detail level.
    Balances clinical precision with accessibility.
    """
    
    def __init__(self):
        """Initialize the summarizer with output templates."""
        self.triage_levels = {
            'high_risk': 'URGENT',
            'moderate_risk': 'HIGH PRIORITY', 
            'low_risk': 'ROUTINE'
        }
        
        self.triage_colors = {
            'URGENT': '🔴',
            'HIGH PRIORITY': '🟠',
            'ROUTINE': '🟢'
        }
    
    def summarize(self, reasoning: Dict) -> SummaryOutput:
        """
        Generate doctor and patient summaries from reasoning output.
        
        Args:
            reasoning: Output from Reasoner Agent
            
        Returns:
            SummaryOutput with both versions
        """
        triage_level = self.triage_levels.get(reasoning['triage_category'], 'UNKNOWN')
        
        doctor_report = self.generate_doctor_report(reasoning)
        patient_summary = self.generate_patient_summary(reasoning)
        next_steps = self.generate_next_steps(reasoning)
        
        return SummaryOutput(
            doctor_report=doctor_report,
            patient_summary=patient_summary,
            triage_level=triage_level,
            next_steps=next_steps
        )
    
    def generate_doctor_report(self, reasoning: Dict) -> str:
        """
        Generate detailed medical report for healthcare providers.
        
        Args:
            reasoning: Reasoning output with clinical data
            
        Returns:
            Structured clinical report
        """
        triage_level = self.triage_levels.get(reasoning['triage_category'], 'UNKNOWN')
        color = self.triage_colors.get(triage_level, '')
        
        report = f"""{color} THYROID TRIAGE CLINICAL REPORT

╔════════════════════════════════════════════════════════════════╗
║ RISK ASSESSMENT                                                ║
╚════════════════════════════════════════════════════════════════╝

Triage Category: {triage_level}
Risk Score: {reasoning['risk_score']:.1%}
Model Confidence: {reasoning['confidence']:.1%}

╔════════════════════════════════════════════════════════════════╗
║ CLINICAL IMPRESSION                                            ║
╚════════════════════════════════════════════════════════════════╝

{reasoning['clinical_impression']}

╔════════════════════════════════════════════════════════════════╗
║ KEY FINDINGS                                                   ║
╚════════════════════════════════════════════════════════════════╝

"""
        for finding in reasoning['key_findings']:
            report += f"  {finding}\n"
        
        report += f"""
╔════════════════════════════════════════════════════════════════╗
║ EVIDENCE-BASED RECOMMENDATIONS                                ║
╚════════════════════════════════════════════════════════════════╝

"""
        for i, rec in enumerate(reasoning['recommendations'], 1):
            report += f"  {i}. {rec}\n"
        
        report += f"""
╔════════════════════════════════════════════════════════════════╗
║ CLINICAL EVIDENCE & CITATIONS                                 ║
╚════════════════════════════════════════════════════════════════╝

"""
        for citation in reasoning['evidence_citations']:
            report += f"  • {citation}\n"
        
        report += f"""
╔════════════════════════════════════════════════════════════════╗
║ UNCERTAINTY & LIMITATIONS                                      ║
╚════════════════════════════════════════════════════════════════╝

"""
        if reasoning['uncertainty_notes']:
            for note in reasoning['uncertainty_notes']:
                report += f"  ⚠️  {note}\n"
        else:
            report += "  ✓ Assessment confidence adequate for clinical decision-making\n"
        
        report += """
╔════════════════════════════════════════════════════════════════╗
║ DISCLAIMER                                                     ║
╚════════════════════════════════════════════════════════════════╝

This assessment is for CLINICAL DECISION SUPPORT ONLY.
It is NOT a diagnosis and should NOT replace professional judgment.
All recommendations must be reviewed and validated by qualified 
healthcare providers in accordance with institutional protocols.
Patient safety and clinical autonomy are paramount.

"""
        return report
    
    def generate_patient_summary(self, reasoning: Dict) -> str:
        """
        Generate patient-friendly summary with clear next steps.
        
        Args:
            reasoning: Reasoning output
            
        Returns:
            Patient-friendly summary
        """
        triage_level = self.triage_levels.get(reasoning['triage_category'], 'UNKNOWN')
        color = self.triage_colors.get(triage_level, '')
        
        # Simplify risk assessment for patients
        risk_explanation = {
            'high_risk': 'Your results suggest that your thyroid may not be working properly and needs prompt medical attention.',
            'moderate_risk': 'Your results show some signs of thyroid concern that need follow-up with your doctor.',
            'low_risk': 'Your thyroid appears to be functioning normally. Routine check-ups are still recommended.'
        }
        
        explanation = risk_explanation.get(reasoning['triage_category'], 'Please discuss results with your doctor.')
        
        summary = f"""{color} YOUR THYROID HEALTH SUMMARY

Hello! Here's what we found in your thyroid screening:

═══════════════════════════════════════════════════════════════
WHAT THIS MEANS FOR YOU

{explanation}

Priority Level: {triage_level}

═══════════════════════════════════════════════════════════════
YOUR NEXT STEPS

"""
        # Provide simple next steps
        steps_map = {
            'high_risk': [
                '1. Contact your doctor right away (this week)',
                '2. Schedule an appointment with an endocrinologist if recommended',
                '3. Bring these results to your appointment',
                '4. Be ready to discuss symptoms (fatigue, weight changes, etc.)',
                '5. Ask about blood tests and treatment options'
            ],
            'moderate_risk': [
                '1. Schedule a doctor visit within 1-2 weeks',
                '2. Keep track of any symptoms you notice',
                '3. Bring these results to your appointment',
                '4. Ask about follow-up testing in 4-6 weeks',
                '5. Discuss lifestyle changes that may help'
            ],
            'low_risk': [
                '1. No urgent action needed',
                '2. Schedule routine check-up with your doctor',
                '3. Get your thyroid checked again in 1-2 years',
                '4. Maintain a healthy lifestyle',
                '5. Report any new symptoms to your doctor'
            ]
        }
        
        steps = steps_map.get(reasoning['triage_category'], ['Contact your healthcare provider'])
        for step in steps:
            summary += f"{step}\n"
        
        summary += """
═══════════════════════════════════════════════════════════════
COMMON QUESTIONS

Q: Does this mean I have a thyroid problem?
A: Not necessarily. This is a screening tool, not a diagnosis. Your 
   doctor will do more tests to be sure.

Q: What should I tell my doctor?
A: Show them these results and discuss any symptoms you have, like
   tiredness, weight changes, or feeling hot/cold easily.

Q: Do I need medication?
A: That depends on what your doctor finds. Some people need medicine,
   others just need monitoring.

═══════════════════════════════════════════════════════════════
IMPORTANT

This is not a medical diagnosis. Please discuss these results with
your doctor. They know your full health history and can make the
best decisions for your care.

If you have urgent concerns, contact your doctor or seek immediate
medical attention.

═══════════════════════════════════════════════════════════════
"""
        return summary
    
    def generate_next_steps(self, reasoning: Dict) -> str:
        """
        Generate action steps from triage decision.
        
        Args:
            reasoning: Reasoning output
            
        Returns:
            Action steps string
        """
        category = reasoning['triage_category']
        
        timeframes = {
            'high_risk': 'Within 24-48 hours',
            'moderate_risk': 'Within 1-2 weeks',
            'low_risk': 'Within 4-6 weeks'
        }
        
        actions = {
            'high_risk': 'Urgent endocrinology referral, initiate treatment, frequent monitoring',
            'moderate_risk': 'Schedule endocrinology consultation, recheck TSH in 4-6 weeks',
            'low_risk': 'Routine follow-up, annual TSH screening, patient education'
        }
        
        timeframe = timeframes.get(category, 'As soon as possible')
        action = actions.get(category, 'Contact healthcare provider')
        
        return f"Timeline: {timeframe}\nAction: {action}"
    
    def generate_triage_category(self, risk_score: float) -> str:
        """
        Assign triage category (URGENT, HIGH_PRIORITY, ROUTINE).
        
        Args:
            risk_score: Risk score from model
            
        Returns:
            Triage category string
        """
        if risk_score >= 0.70:
            return "URGENT"
        elif risk_score >= 0.40:
            return "HIGH_PRIORITY"
        else:
            return "ROUTINE"
