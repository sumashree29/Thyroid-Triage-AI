"""
Main Entry Point for Thyroid Triage AI System
Complete agentic AI system for clinical decision support.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.workflow import TriageWorkflow, TriageInput, TriageOutput


# ============================================================
# ETHICAL DISCLAIMER
# ============================================================
ETHICAL_DISCLAIMER = """
╔════════════════════════════════════════════════════════════════╗
║        THYROID TRIAGE AI - ETHICAL DISCLAIMER                 ║
╚════════════════════════════════════════════════════════════════╝

⚠️  CRITICAL NOTICE ⚠️

This system provides CLINICAL DECISION SUPPORT ONLY.
It is NOT a medical diagnosis and NOT a replacement for 
professional clinical judgment.

INTENDED USE:
✓ Aid clinicians in triaging thyroid cases
✓ Provide evidence-based recommendations
✓ Generate explanations and educational material
✓ Support (not replace) clinical decision-making

NOT INTENDED FOR:
✗ Autonomous patient diagnosis
✗ Replacing healthcare provider evaluation
✗ Use by non-medical personnel
✗ Medical diagnosis without clinical review

LIMITATIONS:
• Predictions based on training data patterns
• May be inaccurate if inputs are missing/incorrect
• Cannot account for all clinical complexity
• Uncertainty quantified but not absolute
• Biases in training data may be present

SAFETY REQUIREMENTS:
1. All outputs must be reviewed by qualified healthcare provider
2. Decisions must follow institutional protocols
3. Patient safety is paramount
4. Clinical judgment supersedes AI recommendations
5. Document use in patient record with full transparency

RESPONSIBILITIES:
• Healthcare Organization: Responsible for clinical outcomes
• Clinicians: Must validate all recommendations
• Patients: Right to understand and refuse recommendations
• System Developer: Continuous monitoring for errors/bias

DATA PRIVACY:
• Patient information must be protected (HIPAA/GDPR)
• No data retained longer than necessary
• Audit trails maintained for accountability
• Informed consent required before use

DISCLAIMER OF LIABILITY:
This system is provided "as-is". Users assume all responsibility
for clinical decisions made with this system's assistance.

FOR QUESTIONS OR ISSUES:
Report errors, biases, or unexpected results immediately.
System is designed for transparency and continuous improvement.

═══════════════════════════════════════════════════════════════
By using this system, you acknowledge understanding and 
acceptance of these terms and limitations.
═══════════════════════════════════════════════════════════════
"""


class ThyroidTriageSystem:
    """
    Main system class for thyroid triage and clinical decision support.
    Orchestrates multi-agent workflow.
    """
    
    def __init__(self, show_disclaimer: bool = True):
        """
        Initialize the thyroid triage system.
        
        Args:
            show_disclaimer: Whether to display ethical disclaimer
        """
        if show_disclaimer:
            print(ETHICAL_DISCLAIMER)
        
        # Initialize workflow
        try:
            self.workflow = TriageWorkflow()
            self.initialized = True
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            self.initialized = False
    
    def triage_patient(
        self,
        patient_id: str,
        patient_data: Dict,
        audience: str = "doctor"
    ) -> TriageOutput:
        """
        Run complete triage for a patient.
        
        Args:
            patient_id: Unique patient identifier
            patient_data: Dictionary with patient parameters
            audience: 'doctor' or 'patient' for output format
            
        Returns:
            TriageOutput with complete assessment
        """
        if not self.initialized:
            return TriageOutput(
                patient_id=patient_id,
                risk_score=0.0,
                confidence=0.0,
                triage_category="UNKNOWN",
                doctor_report="System not initialized",
                patient_summary="System not initialized",
                evidence_sources=[],
                workflow_status="error: system not initialized"
            )
        
        # Create triage input
        triage_input = TriageInput(
            patient_id=patient_id,
            patient_data=patient_data,
            audience=audience
        )
        
        # Process through workflow
        return self.workflow.process(triage_input)
    
    def display_results(self, output: TriageOutput, audience: str = "doctor"):
        """
        Display triage results in appropriate format.
        
        Args:
            output: TriageOutput from workflow
            audience: 'doctor' or 'patient'
        """
        if audience == "doctor":
            print(output.doctor_report)
        else:
            print(output.patient_summary)
    
    def export_results(self, output: TriageOutput, filepath: str = None) -> str:
        """
        Export results to JSON file.
        
        Args:
            output: TriageOutput to export
            filepath: Where to save results (optional)
            
        Returns:
            JSON string of results
        """
        data = {
            'patient_id': output.patient_id,
            'risk_score': output.risk_score,
            'confidence': output.confidence,
            'triage_category': output.triage_category,
            'evidence_sources': output.evidence_sources,
            'status': output.workflow_status
        }
        
        json_str = json.dumps(data, indent=2)
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_str)
            print(f"✓ Results saved to {filepath}")
        
        return json_str


def demo_patient_case():
    """
    Demo with sample patient case.
    """
    print("\n" + "="*60)
    print("DEMO: SAMPLE PATIENT CASE")
    print("="*60 + "\n")
    
    # Sample patient with hypothyroidism risk
    patient_data = {
        'age': 52,
        'sex': 'F',
        'tsh': 6.2,  # Elevated (high risk)
        't3': 1.8,
        'tt4': 85,
        't4u': 0.75,
        'fti': 65,
        'on_thyroxine': 0,
        'query_on_thyroxine': 0,
        'on_antithyroid_medication': 0,
        'sick': 0,
        'pregnant': 0,
        'thyroid_surgery': 0,
        'i131_treatment': 0,
        'query_hypothyroid': 1,
        'query_hyperthyroid': 0,
        'lithium': 0,
        'goitre': 0,
        'tumor': 0,
        'hypopituitary': 0,
        'psych': 0,
        'tsh_measured': 1,
        'other_thyroid_measure': 1
    }
    
    print("PATIENT DATA:")
    for key, value in patient_data.items():
        if key in ['age', 'sex', 'tsh', 't3', 'tt4']:
            print(f"  {key}: {value}")
    print("  [additional clinical parameters...]\n")
    
    # Initialize system
    system = ThyroidTriageSystem(show_disclaimer=False)
    
    if not system.initialized:
        print("❌ System initialization failed")
        return
    
    # Run triage
    print("Running triage workflow...\n")
    result = system.triage_patient(
        patient_id="DEMO-001",
        patient_data=patient_data,
        audience="doctor"
    )
    
    # Display doctor report
    print("\n" + "="*60)
    print("DOCTOR REPORT")
    print("="*60)
    print(result.doctor_report)
    
    # Display patient summary
    print("\n" + "="*60)
    print("PATIENT SUMMARY")
    print("="*60)
    print(result.patient_summary)
    
    # Save results
    system.export_results(
        result,
        filepath="output/demo_results.json"
    )


def interactive_mode():
    """
    Interactive mode for entering patient data.
    """
    print("\n" + "="*60)
    print("INTERACTIVE THYROID TRIAGE MODE")
    print("="*60 + "\n")
    
    system = ThyroidTriageSystem(show_disclaimer=False)
    
    while True:
        print("\nOptions:")
        print("  1. Enter new patient")
        print("  2. Load sample patient")
        print("  3. Exit")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == '1':
            patient_id = input("Patient ID: ").strip()
            
            patient_data = {
                'age': float(input("Age: ")),
                'sex': input("Sex (M/F): ").strip(),
                'tsh': float(input("TSH (mIU/L): ")),
                't3': float(input("T3 (ng/dL): ")),
                'tt4': float(input("Total T4 (ng/dL): ")),
                't4u': float(input("T4 Uptake (0.0-1.0): ")),
                'fti': float(input("Free T4 Index: "))
            }
            
            result = system.triage_patient(patient_id, patient_data, audience="doctor")
            system.display_results(result, audience="doctor")
            
            save = input("\nSave results? (y/n): ").lower()
            if save == 'y':
                filename = f"output/{patient_id}_results.json"
                system.export_results(result, filepath=filename)
        
        elif choice == '2':
            demo_patient_case()
        
        elif choice == '3':
            print("\nExiting thyroid triage system. Thank you!")
            break
        
        else:
            print("Invalid option. Please try again.")


def main():
    """
    Main entry point.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Thyroid Triage AI - Clinical Decision Support System"
    )
    parser.add_argument(
        '--mode',
        choices=['demo', 'interactive', 'silent'],
        default='demo',
        help="Execution mode"
    )
    parser.add_argument(
        '--patient-id',
        default='P001',
        help="Patient ID for single case"
    )
    parser.add_argument(
        '--hide-disclaimer',
        action='store_true',
        help="Hide ethical disclaimer"
    )
    
    args = parser.parse_args()
    
    # Show disclaimer unless hidden
    show_disclaimer = not args.hide_disclaimer
    
    if args.mode == 'interactive':
        interactive_mode()
    else:
        # Demo mode
        system = ThyroidTriageSystem(show_disclaimer=show_disclaimer)
        demo_patient_case()


if __name__ == "__main__":
    main()

