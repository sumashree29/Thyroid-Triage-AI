import sys
from pathlib import Path
import pandas as pd
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from core.workflow import TriageWorkflow, TriageInput

def main():
    print("Loading TriageWorkflow...")
    workflow = TriageWorkflow()
    
    # Let's get an interesting patient from real_confounders.csv
    try:
        df = pd.read_csv("data/raw/real_confounders.csv", na_values='?')
        # Pick the first patient
        patient_data = df.iloc[0].to_dict()
        
        # Clean up nan values for JSON printing
        patient_data = {k: v for k, v in patient_data.items() if not pd.isna(v)}
        
        print(f"\nProcessing Real Confounder Patient (Class R)...")
        print(f"Input Patient Data: {json.dumps(patient_data, indent=2)}")
        
        # Run workflow
        triage_input = TriageInput(
            patient_id="TEST001",
            patient_data=patient_data,
            audience="doctor"
        )
        result = workflow.process(triage_input)
        
        print("\n\n" + "="*80)
        print("WORKFLOW END-TO-END OUTPUT")
        print("="*80)
        
        print("\n>>> DOCTOR-FACING SUMMARY REPORT <<<\n")
        print(result.doctor_report)
        print("\n>>> PATIENT-FACING SUMMARY REPORT <<<\n")
        print(result.patient_summary)
            
        print("\n>>> STRUCTURED REASONING OUTPUT <<<\n")
        print("Triage Category:", result.triage_category)
        print("Conformal Set:", result.conformal_set)
        print("Confounder Flags:", result.confounder_flags)
        print("Risk Score:", result.risk_score)
        
    except Exception as e:
        print(f"Error during workflow execution: {e}")

if __name__ == "__main__":
    main()
