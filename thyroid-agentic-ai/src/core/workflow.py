"""
Orchestration Layer for Multi-Agent System
Coordinates the workflow between all agents.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.risk_scoring import RiskScoringAgent, RiskScore
from agents.retriever import RetrieverAgent
from agents.reasoner import ReasonerAgent, ReasoningOutput
from agents.summarizer import SummarizerAgent, SummaryOutput
from agents.confounder import ConfounderAgent
from core.conformal import ConformalWrapper


@dataclass
class TriageInput:
    """Input structure for triage workflow."""
    patient_id: str
    patient_data: Dict[str, Any]
    audience: str = "doctor"  # 'doctor' or 'patient'


@dataclass
class TriageOutput:
    """Output structure from triage workflow."""
    patient_id: str
    risk_score: float
    confidence: float
    triage_category: str
    doctor_report: str
    patient_summary: str
    evidence_sources: list
    workflow_status: str = "success"
    confounder_flags: Optional[list] = None
    conformal_set: Optional[dict] = None


class TriageWorkflow:
    """
    Orchestrates the multi-agent system for thyroid triage.
    Flow: Data → Risk Scoring → Retrieval → Reasoning → Summarization
    
    Agents:
    1. Risk Scorer: ML model inference
    2. Retriever: RAG for clinical guidelines
    3. Reasoner: Link predictions to evidence
    4. Summarizer: Audience-specific outputs
    """
    
    def __init__(
        self,
        model_path: str = "models/risk_classifier.pkl",
        encoder_path: str = "models/encoder.pkl",
        metadata_path: str = "models/metadata.pkl"
    ):
        """
        Initialize workflow with all agents.
        
        Args:
            model_path: Path to trained model
            encoder_path: Path to feature encoder
            metadata_path: Path to model metadata
        """
        print("\n" + "="*60)
        print("INITIALIZING THYROID TRIAGE WORKFLOW")
        print("="*60 + "\n")
        
        try:
            # Initialize Agent 1: Risk Scoring
            self.risk_scorer = RiskScoringAgent(
                model_path=model_path,
                encoder_path=encoder_path,
                metadata_path=metadata_path
            )
            print("✓ Agent 1 (Risk Scorer): Initialized")
        except Exception as e:
            print(f"⚠️  Agent 1 (Risk Scorer): {e}")
            self.risk_scorer = None
        
        # Initialize Agent 2: Retriever
        try:
            self.retriever = RetrieverAgent()
            print("✓ Agent 2 (Retriever): Initialized")
        except Exception as e:
            print(f"⚠️  Agent 2 (Retriever): {e}")
            self.retriever = None
        
        # Initialize Agent 3: Reasoner
        try:
            self.reasoner = ReasonerAgent()
            print("✓ Agent 3 (Reasoner): Initialized")
        except Exception as e:
            print(f"⚠️  Agent 3 (Reasoner): {e}")
            self.reasoner = None
        
        # Initialize Agent 4: Summarizer
        try:
            self.summarizer = SummarizerAgent()
            print("✓ Agent 4 (Summarizer): Initialized")
        except Exception as e:
            print(f"⚠️  Agent 4 (Summarizer): {e}")
            self.summarizer = None
            
        # Initialize Agent 5: Confounder Detector
        try:
            self.confounder_agent = ConfounderAgent()
            print("✓ Agent 5 (Confounder Detector): Initialized")
        except Exception as e:
            print(f"⚠️  Agent 5 (Confounder Detector): {e}")
            self.confounder_agent = None
            
        # Initialize Conformal Wrapper
        try:
            self.conformal_wrapper = ConformalWrapper()
            print("✓ Conformal Wrapper: Initialized")
        except Exception as e:
            print(f"⚠️  Conformal Wrapper: {e}")
            self.conformal_wrapper = None
        
        print("\n" + "="*60 + "\n")
    
    def process(self, triage_input: TriageInput) -> TriageOutput:
        """
        Execute the full triage workflow.
        
        Args:
            triage_input: Input with patient data
            
        Returns:
            Comprehensive triage output
        """
        print(f"Processing patient: {triage_input.patient_id}")
        print(f"Audience: {triage_input.audience}\n")
        
        try:
            # ============================================================
            # STEP 1: RISK SCORING AGENT
            # ============================================================
            print("→ STEP 1: Risk Scoring Agent")
            import pandas as pd
            patient_df = pd.DataFrame([triage_input.patient_data])
            risk_result = self.risk_scorer.score_patient(patient_df)
            print(f"  Risk Score: {risk_result.risk_score:.1%}")
            print(f"  Confidence: {risk_result.confidence:.1%}")
            if risk_result.uncertainty_flags:
                for flag in risk_result.uncertainty_flags:
                    print(f"  ⚠️  {flag}")
            print()
            
            # ============================================================
            # STEP 1B: CONFOUNDER DETECTION AGENT
            # ============================================================
            print("→ STEP 1B: Confounder Detection")
            try:
                confounder_flags = self.confounder_agent.detect(triage_input.patient_data)
                if confounder_flags:
                    print(f"  ⚠️  Detected {len(confounder_flags)} potential confounders/interferences.")
                    for flag in confounder_flags:
                         print(f"    - {flag['interference_type']} ({flag['confidence']} confidence)")
                else:
                    print("  ✓ No immunoassay interferences detected.")
            except Exception as e:
                print(f"  ⚠️  Confounder detection failed: {e}")
                confounder_flags = []
            print()
            
            # ============================================================
            # STEP 1C: CONFORMAL PREDICTION
            # ============================================================
            print("→ STEP 1C: Conformal Prediction Calibration")
            try:
                prob_vector = {0: 1.0 - risk_result.risk_score, 1: risk_result.risk_score}
                conformal_result = self.conformal_wrapper.get_prediction_set(prob_vector)
                print(f"  Prediction Set: {conformal_result['prediction_set']} at {conformal_result['coverage_level']*100:.0f}% coverage")
            except Exception as e:
                print(f"  ⚠️  Conformal wrapper failed: {e}")
                conformal_result = None
            print()
            
            # ============================================================
            # STEP 2: RETRIEVER AGENT
            # ============================================================
            print("→ STEP 2: Retriever Agent (RAG)")
            
            # Retrieve guidelines based on risk level
            risk_level = 'high' if risk_result.risk_score > 0.5 else 'low'
            guidelines = self.retriever.retrieve_by_risk_level(risk_level)
            
            # Also retrieve by symptoms if available
            symptoms = []
            if triage_input.patient_data.get('tsh', 0) > 4.5:
                symptoms.append('hypothyroidism')
            if triage_input.patient_data.get('tsh', 0) < 0.45:
                symptoms.append('hyperthyroidism')
            
            if symptoms:
                symptom_guidelines = self.retriever.retrieve_by_symptoms(symptoms)
                # Merge and deduplicate
                for sg in symptom_guidelines:
                    if not any(g.id == sg.id for g in guidelines):
                        guidelines.append(sg)
            
            print(f"  Retrieved {len(guidelines)} guidelines")
            for i, guide in enumerate(guidelines[:3], 1):
                print(f"    {i}. {guide.source} ({guide.severity})")
            print()
            
            # ============================================================
            # STEP 3: REASONING AGENT
            # ============================================================
            print("→ STEP 3: Reasoning Agent")
            
            # Convert guidelines to dict format for reasoning
            guidelines_dict = [
                {
                    'id': g.id,
                    'content': g.content,
                    'source': g.source,
                    'category': g.category,
                    'severity': g.severity
                }
                for g in guidelines
            ]
            
            reasoning = self.reasoner.reason(
                risk_score=risk_result.risk_score,
                confidence=risk_result.confidence,
                guidelines=guidelines_dict,
                patient_data=triage_input.patient_data,
                uncertainty_flags=risk_result.uncertainty_flags,
                confounder_flags=confounder_flags,
                conformal_set=conformal_result
            )
            
            print(f"  Triage Category: {reasoning.triage_category}")
            print(f"  Key Findings: {len(reasoning.key_findings)}")
            for finding in reasoning.key_findings[:2]:
                print(f"    • {finding}")
            print()
            
            # ============================================================
            # STEP 4: SUMMARIZER AGENT
            # ============================================================
            print("→ STEP 4: Summarizer Agent")
            
            # Convert ReasoningOutput to dict for summarizer
            reasoning_dict = {
                'risk_score': reasoning.risk_score,
                'confidence': reasoning.confidence,
                'clinical_impression': reasoning.clinical_impression,
                'key_findings': reasoning.key_findings,
                'evidence_citations': reasoning.evidence_citations,
                'recommendations': reasoning.recommendations,
                'uncertainty_notes': reasoning.uncertainty_notes,
                'triage_category': reasoning.triage_category,
                'confounder_flags': confounder_flags,
                'conformal_set': conformal_result
            }
            
            summary = self.summarizer.summarize(reasoning_dict)
            
            print(f"  Triage Level: {summary.triage_level}")
            print(f"  Doctor Report: {len(summary.doctor_report)} chars")
            print(f"  Patient Summary: {len(summary.patient_summary)} chars")
            print()
            
            # ============================================================
            # BUILD FINAL OUTPUT
            # ============================================================
            output = TriageOutput(
                patient_id=triage_input.patient_id,
                risk_score=risk_result.risk_score,
                confidence=risk_result.confidence,
                triage_category=summary.triage_level,
                doctor_report=summary.doctor_report,
                patient_summary=summary.patient_summary,
                evidence_sources=[g.source for g in guidelines[:5]],
                workflow_status="success",
                confounder_flags=confounder_flags,
                conformal_set=conformal_result
            )
            
            print("="*60)
            print(f"✓ WORKFLOW COMPLETE for {triage_input.patient_id}")
            print("="*60 + "\n")
            
            return output
            
        except Exception as e:
            print(f"\n❌ Workflow Error: {e}\n")
            import traceback
            traceback.print_exc()
            
            return TriageOutput(
                patient_id=triage_input.patient_id,
                risk_score=0.5,
                confidence=0.0,
                triage_category="UNKNOWN",
                doctor_report=f"Error: {str(e)}",
                patient_summary=f"Error processing request: {str(e)}",
                evidence_sources=[],
                workflow_status=f"error: {str(e)}"
            )
