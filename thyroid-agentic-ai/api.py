"""
FastAPI Application for Thyroid Triage AI
REST API for clinical decision support system.
"""

from fastapi import FastAPI, HTTPException, Header, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.workflow import TriageWorkflow, TriageInput


# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# PYDANTIC MODELS
# ============================================================
class PatientData(BaseModel):
    """Patient clinical parameters."""
    age: float = Field(..., description="Patient age in years", ge=0, le=150)
    sex: str = Field(..., description="Patient sex (M/F)", pattern="^[MF]$")
    tsh: float = Field(..., description="TSH level (mIU/L)", ge=0)
    t3: Optional[float] = Field(None, description="T3 level")
    tt4: Optional[float] = Field(None, description="Total T4 level")
    t4u: Optional[float] = Field(None, description="T4 uptake")
    fti: Optional[float] = Field(None, description="Free T4 index")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 52,
                "sex": "F",
                "tsh": 6.2,
                "t3": 1.8,
                "tt4": 85,
                "t4u": 0.75,
                "fti": 65
            }
        }


class TriageRequest(BaseModel):
    """Triage request model."""
    patient_id: str = Field(..., description="Unique patient identifier")
    patient_data: PatientData = Field(..., description="Patient clinical data")
    audience: str = Field("doctor", description="Output audience (doctor/patient)")
    include_full_report: bool = Field(True, description="Include full report in response")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P001",
                "patient_data": {
                    "age": 52,
                    "sex": "F",
                    "tsh": 6.2,
                    "t3": 1.8,
                    "tt4": 85,
                    "t4u": 0.75,
                    "fti": 65
                },
                "audience": "doctor",
                "include_full_report": True
            }
        }


class TriageResponse(BaseModel):
    """Triage response model."""
    patient_id: str
    risk_score: float = Field(..., description="Risk score 0-1")
    confidence: float = Field(..., description="Model confidence 0-1")
    triage_category: str = Field(..., description="Triage level (URGENT/HIGH_PRIORITY/ROUTINE)")
    summary: Optional[str] = Field(None, description="Brief summary")
    full_report: Optional[str] = Field(None, description="Full report (if requested)")
    evidence_sources: List[str] = Field(..., description="Evidence citations")
    status: str = Field(..., description="Processing status")
    confounder_flags: Optional[List[Dict]] = Field(None, description="Flags for detected confounders")
    conformal_set: Optional[Dict] = Field(None, description="Conformal prediction set")
    clinical_impression: Optional[str] = Field(None, description="Clinical impression summary")
    key_findings: Optional[List[Dict]] = Field(None, description="Key clinical findings structured")
    recommendations: Optional[List[str]] = Field(None, description="Recommended actions")
    evidence_citations: Optional[List[Dict]] = Field(None, description="Structured evidence citations")
    uncertainty_notes: Optional[List[str]] = Field(None, description="Notes on prediction uncertainty")


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    agents_initialized: Dict[str, bool]


# ============================================================
# FASTAPI APP INITIALIZATION
# ============================================================
app = FastAPI(
    title="Thyroid Triage AI API",
    description="Clinical decision support system for thyroid triage using multi-agent AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# MOUNT STATIC FILES FOR UI
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def add_cache_control_header(request, call_next):
    """Prevent browser caching for all responses during development."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Global workflow instance
workflow = None


@app.on_event("startup")
async def startup_event():
    """Initialize workflow on startup."""
    global workflow
    logger.info("Starting up Thyroid Triage AI API...")
    
    try:
        workflow = TriageWorkflow()
        logger.info("✓ Workflow initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize workflow: {e}")
        workflow = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Thyroid Triage AI API...")


# ============================================================
# HEALTH CHECK ENDPOINTS
# ============================================================
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    Returns system status and initialized agents.
    """
    agents_status = {
        "risk_scorer": workflow.risk_scorer is not None if workflow else False,
        "retriever": workflow.retriever is not None if workflow else False,
        "reasoner": workflow.reasoner is not None if workflow else False,
        "summarizer": workflow.summarizer is not None if workflow else False
    }
    
    status = "healthy" if all(agents_status.values()) else "degraded"
    
    return HealthCheckResponse(
        status=status,
        version="1.0.0",
        agents_initialized=agents_status
    )


@app.get("/")
async def root():
    """Serve the sleek AI UI."""
    return FileResponse('static/index.html')


# ============================================================
# MAIN TRIAGE ENDPOINT
# ============================================================
@app.post("/triage", response_model=TriageResponse)
async def run_triage(
    request: TriageRequest,
    x_clinician_id: Optional[str] = Header(None)
):
    """
    Run complete thyroid triage for a patient.
    
    **Request Body:**
    - `patient_id`: Unique patient identifier
    - `patient_data`: Patient clinical parameters
    - `audience`: Output format ('doctor' or 'patient')
    - `include_full_report`: Whether to include full report
    
    **Returns:**
    - Risk score and confidence
    - Triage category (URGENT/HIGH_PRIORITY/ROUTINE)
    - Summary and full report
    - Evidence citations
    - Processing status
    
    **Important:**
    This is a clinical decision support tool, not a diagnostic system.
    All outputs must be reviewed by qualified healthcare providers.
    """
    
    if workflow is None:
        raise HTTPException(
            status_code=503,
            detail="Workflow not initialized. Service unavailable."
        )
    
    try:
        # Log triage request
        logger.info(f"Triage request for patient: {request.patient_id}")
        if x_clinician_id:
            logger.info(f"  Clinician: {x_clinician_id}")
        
        # Convert patient data to dict
        patient_dict = request.patient_data.dict(exclude_none=True)
        
        # Create triage input
        triage_input = TriageInput(
            patient_id=request.patient_id,
            patient_data=patient_dict,
            audience=request.audience
        )
        
        # Run workflow
        output = workflow.process(triage_input)
        
        # Prepare response
        response = TriageResponse(
            patient_id=output.patient_id,
            risk_score=output.risk_score,
            confidence=output.confidence,
            triage_category=output.triage_category,
            summary=get_summary(output, request.audience),
            full_report=output.doctor_report if request.include_full_report else None,
            evidence_sources=output.evidence_sources,
            status=output.workflow_status,
            confounder_flags=output.confounder_flags,
            conformal_set=output.conformal_set,
            clinical_impression=output.clinical_impression,
            key_findings=output.key_findings,
            recommendations=output.recommendations,
            evidence_citations=output.evidence_citations,
            uncertainty_notes=output.uncertainty_notes
        )
        
        logger.info(f"✓ Triage complete for {request.patient_id} - {response.triage_category}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Triage error for {request.patient_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Triage processing failed: {str(e)}"
        )


@app.post("/batch-triage")
async def batch_triage(requests: List[TriageRequest]):
    """
    Process multiple triage requests in batch.
    
    Returns array of triage results.
    """
    results = []
    
    for req in requests:
        try:
            result = await run_triage(req)
            results.append(result)
        except HTTPException as e:
            results.append({
                "patient_id": req.patient_id,
                "status": "error",
                "error": str(e.detail)
            })
    
    return {"results": results, "total": len(results)}


@app.get("/about")
async def about():
    """
    About the system - includes ethical disclaimer.
    """
    return {
        "name": "Thyroid Triage AI",
        "version": "1.0.0",
        "description": "Multi-agent AI system for thyroid disease triage and clinical decision support",
        "agents": {
            "agent_1": "Risk Scoring (ML model inference + confidence)",
            "agent_2": "Retriever (RAG over clinical guidelines)",
            "agent_3": "Reasoning (Evidence-based explanation)",
            "agent_4": "Summarizer (Doctor & patient outputs)"
        },
        "disclaimer": {
            "type": "CLINICAL DECISION SUPPORT ONLY",
            "not_a_diagnosis": True,
            "requires_clinical_review": True,
            "message": "This system provides decision support, not medical diagnosis. All outputs must be reviewed by qualified healthcare providers in accordance with institutional protocols."
        }
    }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def get_summary(output, audience: str) -> str:
    """Extract appropriate summary based on audience."""
    if audience == "patient":
        # First paragraph of patient summary
        lines = output.patient_summary.split('\n')
        summary_lines = []
        for line in lines:
            if line.strip() and not line.startswith('═'):
                summary_lines.append(line)
            if len(summary_lines) >= 5:
                break
        return '\n'.join(summary_lines)
    else:
        # First section of doctor report
        lines = output.doctor_report.split('\n')
        return '\n'.join(lines[:10])


# ============================================================
# ERROR HANDLERS
# ============================================================
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content={"detail": f"Validation error: {str(exc)}"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║   THYROID TRIAGE AI - FASTAPI SERVER                          ║
╚════════════════════════════════════════════════════════════════╝

Starting server...
API Documentation: http://localhost:8000/docs
Health Check: http://localhost:8000/health
About: http://localhost:8000/about
    """)
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
