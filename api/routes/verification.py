"""
VeriGuard AI - Verification API Routes
Main verification endpoints for hallucination detection.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Body

from core.models import (
    VerificationRequest, VerificationResponse, 
    BatchVerificationRequest
)
from core.pipeline import get_pipeline
from utils.logger import get_logger


router = APIRouter()
logger = get_logger("api.verification")


@router.post(
    "/verify",
    response_model=VerificationResponse,
    summary="Verify text for hallucinations",
    description="""
Analyzes text to extract factual claims and verify them against the knowledge base.

**Process:**
1. Extracts factual claims from the input text
2. Retrieves relevant evidence from knowledge base
3. Verifies each claim against evidence
4. Calculates risk scores and severity levels

**Extraction Modes:**
- `strict`: Only explicit, verifiable facts
- `standard`: Balanced extraction (default)
- `permissive`: Aggressive extraction including implicit claims

**Verification Depth:**
- `quick`: Fast verification, less thorough
- `standard`: Balanced (default)
- `thorough`: Deep analysis with contradiction detection
    """
)
async def verify_text(
    request: VerificationRequest = Body(
        ...,
        example={
            "text": "The capital of Australia is Sydney. Water boils at 100 degrees Celsius. Einstein discovered quantum mechanics in 1925.",
            "extraction_mode": "standard",
            "verification_depth": "standard",
            "include_evidence": True,
            "include_explanations": True,
            "max_claims": 50
        }
    )
) -> VerificationResponse:
    """
    Verify text for factual accuracy and detect hallucinations.
    """
    try:
        pipeline = get_pipeline(auto_init_kb=False)
        response = pipeline.verify(request)
        return response
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/verify/quick",
    response_model=VerificationResponse,
    summary="Quick verification",
    description="Fast verification with minimal evidence and explanations."
)
async def verify_quick(
    text: str = Body(..., embed=True, min_length=1, max_length=10000)
) -> VerificationResponse:
    """
    Quick text verification with default settings.
    """
    request = VerificationRequest(
        text=text,
        extraction_mode="standard",
        verification_depth="quick",
        include_evidence=False,
        include_explanations=False
    )
    
    try:
        pipeline = get_pipeline(auto_init_kb=False)
        return pipeline.verify(request)
    except Exception as e:
        logger.error(f"Quick verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/verify/batch",
    response_model=List[VerificationResponse],
    summary="Batch verification",
    description="Verify multiple texts in a single request."
)
async def verify_batch(
    request: BatchVerificationRequest = Body(
        ...,
        example={
            "texts": [
                "The Earth is approximately 4.5 billion years old.",
                "Mount Everest is located in Africa."
            ],
            "extraction_mode": "standard",
            "verification_depth": "standard"
        }
    )
) -> List[VerificationResponse]:
    """
    Verify multiple texts in batch.
    """
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 100 texts per batch request"
        )
    
    try:
        pipeline = get_pipeline(auto_init_kb=False)
        responses = pipeline.verify_batch(request)
        return responses
    except Exception as e:
        logger.error(f"Batch verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/verify/status/{request_id}",
    summary="Get verification status",
    description="Get the status of an async verification request (for future async support)."
)
async def get_verification_status(request_id: str):
    """
    Get verification status by request ID.
    Currently returns not implemented as all requests are synchronous.
    """
    raise HTTPException(
        status_code=501,
        detail="Async verification not yet implemented. All requests are processed synchronously."
    )
