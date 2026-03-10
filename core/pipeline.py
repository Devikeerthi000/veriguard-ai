"""
VeriGuard AI - Core Pipeline
Main orchestration of the verification pipeline.
"""

import time
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from core.models import (
    VerificationRequest, VerificationResponse, BatchVerificationRequest,
    ExtractedClaim, Evidence, VerificationResult, RiskAssessment,
    ClaimAnalysis, VerificationStatus, SeverityLevel
)
from core.extractor import get_extractor
from core.retriever import get_retriever
from core.verifier import get_verifier
from core.risk_engine import get_risk_engine
from core.index import initialize_knowledge_base
from utils.logger import get_logger, set_correlation_id


logger = get_logger("pipeline")


class VerificationPipeline:
    """
    Main verification pipeline orchestrating:
    1. Claim extraction
    2. Evidence retrieval
    3. Claim verification
    4. Risk assessment
    5. Response aggregation
    """
    
    def __init__(self, auto_init_kb: bool = True):
        """
        Initialize the verification pipeline.
        
        Args:
            auto_init_kb: Automatically initialize knowledge base on startup
        """
        self.extractor = get_extractor()
        self.retriever = get_retriever()
        self.verifier = get_verifier()
        self.risk_engine = get_risk_engine()
        
        if auto_init_kb:
            logger.info("Initializing knowledge base...")
            doc_count = initialize_knowledge_base()
            logger.info(f"Knowledge base ready: {doc_count} documents")
    
    def verify(self, request: VerificationRequest) -> VerificationResponse:
        """
        Execute the full verification pipeline.
        
        Args:
            request: Verification request with text and options
            
        Returns:
            Complete verification response
        """
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())[:12]
        set_correlation_id(request_id)
        
        logger.info(f"Starting verification pipeline", 
                   text_length=len(request.text),
                   mode=request.extraction_mode,
                   depth=request.verification_depth)
        
        # Step 1: Extract claims
        claims = self.extractor.extract(
            text=request.text,
            mode=request.extraction_mode,
            max_claims=request.max_claims
        )
        
        if not claims:
            return self._empty_response(request_id, request.text, start_time)
        
        logger.info(f"Extracted {len(claims)} claims")
        
        # Step 2: Retrieve evidence for all claims
        evidence_map: Dict[str, List[Evidence]] = {}
        for claim in claims:
            evidence = self.retriever.retrieve(
                query=claim.text,
                top_k=5 if request.verification_depth == "thorough" else 3
            )
            evidence_map[claim.text] = evidence
        
        # Step 3: Verify each claim
        verifications: List[VerificationResult] = []
        for claim in claims:
            evidence = evidence_map.get(claim.text, [])
            result = self.verifier.verify(
                claim=claim,
                evidence=evidence,
                depth=request.verification_depth
            )
            verifications.append(result)
        
        # Step 4: Assess risk for each claim
        risk_assessments: List[RiskAssessment] = []
        for claim, verification in zip(claims, verifications):
            risk = self.risk_engine.assess(claim, verification)
            risk_assessments.append(risk)
        
        # Step 5: Build claim analyses
        analyses: List[ClaimAnalysis] = []
        for claim, verification, risk in zip(claims, verifications, risk_assessments):
            # Optionally filter evidence from response
            if not request.include_evidence:
                verification.evidence_used = []
            
            # Optionally filter explanations
            if not request.include_explanations:
                verification.explanation = ""
            
            analyses.append(ClaimAnalysis(
                claim=claim,
                verification=verification,
                risk=risk
            ))
        
        # Step 6: Calculate aggregate metrics
        aggregate = self.risk_engine.calculate_aggregate_risk(risk_assessments)
        
        # Build status distribution
        status_dist = {}
        for v in verifications:
            status_key = v.status.value
            status_dist[status_key] = status_dist.get(status_key, 0) + 1
        
        # Calculate hallucination rate
        contradicted = sum(1 for v in verifications if v.status == VerificationStatus.CONTRADICTED)
        hallucination_rate = contradicted / len(verifications) if verifications else 0
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        response = VerificationResponse(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            input_text=request.text,
            total_claims=len(claims),
            verified_claims=len(verifications),
            overall_risk_score=aggregate["overall_score"],
            overall_severity=aggregate["overall_severity"],
            hallucination_rate=hallucination_rate,
            analyses=analyses,
            status_distribution=status_dist,
            severity_distribution=aggregate["severity_distribution"],
            processing_time_ms=round(processing_time, 2)
        )
        
        logger.info(
            f"Verification complete",
            claims=len(claims),
            hallucination_rate=f"{hallucination_rate:.1%}",
            risk_score=aggregate["overall_score"],
            time_ms=round(processing_time, 2)
        )
        
        return response
    
    def verify_text(
        self,
        text: str,
        mode: str = "standard",
        depth: str = "standard"
    ) -> VerificationResponse:
        """
        Convenience method for simple text verification.
        
        Args:
            text: Text to verify
            mode: Extraction mode
            depth: Verification depth
            
        Returns:
            Verification response
        """
        request = VerificationRequest(
            text=text,
            extraction_mode=mode,
            verification_depth=depth
        )
        return self.verify(request)
    
    def verify_batch(
        self,
        request: BatchVerificationRequest
    ) -> List[VerificationResponse]:
        """
        Verify multiple texts.
        
        Args:
            request: Batch verification request
            
        Returns:
            List of verification responses
        """
        responses = []
        for text in request.texts:
            single_request = VerificationRequest(
                text=text,
                extraction_mode=request.extraction_mode,
                verification_depth=request.verification_depth
            )
            response = self.verify(single_request)
            responses.append(response)
        return responses
    
    def _empty_response(
        self,
        request_id: str,
        text: str,
        start_time: float
    ) -> VerificationResponse:
        """Create response when no claims are extracted."""
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return VerificationResponse(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            input_text=text,
            total_claims=0,
            verified_claims=0,
            overall_risk_score=0.0,
            overall_severity=SeverityLevel.NEGLIGIBLE,
            hallucination_rate=0.0,
            analyses=[],
            status_distribution={},
            severity_distribution={s.value: 0 for s in SeverityLevel},
            processing_time_ms=round(processing_time, 2)
        )


# Global pipeline instance
_pipeline: Optional[VerificationPipeline] = None


def get_pipeline(auto_init_kb: bool = True) -> VerificationPipeline:
    """Get or create the pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = VerificationPipeline(auto_init_kb=auto_init_kb)
    return _pipeline
