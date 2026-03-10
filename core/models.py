"""
VeriGuard AI - Data Models
Pydantic models for type-safe data handling throughout the application.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


class VerificationStatus(str, Enum):
    """Claim verification status."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNVERIFIABLE = "UNVERIFIABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class SeverityLevel(str, Enum):
    """Risk severity classification."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NEGLIGIBLE = "NEGLIGIBLE"


class ClaimType(str, Enum):
    """Type of factual claim."""
    FACTUAL = "FACTUAL"
    STATISTICAL = "STATISTICAL"
    TEMPORAL = "TEMPORAL"
    CAUSAL = "CAUSAL"
    COMPARATIVE = "COMPARATIVE"
    DEFINITIONAL = "DEFINITIONAL"


class SourceCredibility(str, Enum):
    """Source credibility rating."""
    AUTHORITATIVE = "AUTHORITATIVE"
    RELIABLE = "RELIABLE"
    MODERATE = "MODERATE"
    QUESTIONABLE = "QUESTIONABLE"
    UNRELIABLE = "UNRELIABLE"


# ============== Input Models ==============

class VerificationRequest(BaseModel):
    """Request model for verification endpoint."""
    text: str = Field(..., min_length=1, max_length=50000, description="Text to verify")
    extraction_mode: Literal["strict", "standard", "permissive"] = Field(
        default="standard",
        description="Claim extraction strictness"
    )
    verification_depth: Literal["quick", "standard", "thorough"] = Field(
        default="standard",
        description="Verification thoroughness"
    )
    include_evidence: bool = Field(default=True, description="Include evidence in response")
    include_explanations: bool = Field(default=True, description="Include detailed explanations")
    max_claims: int = Field(default=50, ge=1, le=100, description="Maximum claims to extract")


class BatchVerificationRequest(BaseModel):
    """Request model for batch verification."""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    extraction_mode: Literal["strict", "standard", "permissive"] = Field(default="standard")
    verification_depth: Literal["quick", "standard", "thorough"] = Field(default="standard")


# ============== Internal Models ==============

class ExtractedClaim(BaseModel):
    """Model for an extracted claim."""
    id: str = Field(..., description="Unique claim identifier")
    text: str = Field(..., description="Claim text")
    claim_type: ClaimType = Field(default=ClaimType.FACTUAL)
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    source_span: Optional[tuple] = Field(default=None, description="Character span in original text")
    entities: List[str] = Field(default_factory=list, description="Named entities in claim")
    temporal_references: List[str] = Field(default_factory=list, description="Time references")
    numerical_values: List[Dict[str, Any]] = Field(default_factory=list, description="Numbers with context")


class Evidence(BaseModel):
    """Model for retrieved evidence."""
    id: str = Field(..., description="Evidence document ID")
    text: str = Field(..., description="Evidence text")
    similarity_score: float = Field(ge=0.0, le=1.0, description="Semantic similarity score")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance after reranking")
    source_category: str = Field(default="general", description="Knowledge base category")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Result of verifying a single claim."""
    claim_id: str = Field(..., description="Reference to claim")
    status: VerificationStatus = Field(..., description="Verification status")
    confidence: float = Field(ge=0.0, le=1.0, description="Verification confidence")
    explanation: str = Field(..., description="Detailed explanation")
    evidence_used: List[Evidence] = Field(default_factory=list)
    contradictions_found: List[str] = Field(default_factory=list)
    supporting_facts: List[str] = Field(default_factory=list)
    numerical_accuracy: Optional[Dict[str, Any]] = Field(default=None)
    temporal_validity: Optional[Dict[str, Any]] = Field(default=None)


class RiskAssessment(BaseModel):
    """Risk assessment for a verification result."""
    claim_id: str = Field(..., description="Reference to claim")
    severity: SeverityLevel = Field(..., description="Risk severity")
    risk_score: float = Field(ge=0.0, le=1.0, description="Numerical risk score")
    impact_category: str = Field(default="general", description="potential impact area")
    recommended_action: str = Field(..., description="Suggested action")
    factors: List[str] = Field(default_factory=list, description="Contributing risk factors")


class ClaimAnalysis(BaseModel):
    """Complete analysis of a single claim."""
    claim: ExtractedClaim
    verification: VerificationResult
    risk: RiskAssessment


# ============== Response Models ==============

class VerificationResponse(BaseModel):
    """Complete verification response."""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    input_text: str = Field(..., description="Original input text")
    total_claims: int = Field(..., description="Number of claims extracted")
    verified_claims: int = Field(..., description="Number of claims verified")
    
    # Summary metrics
    overall_risk_score: float = Field(ge=0.0, le=1.0)
    overall_severity: SeverityLevel
    hallucination_rate: float = Field(ge=0.0, le=1.0, description="Percentage of contradicted claims")
    
    # Detailed results
    analyses: List[ClaimAnalysis] = Field(default_factory=list)
    
    # Statistics
    status_distribution: Dict[str, int] = Field(default_factory=dict)
    severity_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Performance metrics
    processing_time_ms: float = Field(..., description="Total processing time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    timestamp: datetime
    components: Dict[str, str]


class IndexStats(BaseModel):
    """Knowledge base index statistics."""
    total_documents: int
    total_categories: int
    index_size_mb: float
    last_updated: datetime
    embedding_model: str
    embedding_dimension: int
