"""VeriGuard AI - Core Module"""

from core.models import (
    VerificationRequest, VerificationResponse, BatchVerificationRequest,
    ExtractedClaim, Evidence, VerificationResult, RiskAssessment,
    ClaimAnalysis, VerificationStatus, SeverityLevel, ClaimType
)
from core.pipeline import VerificationPipeline, get_pipeline
from core.extractor import ClaimExtractor, get_extractor
from core.verifier import VerificationEngine, get_verifier
from core.retriever import HybridRetriever, get_retriever
from core.risk_engine import RiskEngine, get_risk_engine
from core.embedder import EmbeddingEngine, get_embedder
from core.index import KnowledgeIndex, get_knowledge_index, initialize_knowledge_base


__all__ = [
    # Models
    "VerificationRequest",
    "VerificationResponse",
    "BatchVerificationRequest",
    "ExtractedClaim",
    "Evidence",
    "VerificationResult",
    "RiskAssessment",
    "ClaimAnalysis",
    "VerificationStatus",
    "SeverityLevel",
    "ClaimType",
    # Pipeline
    "VerificationPipeline",
    "get_pipeline",
    # Components
    "ClaimExtractor",
    "get_extractor",
    "VerificationEngine",
    "get_verifier",
    "HybridRetriever",
    "get_retriever",
    "RiskEngine",
    "get_risk_engine",
    "EmbeddingEngine",
    "get_embedder",
    "KnowledgeIndex",
    "get_knowledge_index",
    "initialize_knowledge_base",
]
