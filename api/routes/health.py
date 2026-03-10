"""
VeriGuard AI - Health Check Routes
System health and status endpoints.
"""

from datetime import datetime
from typing import Dict

from fastapi import APIRouter

from config.settings import settings
from core.models import HealthResponse
from utils.logger import get_logger


router = APIRouter()
logger = get_logger("api.health")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the API and its components."
)
async def health_check() -> HealthResponse:
    """
    Perform health check on all system components.
    """
    components = {}
    overall_status = "healthy"
    
    # Check core components
    try:
        from core.embedder import get_embedder
        embedder = get_embedder()
        components["embedder"] = "healthy"
    except Exception as e:
        components["embedder"] = f"unhealthy: {str(e)}"
        overall_status = "degraded"
    
    try:
        from core.index import get_knowledge_index
        index = get_knowledge_index()
        if index.total_documents > 0:
            components["knowledge_base"] = f"healthy ({index.total_documents} docs)"
        else:
            components["knowledge_base"] = "empty"
            overall_status = "degraded"
    except Exception as e:
        components["knowledge_base"] = f"unhealthy: {str(e)}"
        overall_status = "degraded"
    
    try:
        from groq import Groq
        import os
        api_key = os.getenv("GROQ_API_KEY") or settings.llm.groq_api_key
        if api_key:
            components["llm"] = "configured"
        else:
            components["llm"] = "not configured"
            overall_status = "degraded"
    except Exception as e:
        components["llm"] = f"error: {str(e)}"
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=settings.version,
        timestamp=datetime.utcnow(),
        components=components
    )


@router.get(
    "/",
    summary="API root",
    description="API information and links."
)
async def root() -> Dict:
    """
    API root with basic information.
    """
    return {
        "name": settings.app_name,
        "version": settings.version,
        "description": "LLM Hallucination Detection & Verification Engine",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "verify": "POST /api/v1/verify",
            "verify_quick": "POST /api/v1/verify/quick",
            "verify_batch": "POST /api/v1/verify/batch",
            "knowledge_stats": "GET /api/v1/knowledge/stats",
            "search": "POST /api/v1/knowledge/search"
        }
    }


@router.get(
    "/ready",
    summary="Readiness check",
    description="Check if the service is ready to accept requests."
)
async def readiness() -> Dict:
    """
    Kubernetes-style readiness probe.
    """
    try:
        from core.index import get_knowledge_index
        index = get_knowledge_index()
        
        if index.total_documents > 0:
            return {"ready": True, "documents": index.total_documents}
        else:
            return {"ready": False, "reason": "Knowledge base not initialized"}
    except Exception as e:
        return {"ready": False, "reason": str(e)}


@router.get(
    "/live",
    summary="Liveness check",
    description="Check if the service is alive."
)
async def liveness() -> Dict:
    """
    Kubernetes-style liveness probe.
    """
    return {"alive": True, "timestamp": datetime.utcnow().isoformat()}
