"""
VeriGuard AI - FastAPI Application
Production-grade REST API for hallucination detection.
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import verification, health, knowledge
from config.settings import settings
from utils.logger import get_logger, set_correlation_id
from core.index import initialize_knowledge_base


logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting VeriGuard AI API...")
    
    try:
        doc_count = initialize_knowledge_base()
        logger.info(f"Knowledge base initialized: {doc_count} documents")
    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down VeriGuard AI API...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="VeriGuard AI",
        description="""
# VeriGuard AI - LLM Hallucination Detection & Verification Engine

Advanced fact-checking API that detects hallucinations in AI-generated text.

## Features

- **Claim Extraction**: Automatically extracts factual claims from text
- **Evidence Retrieval**: Semantic search across comprehensive knowledge base
- **Multi-Stage Verification**: LLM-powered claim verification with contradiction detection
- **Risk Assessment**: Severity scoring and actionable recommendations

## Use Cases

- Verify AI-generated content before publication
- Detect misinformation in automated pipelines
- Quality assurance for RAG systems
- Compliance checking for regulated industries
        """,
        version=settings.version,
        docs_url="/docs" if settings.api.enable_docs else None,
        redoc_url="/redoc" if settings.api.enable_docs else None,
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(time.time_ns())[:12])
        set_correlation_id(request_id)
        
        start_time = time.perf_counter()
        
        response = await call_next(request)
        
        process_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"{request.method} {request.url.path}",
            status_code=response.status_code,
            process_time_ms=round(process_time, 2)
        )
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-MS"] = str(round(process_time, 2))
        
        return response
    
    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(verification.router, prefix="/api/v1", tags=["Verification"])
    app.include_router(knowledge.router, prefix="/api/v1", tags=["Knowledge Base"])
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api.workers
    )
