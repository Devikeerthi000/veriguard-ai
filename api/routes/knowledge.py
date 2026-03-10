"""
VeriGuard AI - Knowledge Base Routes
Knowledge base management and search endpoints.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from core.index import get_knowledge_index
from core.retriever import get_retriever
from core.models import Evidence
from utils.logger import get_logger


router = APIRouter()
logger = get_logger("api.knowledge")


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    categories: Optional[List[str]] = Field(default=None)
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Search result model."""
    text: str
    score: float
    category: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: List[SearchResult]
    total_results: int


@router.get(
    "/knowledge/stats",
    summary="Knowledge base statistics",
    description="Get statistics about the knowledge base."
)
async def get_knowledge_stats() -> Dict:
    """
    Get knowledge base statistics.
    """
    try:
        index = get_knowledge_index()
        stats = index.get_stats()
        
        # Add category breakdown
        category_counts = {
            cat: len(indices) 
            for cat, indices in index.categories.items()
        }
        
        return {
            "total_documents": stats["total_documents"],
            "embedding_dimension": stats["dimension"],
            "categories": category_counts,
            "index_type": stats["index_type"],
            "created_at": stats["created_at"],
            "updated_at": stats["updated_at"]
        }
    except Exception as e:
        logger.error(f"Failed to get knowledge stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/knowledge/search",
    response_model=SearchResponse,
    summary="Search knowledge base",
    description="Semantic search across the knowledge base."
)
async def search_knowledge(
    request: SearchRequest = Body(
        ...,
        example={
            "query": "What is the capital of Australia?",
            "top_k": 5,
            "categories": None,
            "min_score": 0.3
        }
    )
) -> SearchResponse:
    """
    Search the knowledge base semantically.
    """
    try:
        retriever = get_retriever()
        
        evidence_list = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            categories=request.categories
        )
        
        # Filter by minimum score
        evidence_list = [
            ev for ev in evidence_list 
            if ev.similarity_score >= request.min_score
        ]
        
        results = [
            SearchResult(
                text=ev.text,
                score=ev.similarity_score,
                category=ev.source_category,
                metadata=ev.metadata
            )
            for ev in evidence_list
        ]
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results)
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/knowledge/categories",
    summary="List categories",
    description="Get all knowledge base categories and their document counts."
)
async def list_categories() -> Dict[str, int]:
    """
    List all knowledge base categories.
    """
    try:
        index = get_knowledge_index()
        return {
            cat: len(indices) 
            for cat, indices in index.categories.items()
        }
    except Exception as e:
        logger.error(f"Failed to list categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/knowledge/rebuild",
    summary="Rebuild knowledge index",
    description="Force rebuild the knowledge base index from source files."
)
async def rebuild_index() -> Dict:
    """
    Rebuild the knowledge base index.
    """
    try:
        from core.index import initialize_knowledge_base
        
        doc_count = initialize_knowledge_base(force_rebuild=True)
        
        return {
            "status": "success",
            "documents_indexed": doc_count,
            "message": "Knowledge base index rebuilt successfully"
        }
    except Exception as e:
        logger.error(f"Failed to rebuild index: {e}")
        raise HTTPException(status_code=500, detail=str(e))
