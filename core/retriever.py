"""
VeriGuard AI - Advanced Retrieval System
Multi-stage retrieval with reranking and relevance scoring.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from core.embedder import get_embedder
from core.index import get_knowledge_index, IndexDocument
from core.models import Evidence
from config.settings import settings
from utils.logger import get_logger
from utils.cache import cache_result


logger = get_logger("retriever")


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    top_k: int = 5
    initial_k: int = 20  # Retrieve more for reranking
    similarity_threshold: float = 0.3
    use_reranking: bool = True
    diversity_factor: float = 0.1  # MMR diversity


class HybridRetriever:
    """
    Advanced retrieval with:
    - Semantic vector search
    - Optional cross-encoder reranking
    - Maximum Marginal Relevance (MMR) for diversity
    - Category-aware retrieval
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig(
            top_k=settings.retriever.top_k,
            similarity_threshold=settings.retriever.similarity_threshold,
            use_reranking=settings.retriever.use_reranking
        )
        self.embedder = get_embedder()
        self.index = get_knowledge_index()
        
        # Lazy load reranker
        self._reranker = None
        
        logger.info("HybridRetriever initialized")
    
    @property
    def reranker(self):
        """Lazy load cross-encoder reranker."""
        if self._reranker is None and self.config.use_reranking:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(
                    settings.retriever.rerank_model,
                    max_length=512
                )
                logger.info(f"Loaded reranker: {settings.retriever.rerank_model}")
            except Exception as e:
                logger.warning(f"Could not load reranker: {e}")
                self._reranker = False  # Sentinel to avoid retrying
        return self._reranker if self._reranker else None
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        categories: Optional[List[str]] = None,
        use_reranking: Optional[bool] = None,
        use_mmr: bool = True
    ) -> List[Evidence]:
        """
        Retrieve relevant evidence for a query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            categories: Filter by knowledge base categories
            use_reranking: Whether to apply cross-encoder reranking
            use_mmr: Whether to apply MMR diversity
            
        Returns:
            List of Evidence objects
        """
        top_k = top_k or self.config.top_k
        use_reranking = use_reranking if use_reranking is not None else self.config.use_reranking
        
        # Initial retrieval (get more for reranking)
        initial_k = self.config.initial_k if use_reranking else top_k
        
        results = self.index.search(
            query=query,
            top_k=initial_k,
            categories=categories,
            min_score=self.config.similarity_threshold
        )
        
        if not results:
            logger.debug(f"No results for query: {query[:100]}...")
            return []
        
        # Rerank with cross-encoder
        if use_reranking and self.reranker and len(results) > 1:
            results = self._rerank(query, results)
        
        # Apply MMR for diversity
        if use_mmr and len(results) > 1:
            results = self._mmr_diversify(query, results, top_k)
        else:
            results = results[:top_k]
        
        # Convert to Evidence objects
        evidence_list = []
        for doc, score in results:
            evidence = Evidence(
                id=doc.id,
                text=doc.text,
                similarity_score=score,
                relevance_score=score,  # Will be updated by reranking
                source_category=doc.category,
                metadata=doc.metadata
            )
            evidence_list.append(evidence)
        
        logger.debug(f"Retrieved {len(evidence_list)} evidence documents")
        return evidence_list
    
    def _rerank(
        self,
        query: str,
        results: List[Tuple[IndexDocument, float]]
    ) -> List[Tuple[IndexDocument, float]]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Original query
            results: Initial results from vector search
            
        Returns:
            Reranked results
        """
        # Prepare pairs for cross-encoder
        pairs = [(query, doc.text) for doc, _ in results]
        
        # Get cross-encoder scores
        scores = self.reranker.predict(pairs)
        
        # Combine with original scores (weighted average)
        reranked = []
        for (doc, orig_score), new_score in zip(results, scores):
            # Normalize cross-encoder score to 0-1 range
            norm_score = 1 / (1 + np.exp(-new_score))  # Sigmoid
            # Weighted combination
            combined_score = 0.7 * norm_score + 0.3 * orig_score
            reranked.append((doc, float(combined_score)))
        
        # Sort by combined score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    def _mmr_diversify(
        self,
        query: str,
        results: List[Tuple[IndexDocument, float]],
        top_k: int
    ) -> List[Tuple[IndexDocument, float]]:
        """
        Apply Maximum Marginal Relevance for diverse results.
        
        Args:
            query: Original query
            results: Candidate results
            top_k: Number of results to select
            
        Returns:
            Diversified results
        """
        if len(results) <= top_k:
            return results
        
        # Get embeddings
        query_emb = self.embedder.embed_single(query)
        doc_embs = np.array([doc.embedding for doc, _ in results])
        
        # Initialize with best result
        selected = [0]
        candidates = list(range(1, len(results)))
        
        lambda_param = 1 - self.config.diversity_factor
        
        while len(selected) < top_k and candidates:
            best_score = -float('inf')
            best_idx = -1
            
            for idx in candidates:
                # Relevance to query
                relevance = np.dot(doc_embs[idx], query_emb)
                
                # Maximum similarity to already selected
                max_sim = max(
                    np.dot(doc_embs[idx], doc_embs[sel_idx])
                    for sel_idx in selected
                )
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx >= 0:
                selected.append(best_idx)
                candidates.remove(best_idx)
        
        # Return selected results
        return [results[idx] for idx in selected]
    
    def retrieve_for_claims(
        self,
        claims: List[str],
        top_k: int = 3
    ) -> Dict[str, List[Evidence]]:
        """
        Retrieve evidence for multiple claims efficiently.
        
        Args:
            claims: List of claim texts
            top_k: Results per claim
            
        Returns:
            Dictionary mapping claims to evidence lists
        """
        results = {}
        for claim in claims:
            results[claim] = self.retrieve(claim, top_k=top_k)
        return results


# Global retriever instance
_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Get or create the retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
