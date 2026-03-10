"""
VeriGuard AI - Advanced Embedding Engine
Production-grade embeddings with caching, batching, and multiple model support.
"""

import numpy as np
from typing import List, Optional, Union
from pathlib import Path
import hashlib

from sentence_transformers import SentenceTransformer
from config.settings import settings
from utils.logger import get_logger, log_execution_time
from utils.cache import CacheManager, generate_cache_key


logger = get_logger("embedder")


class EmbeddingEngine:
    """
    Advanced embedding engine with:
    - Multiple model support
    - Intelligent caching
    - Batch processing
    - L2 normalization for cosine similarity
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for model reuse."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True
    ):
        if hasattr(self, "_initialized"):
            return
        
        self.model_name = model_name or settings.embedding.model_name
        self.normalize = normalize if normalize is not None else settings.embedding.normalize
        self.batch_size = settings.embedding.batch_size
        
        logger.info(f"Initializing embedding model: {self.model_name}")
        
        # Load model with optimal settings
        self.model = SentenceTransformer(
            self.model_name,
            device=device
        )
        
        # Get embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize cache
        self.cache = CacheManager(
            backend=settings.cache.backend,
            max_size=settings.cache.max_size,
            ttl=settings.cache.ttl_seconds
        )
        
        self._initialized = True
        logger.info(f"Embedding engine initialized: dim={self.dimension}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for a text."""
        return f"emb:{self.model_name}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
    
    def embed_single(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Embed a single text with caching.
        
        Args:
            text: Text to embed
            use_cache: Whether to use caching
            
        Returns:
            Embedding vector as numpy array
        """
        if use_cache:
            cache_key = self._get_cache_key(text)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )
        
        if use_cache:
            self.cache.set(cache_key, embedding)
        
        return embedding
    
    def embed(
        self,
        texts: Union[str, List[str]],
        use_cache: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Embed one or more texts with batching and caching.
        
        Args:
            texts: Single text or list of texts
            use_cache: Whether to use caching
            show_progress: Show progress bar for large batches
            
        Returns:
            Embeddings as numpy array (n_texts, dimension)
        """
        if isinstance(texts, str):
            return self.embed_single(texts, use_cache)
        
        if len(texts) == 0:
            return np.array([])
        
        # Check cache for each text
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cached = self.cache.get(cache_key)
                if cached is not None:
                    embeddings.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Embed uncached texts in batches
        if uncached_texts:
            logger.debug(f"Embedding {len(uncached_texts)} texts (cache hits: {len(embeddings)})")
            
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=show_progress
            )
            
            # Cache new embeddings
            if use_cache:
                for text, emb in zip(uncached_texts, new_embeddings):
                    cache_key = self._get_cache_key(text)
                    self.cache.set(cache_key, emb)
            
            # Add to results
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, emb))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings])
        
        return result
    
    def similarity(
        self,
        query: Union[str, np.ndarray],
        documents: Union[List[str], np.ndarray]
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query: Query text or embedding
            documents: Document texts or embeddings
            
        Returns:
            Similarity scores as numpy array
        """
        # Get query embedding
        if isinstance(query, str):
            query_emb = self.embed_single(query)
        else:
            query_emb = query
        
        # Get document embeddings
        if isinstance(documents, list) and len(documents) > 0 and isinstance(documents[0], str):
            doc_embs = self.embed(documents)
        else:
            doc_embs = documents
        
        if len(doc_embs) == 0:
            return np.array([])
        
        # Compute cosine similarity (embeddings are already normalized)
        if self.normalize:
            similarities = np.dot(doc_embs, query_emb)
        else:
            # Normalize on the fly
            query_norm = query_emb / np.linalg.norm(query_emb)
            doc_norms = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
            similarities = np.dot(doc_norms, query_norm)
        
        return similarities
    
    def get_stats(self) -> dict:
        """Get embedding engine statistics."""
        cache_stats = self.cache.get_stats() if hasattr(self.cache, "get_stats") else {}
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "cache": cache_stats
        }


# Convenience function
def get_embedder() -> EmbeddingEngine:
    """Get the singleton embedding engine instance."""
    return EmbeddingEngine()
