"""
VeriGuard AI - Knowledge Base & Vector Index
Production-grade FAISS indexing with multi-category support and persistence.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import faiss

from core.embedder import EmbeddingEngine, get_embedder
from config.settings import settings
from utils.logger import get_logger, log_execution_time


logger = get_logger("index")


@dataclass
class IndexDocument:
    """Document in the knowledge base."""
    id: str
    text: str
    category: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class KnowledgeIndex:
    """
    Advanced knowledge base indexing with:
    - FAISS vector search with IVF for scale
    - Multi-category organization
    - Persistence and incremental updates
    - Metadata filtering
    """
    
    def __init__(
        self,
        dimension: Optional[int] = None,
        index_path: Optional[str] = None
    ):
        self.embedder = get_embedder()
        self.dimension = dimension or self.embedder.dimension
        self.index_path = Path(index_path or settings.index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize index
        self.index: Optional[faiss.Index] = None
        self.documents: List[IndexDocument] = []
        self.id_to_idx: Dict[str, int] = {}
        self.categories: Dict[str, List[int]] = {}
        
        # Metadata
        self.created_at: Optional[datetime] = None
        self.updated_at: Optional[datetime] = None
        self.total_documents: int = 0
        
        logger.info(f"KnowledgeIndex initialized: dim={self.dimension}")
    
    def _create_index(self, n_documents: int) -> faiss.Index:
        """
        Create appropriate FAISS index based on data size.
        
        For small datasets (<10k): Use flat L2 index (exact search)
        For medium datasets (10k-1M): Use IVF with flat quantizer
        For large datasets (>1M): Use IVF with PQ compression
        """
        if n_documents < 10000:
            # Exact search for small datasets
            logger.info("Creating flat L2 index (exact search)")
            index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine for normalized vectors
        elif n_documents < 1000000:
            # IVF index for medium datasets
            n_clusters = min(int(np.sqrt(n_documents) * 4), n_documents // 10)
            logger.info(f"Creating IVF index with {n_clusters} clusters")
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)
        else:
            # IVF-PQ for large datasets
            n_clusters = int(np.sqrt(n_documents) * 4)
            m = 8  # Number of subquantizers
            logger.info(f"Creating IVF-PQ index with {n_clusters} clusters")
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, n_clusters, m, 8)
        
        return index
    
    def load_knowledge_base(
        self,
        path: Optional[str] = None,
        force_rebuild: bool = False
    ) -> int:
        """
        Load documents from knowledge base directory.
        
        Args:
            path: Path to knowledge base directory
            force_rebuild: Force rebuilding even if cache exists
            
        Returns:
            Number of documents loaded
        """
        kb_path = Path(path or settings.knowledge_base_path)
        
        # Check for cached index
        cache_file = self.index_path / "index.pkl"
        if not force_rebuild and cache_file.exists():
            logger.info("Loading cached index...")
            return self._load_cached_index(cache_file)
        
        logger.info(f"Loading knowledge base from {kb_path}")
        documents = []
        
        # Load from individual category files
        if kb_path.is_dir():
            for file_path in kb_path.glob("*.txt"):
                category = file_path.stem.replace("_", " ").title()
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith("#"):
                            doc_id = f"{category}:{line_num}"
                            documents.append(IndexDocument(
                                id=doc_id,
                                text=line,
                                category=category,
                                metadata={"source_file": file_path.name, "line": line_num}
                            ))
        else:
            # Single file (legacy support)
            with open(kb_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        documents.append(IndexDocument(
                            id=f"doc:{line_num}",
                            text=line,
                            category="general",
                            metadata={"line": line_num}
                        ))
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Build index
        self.build_index(documents)
        
        # Cache the index
        self._save_cached_index(cache_file)
        
        return len(documents)
    
    def build_index(self, documents: List[IndexDocument]) -> None:
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of documents to index
        """
        if not documents:
            logger.warning("No documents to index")
            return
        
        logger.info(f"Building index for {len(documents)} documents...")
        
        # Store documents
        self.documents = documents
        self.total_documents = len(documents)
        
        # Build ID mapping and category index
        self.id_to_idx = {}
        self.categories = {}
        
        for idx, doc in enumerate(documents):
            self.id_to_idx[doc.id] = idx
            
            if doc.category not in self.categories:
                self.categories[doc.category] = []
            self.categories[doc.category].append(idx)
        
        # Generate embeddings
        texts = [doc.text for doc in documents]
        logger.info("Generating embeddings...")
        embeddings = self.embedder.embed(texts, show_progress=True)
        
        # Store embeddings in documents
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        
        # Create and populate index
        self.index = self._create_index(len(documents))
        
        # Train IVF index if needed
        if hasattr(self.index, 'train'):
            logger.info("Training index...")
            self.index.train(embeddings.astype(np.float32))
        
        # Add vectors
        self.index.add(embeddings.astype(np.float32))
        
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        logger.info(f"Index built: {len(documents)} documents, {len(self.categories)} categories")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        categories: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[Tuple[IndexDocument, float]]:
        """
        Search the index for relevant documents.
        
        Args:
            query: Query text
            top_k: Number of results to return
            categories: Optional filter by categories
            min_score: Minimum similarity score threshold
            
        Returns:
            List of (document, score) tuples
        """
        if self.index is None or self.total_documents == 0:
            logger.warning("Index is empty")
            return []
        
        # Get query embedding
        query_embedding = self.embedder.embed_single(query)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Determine search scope
        if categories:
            # Filter to specific categories
            valid_indices = []
            for cat in categories:
                if cat in self.categories:
                    valid_indices.extend(self.categories[cat])
            
            if not valid_indices:
                return []
            
            # Search more results to account for filtering
            search_k = min(top_k * 3, self.total_documents)
        else:
            search_k = min(top_k, self.total_documents)
        
        # Set search parameters for IVF
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(32, self.index.nlist)
        
        # Search
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Process results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
            
            doc = self.documents[idx]
            
            # Apply category filter
            if categories and doc.category not in categories:
                continue
            
            # Apply score threshold
            if score < min_score:
                continue
            
            results.append((doc, float(score)))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def add_documents(self, documents: List[IndexDocument]) -> int:
        """
        Add new documents to existing index.
        
        Args:
            documents: Documents to add
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        # Generate embeddings
        texts = [doc.text for doc in documents]
        embeddings = self.embedder.embed(texts)
        
        # Add to documents list
        start_idx = len(self.documents)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
            self.documents.append(doc)
            self.id_to_idx[doc.id] = len(self.documents) - 1
            
            if doc.category not in self.categories:
                self.categories[doc.category] = []
            self.categories[doc.category].append(len(self.documents) - 1)
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        self.total_documents = len(self.documents)
        self.updated_at = datetime.utcnow()
        
        return len(documents)
    
    def _save_cached_index(self, path: Path) -> None:
        """Save index to disk."""
        cache_data = {
            "documents": self.documents,
            "id_to_idx": self.id_to_idx,
            "categories": self.categories,
            "dimension": self.dimension,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_documents": self.total_documents
        }
        
        # Save metadata
        with open(path, "wb") as f:
            pickle.dump(cache_data, f)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        
        logger.info(f"Index cached to {path}")
    
    def _load_cached_index(self, path: Path) -> int:
        """Load index from disk."""
        with open(path, "rb") as f:
            cache_data = pickle.load(f)
        
        self.documents = cache_data["documents"]
        self.id_to_idx = cache_data["id_to_idx"]
        self.categories = cache_data["categories"]
        self.dimension = cache_data["dimension"]
        self.created_at = cache_data["created_at"]
        self.updated_at = cache_data["updated_at"]
        self.total_documents = cache_data["total_documents"]
        
        # Load FAISS index
        self.index = faiss.read_index(str(path.with_suffix(".faiss")))
        
        logger.info(f"Loaded cached index: {self.total_documents} documents")
        return self.total_documents
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_documents": self.total_documents,
            "dimension": self.dimension,
            "categories": {cat: len(indices) for cat, indices in self.categories.items()},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "index_type": type(self.index).__name__ if self.index else None
        }


# Global index instance
_knowledge_index: Optional[KnowledgeIndex] = None


def get_knowledge_index() -> KnowledgeIndex:
    """Get or create the knowledge index singleton."""
    global _knowledge_index
    if _knowledge_index is None:
        _knowledge_index = KnowledgeIndex()
    return _knowledge_index


def initialize_knowledge_base(force_rebuild: bool = False) -> int:
    """Initialize the knowledge base index."""
    index = get_knowledge_index()
    return index.load_knowledge_base(force_rebuild=force_rebuild)
