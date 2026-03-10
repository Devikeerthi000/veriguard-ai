"""
VeriGuard AI - Configuration Management
Production-grade settings with environment variable support and validation.
"""

import os
from typing import Optional, List, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from functools import lru_cache


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    model_name: str = Field(
        default="all-mpnet-base-v2",
        description="Sentence transformer model for embeddings"
    )
    dimension: int = Field(default=768, description="Embedding dimension")
    batch_size: int = Field(default=32, description="Batch size for encoding")
    normalize: bool = Field(default=True, description="L2 normalize embeddings")
    
    class Config:
        env_prefix = "EMBEDDING_"


class LLMConfig(BaseSettings):
    """LLM provider configuration."""
    provider: Literal["groq", "openai", "anthropic"] = Field(default="groq")
    model: str = Field(default="llama-3.3-70b-versatile")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048)
    timeout: int = Field(default=30)
    retry_attempts: int = Field(default=3)
    
    # API Keys
    groq_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    
    class Config:
        env_prefix = "LLM_"
    
    @validator("groq_api_key", pre=True, always=True)
    def get_groq_key(cls, v):
        return v or os.getenv("GROQ_API_KEY")


class RetrieverConfig(BaseSettings):
    """Retrieval system configuration."""
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.3, description="Minimum similarity score")
    use_reranking: bool = Field(default=True)
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    class Config:
        env_prefix = "RETRIEVER_"


class VerificationConfig(BaseSettings):
    """Verification pipeline configuration."""
    enable_contradiction_detection: bool = Field(default=True)
    enable_temporal_analysis: bool = Field(default=True)
    enable_numeric_precision: bool = Field(default=True)
    enable_source_credibility: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.7)
    max_claims_per_request: int = Field(default=50)
    
    class Config:
        env_prefix = "VERIFICATION_"


class CacheConfig(BaseSettings):
    """Caching configuration."""
    enabled: bool = Field(default=True)
    backend: Literal["memory", "redis", "disk"] = Field(default="memory")
    redis_url: str = Field(default="redis://localhost:6379/0")
    ttl_seconds: int = Field(default=3600)
    max_size: int = Field(default=10000)
    
    class Config:
        env_prefix = "CACHE_"


class APIConfig(BaseSettings):
    """API server configuration."""
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    cors_origins: List[str] = Field(default=["*"])
    rate_limit: int = Field(default=100)  # requests per minute
    api_key_header: str = Field(default="X-API-Key")
    enable_docs: bool = Field(default=True)
    
    class Config:
        env_prefix = "API_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application metadata
    app_name: str = Field(default="VeriGuard AI")
    version: str = Field(default="2.0.0")
    environment: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    log_format: str = Field(default="json")
    
    # Data paths
    knowledge_base_path: str = Field(default="data/knowledge_base")
    index_path: str = Field(default="data/indices")
    models_path: str = Field(default="models")
    
    # Sub-configurations
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience exports
settings = get_settings()
