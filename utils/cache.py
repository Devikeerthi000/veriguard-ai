"""
VeriGuard AI - Caching System
Multi-backend caching with TTL support and intelligent invalidation.
"""

import hashlib
import json
import time
import pickle
from abc import ABC, abstractmethod
from typing import Any, Optional, Callable
from functools import wraps
from collections import OrderedDict
from threading import Lock


class CacheBackend(ABC):
    """Abstract cache backend."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        pass


class MemoryCache(CacheBackend):
    """In-memory LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.expiry: dict = {}
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
    
    def _is_expired(self, key: str) -> bool:
        if key not in self.expiry:
            return False
        return time.time() > self.expiry[key]
    
    def _evict_expired(self) -> None:
        current_time = time.time()
        expired_keys = [
            k for k, exp_time in self.expiry.items() 
            if current_time > exp_time
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.expiry.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache or self._is_expired(key):
                self.misses += 1
                if key in self.cache:
                    del self.cache[key]
                    del self.expiry[key]
                return None
            
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self.lock:
            # Evict expired entries periodically
            if len(self.cache) % 100 == 0:
                self._evict_expired()
            
            # Remove oldest if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.expiry.pop(oldest_key, None)
            
            self.cache[key] = value
            self.cache.move_to_end(key)
            self.expiry[key] = time.time() + (ttl or self.default_ttl)
    
    def delete(self, key: str) -> None:
        with self.lock:
            self.cache.pop(key, None)
            self.expiry.pop(key, None)
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.expiry.clear()
    
    def exists(self, key: str) -> bool:
        with self.lock:
            if key not in self.cache:
                return False
            if self._is_expired(key):
                del self.cache[key]
                del self.expiry[key]
                return False
            return True
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%"
        }


class DiskCache(CacheBackend):
    """Disk-based cache using pickle serialization."""
    
    def __init__(self, cache_dir: str = ".cache", default_ttl: int = 3600):
        import os
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        os.makedirs(cache_dir, exist_ok=True)
        self.metadata_file = os.path.join(cache_dir, "_metadata.json")
        self._load_metadata()
    
    def _load_metadata(self):
        import os
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f)
    
    def _key_to_path(self, key: str) -> str:
        import os
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_key}.pkl")
    
    def get(self, key: str) -> Optional[Any]:
        import os
        if key in self.metadata:
            if time.time() > self.metadata[key]["expiry"]:
                self.delete(key)
                return None
            
            path = self._key_to_path(key)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return pickle.load(f)
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        path = self._key_to_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f)
        
        self.metadata[key] = {
            "expiry": time.time() + (ttl or self.default_ttl),
            "path": path
        }
        self._save_metadata()
    
    def delete(self, key: str) -> None:
        import os
        if key in self.metadata:
            path = self._key_to_path(key)
            if os.path.exists(path):
                os.remove(path)
            del self.metadata[key]
            self._save_metadata()
    
    def clear(self) -> None:
        import os
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
        self.metadata = {}
        self._save_metadata()
    
    def exists(self, key: str) -> bool:
        if key not in self.metadata:
            return False
        if time.time() > self.metadata[key]["expiry"]:
            self.delete(key)
            return False
        return True


class CacheManager:
    """Unified cache manager with multiple backend support."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self, 
        backend: str = "memory", 
        max_size: int = 10000,
        ttl: int = 3600,
        redis_url: Optional[str] = None
    ):
        if hasattr(self, "_initialized"):
            return
        
        self.backend_type = backend
        self.ttl = ttl
        
        if backend == "memory":
            self.backend = MemoryCache(max_size=max_size, default_ttl=ttl)
        elif backend == "disk":
            self.backend = DiskCache(default_ttl=ttl)
        elif backend == "redis":
            # Redis support would require redis-py package
            raise NotImplementedError("Redis backend requires additional setup")
        else:
            raise ValueError(f"Unknown cache backend: {backend}")
        
        self._initialized = True
    
    def get(self, key: str) -> Optional[Any]:
        return self.backend.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self.backend.set(key, value, ttl or self.ttl)
    
    def delete(self, key: str) -> None:
        self.backend.delete(key)
    
    def clear(self) -> None:
        self.backend.clear()
    
    def exists(self, key: str) -> bool:
        return self.backend.exists(key)
    
    def get_or_set(self, key: str, factory: Callable, ttl: Optional[int] = None) -> Any:
        """Get from cache or compute and store."""
        value = self.get(key)
        if value is None:
            value = factory()
            self.set(key, value, ttl)
        return value
    
    def get_stats(self) -> dict:
        """Get cache statistics if available."""
        if hasattr(self.backend, "get_stats"):
            return self.backend.get_stats()
        return {"backend": self.backend_type}


def generate_cache_key(*args, **kwargs) -> str:
    """Generate a deterministic cache key from arguments."""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()[:32]


def cache_result(ttl: Optional[int] = None, key_prefix: str = ""):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = CacheManager()
            cache_key = f"{key_prefix}{func.__name__}:{generate_cache_key(*args, **kwargs)}"
            
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = CacheManager()
            cache_key = f"{key_prefix}{func.__name__}:{generate_cache_key(*args, **kwargs)}"
            
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator
