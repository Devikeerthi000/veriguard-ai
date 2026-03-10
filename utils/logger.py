"""
VeriGuard AI - Production Logging System
Structured logging with JSON support, correlation IDs, and performance tracking.
"""

import logging
import sys
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from functools import wraps
from contextvars import ContextVar
from pathlib import Path

# Context variable for request correlation
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production environments."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if present
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_data["correlation_id"] = correlation_id
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        correlation_id = correlation_id_var.get()
        if correlation_id:
            record.msg = f"[{correlation_id[:8]}] {record.msg}"
        
        return super().format(record)


class VeriGuardLogger:
    """Enhanced logger with structured logging support."""
    
    def __init__(self, name: str, level: str = "INFO", log_format: str = "json"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        if log_format == "json":
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(ColoredFormatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
        
        self.logger.addHandler(console_handler)
        
        # File handler for production
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"veriguard_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)
    
    def _log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """Internal log method with extra data support."""
        record = self.logger.makeRecord(
            self.logger.name,
            getattr(logging, level.upper()),
            "(unknown file)",
            0,
            message,
            (),
            None
        )
        if extra:
            record.extra_data = extra
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, kwargs if kwargs else None)
    
    def info(self, message: str, **kwargs):
        self._log("INFO", message, kwargs if kwargs else None)
    
    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, kwargs if kwargs else None)
    
    def error(self, message: str, **kwargs):
        self._log("ERROR", message, kwargs if kwargs else None)
    
    def critical(self, message: str, **kwargs):
        self._log("CRITICAL", message, kwargs if kwargs else None)
    
    def exception(self, message: str, **kwargs):
        self.logger.exception(message, extra={"extra_data": kwargs})


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for request tracking."""
    cid = correlation_id or str(uuid.uuid4())
    correlation_id_var.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id_var.get()


def log_execution_time(logger: VeriGuardLogger):
    """Decorator to log function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                logger.info(
                    f"{func.__name__} completed",
                    execution_time_ms=round(execution_time * 1000, 2),
                    status="success"
                )
                return result
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.error(
                    f"{func.__name__} failed",
                    execution_time_ms=round(execution_time * 1000, 2),
                    status="error",
                    error=str(e)
                )
                raise
        return wrapper
    return decorator


def log_async_execution_time(logger: VeriGuardLogger):
    """Decorator to log async function execution time."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                logger.info(
                    f"{func.__name__} completed",
                    execution_time_ms=round(execution_time * 1000, 2),
                    status="success"
                )
                return result
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.error(
                    f"{func.__name__} failed",
                    execution_time_ms=round(execution_time * 1000, 2),
                    status="error",
                    error=str(e)
                )
                raise
        return wrapper
    return decorator


# Default logger instance
def get_logger(name: str = "veriguard") -> VeriGuardLogger:
    """Get a logger instance."""
    from config.settings import settings
    return VeriGuardLogger(name, settings.log_level, settings.log_format)
