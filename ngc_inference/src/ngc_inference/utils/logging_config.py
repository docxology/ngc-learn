"""
Logging configuration for NGC Inference.

Professional logging setup using loguru with structured output.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, logs only to console)
        rotation: When to rotate log file
        retention: How long to keep old logs
        format_string: Custom format string
    """
    # Remove default handler
    logger.remove()
    
    # Default format with context
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=log_level,
        colorize=True
    )
    
    # File handler if specified
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format_string,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
        logger.info(f"Logging to file: {log_file}")
    
    logger.info(f"Logging configured at {log_level} level")


def get_logger(name: str):
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Initialize with sensible defaults
setup_logging(log_level="INFO")


