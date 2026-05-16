"""
Logging Configuration

Project-wide logging configuration that can be imported by any module
to set up consistent logging across the ControlNet training pipeline.

Usage:
    from configs.logging_config import init_logging, get_component_logger

    # Initialize logging at application startup
    init_logging()

    # Get a logger for a specific component
    logger = get_component_logger("training")
    logger.info("Training started")
"""

import os
import logging
from typing import Optional

from src.utils.logging_utils import setup_logging, get_logger, DebugMode


# Default logging settings (can be overridden via environment variables)
DEFAULT_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
DEFAULT_LOG_DIR = os.environ.get("LOG_DIR", "./logs")
DEFAULT_USE_JSON = os.environ.get("LOG_JSON", "0").lower() in ("1", "true")
DEFAULT_USE_COLORS = os.environ.get("LOG_COLORS", "1").lower() in ("1", "true")
DEFAULT_MAX_FILE_SIZE_MB = int(os.environ.get("LOG_MAX_SIZE_MB", "100"))
DEFAULT_BACKUP_COUNT = int(os.environ.get("LOG_BACKUP_COUNT", "5"))


_initialized = False


def init_logging(
    log_level: Optional[str] = None,
    log_dir: Optional[str] = None,
    use_json: Optional[bool] = None,
    use_colors: Optional[bool] = None,
    debug_mode: bool = False,
) -> logging.Logger:
    """
    Initialize project-wide logging with sensible defaults.

    Call this once at application startup. Subsequent calls are no-ops
    unless the module-level _initialized flag is reset.

    Args:
        log_level: Override default log level.
        log_dir: Override default log directory.
        use_json: Override JSON logging setting.
        use_colors: Override colored output setting.
        debug_mode: Enable debug mode with execution tracing.

    Returns:
        The configured root pipeline logger.
    """
    global _initialized
    if _initialized:
        return logging.getLogger("controlnet_pipeline")

    logger = setup_logging(
        log_level=log_level or DEFAULT_LOG_LEVEL,
        log_dir=log_dir or DEFAULT_LOG_DIR,
        use_colors=use_colors if use_colors is not None else DEFAULT_USE_COLORS,
        use_json=use_json if use_json is not None else DEFAULT_USE_JSON,
        max_file_size_mb=DEFAULT_MAX_FILE_SIZE_MB,
        backup_count=DEFAULT_BACKUP_COUNT,
        component_logs=True,
    )

    if debug_mode:
        DebugMode.enable()
        logger.info("Debug mode enabled via logging config")

    _initialized = True
    return logger


def get_component_logger(component: str) -> logging.Logger:
    """
    Get a logger for a specific pipeline component.

    Valid components: training, inference, data, evaluation, app

    Args:
        component: Component name.

    Returns:
        Logger routed to the component's log file.
    """
    return get_logger(component, component=component)


def reset_logging() -> None:
    """Reset the logging initialization flag (useful for testing)."""
    global _initialized
    _initialized = False
