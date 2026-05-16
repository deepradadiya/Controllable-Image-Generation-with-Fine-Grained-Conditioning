"""
Utilities Module

This module contains utility functions and helpers:
- Colab-specific utilities and Google Drive integration
- Memory optimization and GPU management utilities
- Visualization tools for training monitoring
- Error handling, retry logic, and recovery utilities
- Logging and debugging utilities
- General helper functions and common utilities
"""

from .colab_helpers import ColabTrainingHelper
from .error_handling import (
    CheckpointError,
    ColabEnvironmentError,
    ControlNetPipelineError,
    DatasetDownloadError,
    DatasetError,
    DatasetExtractionError,
    DatasetValidationError,
    DriveError,
    ErrorRecoveryManager,
    ErrorSeverity,
    GenerationError,
    InferenceError,
    ModelLoadingError,
    SessionTimeoutError,
    TrainingDivergenceError,
    TrainingError,
    TrainingOOMError,
    create_error_report,
    fallback,
    format_error_for_user,
    retry_with_backoff,
    safe_execute,
)
from .memory_utils import MemoryOptimizer
from .performance_monitor import (
    GPUMemoryTracker,
    TrainingSpeedMonitor,
    SystemHealthMonitor,
    PerformanceMonitor,
    PerformanceReport,
)
from .visualize import TrainingVisualizer
from .logging_utils import (
    setup_logging,
    get_logger,
    DebugMode,
    LogAnalyzer,
    log_execution,
    trace_memory,
)

__all__ = [
    "ColabTrainingHelper",
    "MemoryOptimizer",
    "TrainingVisualizer",
    # Logging and debugging
    "setup_logging",
    "get_logger",
    "DebugMode",
    "LogAnalyzer",
    "log_execution",
    "trace_memory",
    # Performance monitoring
    "GPUMemoryTracker",
    "TrainingSpeedMonitor",
    "SystemHealthMonitor",
    "PerformanceMonitor",
    "PerformanceReport",
    # Error handling
    "ControlNetPipelineError",
    "DatasetError",
    "DatasetDownloadError",
    "DatasetExtractionError",
    "DatasetValidationError",
    "TrainingError",
    "TrainingOOMError",
    "TrainingDivergenceError",
    "CheckpointError",
    "InferenceError",
    "ModelLoadingError",
    "GenerationError",
    "ColabEnvironmentError",
    "SessionTimeoutError",
    "DriveError",
    # Recovery and utilities
    "ErrorRecoveryManager",
    "ErrorSeverity",
    "retry_with_backoff",
    "fallback",
    "safe_execute",
    "format_error_for_user",
    "create_error_report",
]