"""
Comprehensive Error Handling and Recovery Module

This module provides structured exception classes, retry logic with exponential
backoff, and graceful degradation utilities for the ControlNet training pipeline.

Key components:
- Structured exception hierarchy for different failure modes
- Retry decorator with configurable exponential backoff
- Graceful degradation utilities (fallback decorator, safe_execute context manager)
- ErrorRecoveryManager for tracking failures and suggesting recovery actions
- Error reporting utilities for user-friendly messages and debug reports

Requirements: 4.7, 12.5
"""

import functools
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

logger = logging.getLogger("controlnet_pipeline.error_handling")

T = TypeVar("T")


# =============================================================================
# Structured Exception Hierarchy
# =============================================================================


class ControlNetPipelineError(Exception):
    """Base exception for all ControlNet pipeline errors.

    All custom exceptions in the pipeline inherit from this class,
    enabling broad exception catching when needed.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        super().__init__(message)


class DatasetError(ControlNetPipelineError):
    """Errors related to dataset operations.

    Covers download failures, extraction failures, and validation errors
    during dataset processing.
    """

    def __init__(
        self,
        message: str,
        error_type: str = "general",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.error_type = error_type
        super().__init__(message, details)


class DatasetDownloadError(DatasetError):
    """Raised when dataset download fails after retries."""

    def __init__(
        self,
        message: str,
        retry_count: int = 0,
        last_error: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.retry_count = retry_count
        self.last_error = last_error
        super().__init__(
            f"Download failed after {retry_count} retries: {message}",
            error_type="download",
            details=details,
        )


class DatasetExtractionError(DatasetError):
    """Raised when condition map extraction fails."""

    def __init__(
        self,
        message: str,
        condition_type: str = "unknown",
        image_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.condition_type = condition_type
        self.image_id = image_id
        super().__init__(
            f"Extraction failed for {condition_type} (image: {image_id}): {message}",
            error_type="extraction",
            details=details,
        )


class DatasetValidationError(DatasetError):
    """Raised when dataset validation fails."""

    def __init__(
        self,
        message: str,
        invalid_samples: int = 0,
        total_samples: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.invalid_samples = invalid_samples
        self.total_samples = total_samples
        super().__init__(
            f"Validation failed ({invalid_samples}/{total_samples} invalid): {message}",
            error_type="validation",
            details=details,
        )


class TrainingError(ControlNetPipelineError):
    """Errors related to model training.

    Covers OOM errors, training divergence, and checkpoint failures.
    """

    def __init__(
        self,
        message: str,
        error_type: str = "general",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.error_type = error_type
        super().__init__(message, details)


class TrainingOOMError(TrainingError):
    """Raised when GPU runs out of memory during training."""

    def __init__(
        self,
        message: str,
        current_usage_gb: Optional[float] = None,
        attempted_allocation_gb: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.current_usage_gb = current_usage_gb
        self.attempted_allocation_gb = attempted_allocation_gb
        super().__init__(
            f"GPU OOM: {message} (usage: {current_usage_gb}GB, "
            f"attempted: {attempted_allocation_gb}GB)",
            error_type="oom",
            details=details,
        )


class TrainingDivergenceError(TrainingError):
    """Raised when training metrics indicate divergence."""

    def __init__(
        self,
        message: str,
        loss_history: Optional[List[float]] = None,
        divergence_threshold: float = 10.0,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.loss_history = loss_history or []
        self.divergence_threshold = divergence_threshold
        super().__init__(
            f"Training diverged (threshold: {divergence_threshold}): {message}",
            error_type="divergence",
            details=details,
        )


class CheckpointError(TrainingError):
    """Raised when checkpoint save/load operations fail."""

    def __init__(
        self,
        message: str,
        checkpoint_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.checkpoint_path = checkpoint_path
        super().__init__(
            f"Checkpoint error at '{checkpoint_path}': {message}",
            error_type="checkpoint",
            details=details,
        )


class InferenceError(ControlNetPipelineError):
    """Errors related to model inference.

    Covers model loading failures and generation failures.
    """

    def __init__(
        self,
        message: str,
        error_type: str = "general",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.error_type = error_type
        super().__init__(message, details)


class ModelLoadingError(InferenceError):
    """Raised when model loading fails."""

    def __init__(
        self,
        message: str,
        model_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.model_path = model_path
        super().__init__(
            f"Failed to load model from '{model_path}': {message}",
            error_type="model_loading",
            details=details,
        )


class GenerationError(InferenceError):
    """Raised when image generation fails."""

    def __init__(
        self,
        message: str,
        condition_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.condition_type = condition_type
        super().__init__(
            f"Generation failed (condition: {condition_type}): {message}",
            error_type="generation",
            details=details,
        )


class ColabEnvironmentError(ControlNetPipelineError):
    """Errors specific to the Google Colab environment.

    Covers session timeouts, Google Drive issues, and resource limitations.
    """

    def __init__(
        self,
        message: str,
        error_type: str = "general",
        time_remaining: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.error_type = error_type
        self.time_remaining = time_remaining
        super().__init__(message, details)


class SessionTimeoutError(ColabEnvironmentError):
    """Raised when Colab session is about to timeout or has timed out."""

    def __init__(
        self,
        message: str,
        time_remaining: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            f"Session timeout: {message}",
            error_type="session_timeout",
            time_remaining=time_remaining,
            details=details,
        )


class DriveError(ColabEnvironmentError):
    """Raised when Google Drive operations fail."""

    def __init__(
        self,
        message: str,
        drive_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.drive_path = drive_path
        super().__init__(
            f"Drive error at '{drive_path}': {message}",
            error_type="drive",
            details=details,
        )


# =============================================================================
# Retry Logic with Exponential Backoff
# =============================================================================


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """Decorator that retries a function with exponential backoff on failure.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exponential_base: Base for exponential backoff calculation.
        retryable_exceptions: Tuple of exception types that trigger a retry.
        on_retry: Optional callback invoked on each retry with (exception, attempt).

    Returns:
        Decorated function with retry logic.

    Example:
        @retry_with_backoff(max_retries=3, base_delay=2.0)
        def download_file(url):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        # Final attempt failed, raise the exception
                        logger.error(
                            f"Function '{func.__name__}' failed after "
                            f"{max_retries + 1} attempts: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay,
                    )

                    logger.warning(
                        f"Function '{func.__name__}' failed (attempt "
                        f"{attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(delay)

            # Should not reach here, but just in case
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


# =============================================================================
# Graceful Degradation Utilities
# =============================================================================


def fallback(default_value: Any = None, exceptions: Tuple[Type[Exception], ...] = (Exception,), log_level: str = "warning"):
    """Decorator that catches exceptions and returns a default value.

    Use this for non-critical operations where a failure should not
    halt the entire pipeline.

    Args:
        default_value: Value to return when the function raises an exception.
        exceptions: Tuple of exception types to catch.
        log_level: Logging level for caught exceptions.

    Returns:
        Decorated function that returns default_value on failure.

    Example:
        @fallback(default_value=0.0)
        def compute_optional_metric(data):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                log_fn = getattr(logger, log_level, logger.warning)
                log_fn(
                    f"Function '{func.__name__}' failed, returning default "
                    f"value ({default_value!r}): {e}"
                )
                return default_value

        return wrapper

    return decorator


@contextmanager
def safe_execute(operation_name: str = "operation", default_value: Any = None, suppress_exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """Context manager for non-critical operations that should not halt the pipeline.

    Catches exceptions, logs them, and yields a result container that holds
    either the successful result or the default value.

    Args:
        operation_name: Human-readable name for logging.
        default_value: Value to use if the operation fails.
        suppress_exceptions: Exception types to suppress.

    Yields:
        A ResultContainer that holds the operation result or default.

    Example:
        with safe_execute("optional metric computation", default_value=0.0) as result:
            result.value = compute_expensive_metric(data)
        # result.value is 0.0 if compute_expensive_metric raised
        print(result.value)
    """
    result = _ResultContainer(value=default_value)
    try:
        yield result
    except suppress_exceptions as e:
        logger.warning(
            f"Non-critical operation '{operation_name}' failed: {e}. "
            f"Using default value: {default_value!r}"
        )
        result.value = default_value
        result.error = e


class _ResultContainer:
    """Simple container for holding operation results within safe_execute."""

    def __init__(self, value: Any = None):
        self.value = value
        self.error: Optional[Exception] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


# =============================================================================
# Error Recovery Manager
# =============================================================================


class ErrorSeverity(Enum):
    """Severity levels for tracked errors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorRecord:
    """Record of a single error occurrence."""

    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_action: Optional[str] = None
    resolved: bool = False


@dataclass
class RecoveryAction:
    """A suggested recovery action for a specific error pattern."""

    description: str
    action_type: str  # e.g., "reduce_batch_size", "clear_cache", "checkpoint_rollback"
    priority: int = 0
    auto_applicable: bool = False


class ErrorRecoveryManager:
    """Tracks failures and suggests recovery actions.

    This class maintains a history of errors, identifies patterns,
    and provides actionable recovery suggestions.

    Example:
        manager = ErrorRecoveryManager()
        try:
            train_step()
        except RuntimeError as e:
            manager.record_error(e, severity=ErrorSeverity.HIGH)
            actions = manager.suggest_recovery()
            for action in actions:
                print(f"Suggested: {action.description}")
    """

    def __init__(self, max_history: int = 100):
        self.error_history: List[ErrorRecord] = []
        self.max_history = max_history
        self._recovery_rules: Dict[str, List[RecoveryAction]] = self._default_recovery_rules()

    def record_error(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorRecord:
        """Record an error occurrence for tracking and analysis.

        Args:
            error: The exception that occurred.
            severity: Severity level of the error.
            context: Additional context about when/where the error occurred.

        Returns:
            The created ErrorRecord.
        """
        record = ErrorRecord(
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            timestamp=datetime.now().isoformat(),
            context=context or {},
        )

        self.error_history.append(record)

        # Trim history if needed
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]

        logger.info(
            f"Recorded error: {record.error_type} "
            f"(severity: {severity.value})"
        )

        return record

    def suggest_recovery(
        self, error: Optional[Exception] = None
    ) -> List[RecoveryAction]:
        """Suggest recovery actions based on error history and current error.

        Args:
            error: Optional current error to base suggestions on.
                   If None, uses the most recent error in history.

        Returns:
            List of RecoveryAction suggestions sorted by priority.
        """
        if error is not None:
            error_type = type(error).__name__
        elif self.error_history:
            error_type = self.error_history[-1].error_type
        else:
            return []

        actions = list(self._recovery_rules.get(error_type, []))

        # Add pattern-based suggestions
        actions.extend(self._pattern_based_suggestions())

        # Sort by priority (higher = more important)
        actions.sort(key=lambda a: a.priority, reverse=True)

        return actions

    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracked errors.

        Returns:
            Dictionary with error counts, patterns, and severity breakdown.
        """
        if not self.error_history:
            return {"total_errors": 0, "by_type": {}, "by_severity": {}}

        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for record in self.error_history:
            by_type[record.error_type] = by_type.get(record.error_type, 0) + 1
            by_severity[record.severity.value] = (
                by_severity.get(record.severity.value, 0) + 1
            )

        return {
            "total_errors": len(self.error_history),
            "by_type": by_type,
            "by_severity": by_severity,
            "most_recent": self.error_history[-1].error_type,
            "most_common": max(by_type, key=by_type.get),  # type: ignore[arg-type]
        }

    def clear_history(self) -> None:
        """Clear all error history."""
        self.error_history.clear()

    def _pattern_based_suggestions(self) -> List[RecoveryAction]:
        """Generate suggestions based on error patterns in history."""
        suggestions = []

        if len(self.error_history) < 2:
            return suggestions

        # Detect repeated OOM errors -> suggest more aggressive memory reduction
        recent_oom = sum(
            1
            for r in self.error_history[-5:]
            if r.error_type in ("TrainingOOMError", "RuntimeError")
            and "memory" in r.message.lower()
        )
        if recent_oom >= 2:
            suggestions.append(
                RecoveryAction(
                    description=(
                        "Multiple OOM errors detected. Consider reducing batch size "
                        "further, enabling gradient checkpointing, or using FP16."
                    ),
                    action_type="aggressive_memory_reduction",
                    priority=10,
                    auto_applicable=False,
                )
            )

        # Detect repeated divergence -> suggest learning rate reduction
        recent_divergence = sum(
            1
            for r in self.error_history[-5:]
            if r.error_type == "TrainingDivergenceError"
        )
        if recent_divergence >= 2:
            suggestions.append(
                RecoveryAction(
                    description=(
                        "Repeated training divergence. Consider reducing learning "
                        "rate by 10x or rolling back to a stable checkpoint."
                    ),
                    action_type="reduce_learning_rate",
                    priority=9,
                    auto_applicable=False,
                )
            )

        # Detect repeated download failures -> suggest alternative source
        recent_download = sum(
            1
            for r in self.error_history[-5:]
            if r.error_type in ("DatasetDownloadError", "DatasetError")
            and "download" in r.message.lower()
        )
        if recent_download >= 2:
            suggestions.append(
                RecoveryAction(
                    description=(
                        "Repeated download failures. Check network connectivity "
                        "or try an alternative dataset source."
                    ),
                    action_type="check_network",
                    priority=7,
                    auto_applicable=False,
                )
            )

        return suggestions

    @staticmethod
    def _default_recovery_rules() -> Dict[str, List[RecoveryAction]]:
        """Define default recovery rules for known error types."""
        return {
            "TrainingOOMError": [
                RecoveryAction(
                    description="Reduce batch size by half",
                    action_type="reduce_batch_size",
                    priority=8,
                    auto_applicable=True,
                ),
                RecoveryAction(
                    description="Enable gradient checkpointing",
                    action_type="enable_gradient_checkpointing",
                    priority=7,
                    auto_applicable=True,
                ),
                RecoveryAction(
                    description="Clear GPU cache and retry",
                    action_type="clear_cache",
                    priority=6,
                    auto_applicable=True,
                ),
            ],
            "TrainingDivergenceError": [
                RecoveryAction(
                    description="Reduce learning rate by factor of 10",
                    action_type="reduce_learning_rate",
                    priority=8,
                    auto_applicable=True,
                ),
                RecoveryAction(
                    description="Rollback to last stable checkpoint",
                    action_type="checkpoint_rollback",
                    priority=7,
                    auto_applicable=False,
                ),
                RecoveryAction(
                    description="Increase gradient accumulation steps",
                    action_type="increase_accumulation",
                    priority=5,
                    auto_applicable=True,
                ),
            ],
            "CheckpointError": [
                RecoveryAction(
                    description="Verify disk space and retry save",
                    action_type="check_disk_space",
                    priority=7,
                    auto_applicable=False,
                ),
                RecoveryAction(
                    description="Save to alternative location (Google Drive)",
                    action_type="save_to_drive",
                    priority=6,
                    auto_applicable=True,
                ),
            ],
            "DatasetDownloadError": [
                RecoveryAction(
                    description="Retry download with increased timeout",
                    action_type="retry_with_timeout",
                    priority=7,
                    auto_applicable=True,
                ),
                RecoveryAction(
                    description="Use cached/partial dataset if available",
                    action_type="use_cached",
                    priority=6,
                    auto_applicable=True,
                ),
                RecoveryAction(
                    description="Try alternative dataset source",
                    action_type="alternative_source",
                    priority=5,
                    auto_applicable=False,
                ),
            ],
            "ModelLoadingError": [
                RecoveryAction(
                    description="Re-download model from HuggingFace Hub",
                    action_type="redownload_model",
                    priority=7,
                    auto_applicable=True,
                ),
                RecoveryAction(
                    description="Verify model file integrity",
                    action_type="verify_model",
                    priority=6,
                    auto_applicable=True,
                ),
            ],
            "SessionTimeoutError": [
                RecoveryAction(
                    description="Save checkpoint immediately to Google Drive",
                    action_type="emergency_checkpoint",
                    priority=10,
                    auto_applicable=True,
                ),
                RecoveryAction(
                    description="Reduce remaining work to fit in time window",
                    action_type="reduce_workload",
                    priority=8,
                    auto_applicable=False,
                ),
            ],
            "DriveError": [
                RecoveryAction(
                    description="Remount Google Drive and retry",
                    action_type="remount_drive",
                    priority=7,
                    auto_applicable=True,
                ),
                RecoveryAction(
                    description="Save to local storage as fallback",
                    action_type="save_local",
                    priority=6,
                    auto_applicable=True,
                ),
            ],
        }


# =============================================================================
# Error Reporting Utilities
# =============================================================================


# Mapping of technical error types to user-friendly messages
_USER_FRIENDLY_MESSAGES: Dict[str, str] = {
    "TrainingOOMError": (
        "The GPU ran out of memory. Try reducing the batch size or "
        "enabling memory optimization options."
    ),
    "TrainingDivergenceError": (
        "Training became unstable. The model may need a lower learning rate "
        "or a checkpoint rollback."
    ),
    "CheckpointError": (
        "Failed to save or load a checkpoint. Check available disk space "
        "and file permissions."
    ),
    "DatasetDownloadError": (
        "Dataset download failed. Check your internet connection and try again."
    ),
    "DatasetExtractionError": (
        "Failed to extract condition maps from some images. "
        "The affected samples will be skipped."
    ),
    "DatasetValidationError": (
        "Some dataset samples are invalid or corrupted. "
        "Check the dataset report for details."
    ),
    "ModelLoadingError": (
        "Failed to load the model. The model file may be corrupted or "
        "incompatible. Try re-downloading."
    ),
    "GenerationError": (
        "Image generation failed. Try different parameters or check "
        "that the model is loaded correctly."
    ),
    "SessionTimeoutError": (
        "The Colab session is about to expire. Your progress has been "
        "saved automatically."
    ),
    "DriveError": (
        "Google Drive operation failed. Try remounting Drive or saving locally."
    ),
    "ColabEnvironmentError": (
        "A Colab environment issue occurred. Check GPU availability "
        "and session status."
    ),
}


def format_error_for_user(error: Exception) -> str:
    """Convert a technical error into a user-friendly message.

    Args:
        error: The exception to format.

    Returns:
        A clear, actionable message suitable for display to end users.
    """
    error_type = type(error).__name__

    # Check for a known user-friendly message
    friendly_msg = _USER_FRIENDLY_MESSAGES.get(error_type)
    if friendly_msg:
        return friendly_msg

    # For ControlNetPipelineError subclasses, use the message directly
    if isinstance(error, ControlNetPipelineError):
        return f"Pipeline error: {error}"

    # For unknown errors, provide a generic but helpful message
    return (
        f"An unexpected error occurred: {type(error).__name__}. "
        f"Please check the logs for more details."
    )


def create_error_report(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    include_traceback: bool = True,
) -> Dict[str, Any]:
    """Generate a detailed error report for debugging.

    Creates a structured report containing all relevant information
    for diagnosing and resolving the error.

    Args:
        error: The exception to report on.
        context: Additional context (e.g., training step, batch info).
        include_traceback: Whether to include the full stack trace.

    Returns:
        Dictionary containing the complete error report.
    """
    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "user_message": format_error_for_user(error),
        "context": context or {},
    }

    # Add details from ControlNetPipelineError subclasses
    if isinstance(error, ControlNetPipelineError):
        report["pipeline_details"] = error.details
        report["error_timestamp"] = error.timestamp

    # Add specific fields from known error types
    if isinstance(error, TrainingOOMError):
        report["gpu_usage_gb"] = error.current_usage_gb
        report["attempted_allocation_gb"] = error.attempted_allocation_gb
    elif isinstance(error, TrainingDivergenceError):
        report["loss_history"] = error.loss_history[-10:]  # Last 10 values
        report["divergence_threshold"] = error.divergence_threshold
    elif isinstance(error, DatasetDownloadError):
        report["retry_count"] = error.retry_count
        if error.last_error:
            report["last_error"] = str(error.last_error)
    elif isinstance(error, CheckpointError):
        report["checkpoint_path"] = error.checkpoint_path
    elif isinstance(error, ModelLoadingError):
        report["model_path"] = error.model_path
    elif isinstance(error, ColabEnvironmentError):
        report["time_remaining"] = error.time_remaining

    # Include traceback if requested
    if include_traceback:
        report["traceback"] = traceback.format_exc()

    return report
