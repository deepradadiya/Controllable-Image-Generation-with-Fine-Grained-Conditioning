"""
Logging and Debugging Utilities

This module provides a structured logging system with configurable levels,
debug mode with detailed execution tracing, log aggregation and analysis tools,
and utility decorators for function-level monitoring.

Requirements satisfied: 12.1, 12.2, 12.3
"""

import os
import sys
import json
import time
import logging
import traceback
import functools
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass, field

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# ANSI color codes for console output
COLORS = {
    "DEBUG": "\033[36m",      # Cyan
    "INFO": "\033[32m",       # Green
    "WARNING": "\033[33m",    # Yellow
    "ERROR": "\033[31m",      # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
}

# Component names for separate log files
COMPONENTS = ("training", "inference", "data", "evaluation", "app")


# ---------------------------------------------------------------------------
# Custom Formatters
# ---------------------------------------------------------------------------

class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes to log output for console display."""

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None,
                 use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            level_color = COLORS.get(record.levelname, COLORS["RESET"])
            record.levelname = (
                f"{level_color}{record.levelname}{COLORS['RESET']}"
            )
            record.msg = f"{level_color}{record.msg}{COLORS['RESET']}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """Formatter that outputs log records as JSON for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }
        # Include extra fields if present
        for key in ("component", "step", "duration_ms", "memory_mb"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry)


# ---------------------------------------------------------------------------
# setup_logging() function
# ---------------------------------------------------------------------------

def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "./logs",
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
    use_colors: bool = True,
    use_json: bool = False,
    max_file_size_mb: int = 100,
    backup_count: int = 5,
    component_logs: bool = True,
    log_filename: str = "controlnet_pipeline.log",
) -> logging.Logger:
    """
    Configure the project-wide logging system.

    Sets up console and file handlers with configurable levels, optional
    colored output, rotating file handlers, and separate log files for
    different pipeline components.

    Args:
        log_level: Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files.
        console_level: Console handler level (defaults to log_level).
        file_level: File handler level (defaults to DEBUG for full capture).
        use_colors: Enable ANSI colored console output.
        use_json: Use JSON formatting for file logs (machine-parseable).
        max_file_size_mb: Maximum size of each log file before rotation.
        backup_count: Number of rotated log files to keep.
        component_logs: Create separate log files per component.
        log_filename: Name of the main log file.

    Returns:
        The root logger configured for the pipeline.
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    root_level = LOG_LEVELS.get(log_level.upper(), logging.INFO)
    console_lvl = LOG_LEVELS.get((console_level or log_level).upper(), root_level)
    file_lvl = LOG_LEVELS.get((file_level or "DEBUG").upper(), logging.DEBUG)

    # Get or create the pipeline root logger
    root_logger = logging.getLogger("controlnet_pipeline")
    root_logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter

    # Remove existing handlers to avoid duplicates on re-init
    root_logger.handlers.clear()

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_lvl)
    if use_colors and sys.stdout.isatty():
        console_fmt = ColoredFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            use_colors=True,
        )
    else:
        console_fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # --- Main file handler with rotation ---
    main_log_path = log_dir_path / log_filename
    file_handler = RotatingFileHandler(
        main_log_path,
        maxBytes=max_file_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_lvl)
    if use_json:
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    root_logger.addHandler(file_handler)

    # --- Component-specific log files ---
    if component_logs:
        for component in COMPONENTS:
            comp_logger = logging.getLogger(f"controlnet_pipeline.{component}")
            comp_log_path = log_dir_path / f"{component}.log"
            comp_handler = RotatingFileHandler(
                comp_log_path,
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count,
                encoding="utf-8",
            )
            comp_handler.setLevel(file_lvl)
            if use_json:
                comp_handler.setFormatter(JSONFormatter())
            else:
                comp_handler.setFormatter(logging.Formatter(
                    fmt="%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                ))
            comp_logger.addHandler(comp_handler)

    root_logger.info(
        f"Logging initialized: level={log_level}, dir={log_dir_path}, "
        f"json={use_json}, components={component_logs}"
    )
    return root_logger


def get_logger(name: str, component: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module or component.

    Args:
        name: Logger name (typically __name__ of the calling module).
        component: Optional component name for routing to component log file.

    Returns:
        Configured logger instance.
    """
    if component and component in COMPONENTS:
        return logging.getLogger(f"controlnet_pipeline.{component}.{name}")
    return logging.getLogger(f"controlnet_pipeline.{name}")


# ---------------------------------------------------------------------------
# DebugMode class / context manager
# ---------------------------------------------------------------------------

@dataclass
class _TraceEntry:
    """Single function trace record."""
    function_name: str
    args_repr: str
    kwargs_repr: str
    start_time: float
    end_time: Optional[float] = None
    return_value_repr: Optional[str] = None
    exception_repr: Optional[str] = None
    memory_before_mb: Optional[float] = None
    memory_after_mb: Optional[float] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class DebugMode:
    """
    Debug mode context manager that enables detailed execution tracing.

    When activated, logs function entry/exit with arguments and return values,
    tracks execution time for each function call, and takes memory snapshots
    at key points. Can be enabled via environment variable DEBUG_MODE=1 or
    programmatically.

    Usage:
        with DebugMode() as dbg:
            result = my_function(arg1, arg2)
            dbg.log_checkpoint("after processing")

        # Or as a decorator enabler
        DebugMode.enable()
        ...
        DebugMode.disable()
    """

    _instance: Optional["DebugMode"] = None
    _enabled: bool = False
    _lock = threading.Lock()

    def __init__(
        self,
        enabled: Optional[bool] = None,
        log_args: bool = True,
        log_return: bool = True,
        log_memory: bool = True,
        max_repr_length: int = 200,
    ):
        """
        Initialize debug mode.

        Args:
            enabled: Force enable/disable. If None, checks DEBUG_MODE env var.
            log_args: Log function arguments on entry.
            log_return: Log return values on exit.
            log_memory: Take memory snapshots before/after calls.
            max_repr_length: Maximum length for repr strings of args/returns.
        """
        if enabled is None:
            enabled = os.environ.get("DEBUG_MODE", "0").lower() in ("1", "true", "yes")
        self._active = enabled
        self.log_args = log_args
        self.log_return = log_return
        self.log_memory = log_memory
        self.max_repr_length = max_repr_length
        self._traces: List[_TraceEntry] = []
        self._checkpoints: List[Dict[str, Any]] = []
        self._logger = logging.getLogger("controlnet_pipeline.debug")

    def __enter__(self) -> "DebugMode":
        with self._lock:
            DebugMode._instance = self
            DebugMode._enabled = self._active
        if self._active:
            self._logger.debug("DebugMode ENABLED - detailed tracing active")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        with self._lock:
            DebugMode._enabled = False
            DebugMode._instance = None
        if self._active:
            self._logger.debug(
                f"DebugMode DISABLED - captured {len(self._traces)} traces, "
                f"{len(self._checkpoints)} checkpoints"
            )
        return None

    @classmethod
    def enable(cls) -> None:
        """Enable debug mode globally."""
        with cls._lock:
            cls._enabled = True

    @classmethod
    def disable(cls) -> None:
        """Disable debug mode globally."""
        with cls._lock:
            cls._enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if debug mode is currently active."""
        return cls._enabled

    def log_checkpoint(self, label: str) -> None:
        """
        Record a named checkpoint with current memory state.

        Args:
            label: Descriptive label for this checkpoint.
        """
        checkpoint = {
            "label": label,
            "timestamp": time.time(),
            "memory_mb": self._get_memory_mb(),
        }
        self._checkpoints.append(checkpoint)
        self._logger.debug(
            f"CHECKPOINT [{label}]: memory={checkpoint['memory_mb']:.1f}MB"
        )

    def trace_function(self, func: Callable) -> Callable:
        """
        Decorator that traces function execution when debug mode is active.

        Args:
            func: Function to trace.

        Returns:
            Wrapped function with tracing.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not DebugMode._enabled:
                return func(*args, **kwargs)

            entry = _TraceEntry(
                function_name=f"{func.__module__}.{func.__qualname__}",
                args_repr=self._safe_repr(args),
                kwargs_repr=self._safe_repr(kwargs),
                start_time=time.time(),
                memory_before_mb=self._get_memory_mb() if self.log_memory else None,
            )

            self._logger.debug(
                f"ENTER {entry.function_name}"
                + (f" args={entry.args_repr}" if self.log_args else "")
                + (f" kwargs={entry.kwargs_repr}" if self.log_args and kwargs else "")
            )

            try:
                result = func(*args, **kwargs)
                entry.end_time = time.time()
                entry.return_value_repr = self._safe_repr(result) if self.log_return else None
                entry.memory_after_mb = self._get_memory_mb() if self.log_memory else None

                self._logger.debug(
                    f"EXIT  {entry.function_name} "
                    f"duration={entry.duration_ms:.2f}ms"
                    + (f" return={entry.return_value_repr}" if self.log_return else "")
                    + (f" mem_delta={entry.memory_after_mb - entry.memory_before_mb:.1f}MB"
                       if self.log_memory and entry.memory_before_mb is not None
                       and entry.memory_after_mb is not None else "")
                )
                self._traces.append(entry)
                return result
            except Exception as e:
                entry.end_time = time.time()
                entry.exception_repr = f"{type(e).__name__}: {e}"
                entry.memory_after_mb = self._get_memory_mb() if self.log_memory else None
                self._logger.debug(
                    f"EXCEPTION {entry.function_name} "
                    f"duration={entry.duration_ms:.2f}ms "
                    f"error={entry.exception_repr}"
                )
                self._traces.append(entry)
                raise

        return wrapper

    def get_traces(self) -> List[Dict[str, Any]]:
        """Get all recorded trace entries as dictionaries."""
        return [
            {
                "function": t.function_name,
                "duration_ms": t.duration_ms,
                "memory_before_mb": t.memory_before_mb,
                "memory_after_mb": t.memory_after_mb,
                "exception": t.exception_repr,
            }
            for t in self._traces
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all traced executions."""
        if not self._traces:
            return {"total_traces": 0}

        durations = [t.duration_ms for t in self._traces]
        errors = [t for t in self._traces if t.exception_repr is not None]
        return {
            "total_traces": len(self._traces),
            "total_duration_ms": sum(durations),
            "avg_duration_ms": sum(durations) / len(durations),
            "max_duration_ms": max(durations),
            "error_count": len(errors),
            "checkpoints": len(self._checkpoints),
        }

    def _safe_repr(self, obj: Any) -> str:
        """Get a truncated repr of an object."""
        try:
            r = repr(obj)
        except Exception:
            r = "<repr failed>"
        if len(r) > self.max_repr_length:
            return r[: self.max_repr_length] + "..."
        return r

    @staticmethod
    def _get_memory_mb() -> float:
        """Get current process memory usage in MB."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 ** 2)
        return 0.0


# ---------------------------------------------------------------------------
# LogAnalyzer class
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    """Parsed log entry."""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line: Optional[int] = None
    raw: str = ""


class LogAnalyzer:
    """
    Analyze log files to extract metrics, identify error patterns,
    and generate summary reports.

    Supports parsing both plain-text and JSON-formatted log files,
    filtering by component, level, or time range, and generating
    frequency-based reports.
    """

    def __init__(self, log_dir: str = "./logs"):
        """
        Initialize the log analyzer.

        Args:
            log_dir: Directory containing log files to analyze.
        """
        self.log_dir = Path(log_dir)
        self._entries: List[LogEntry] = []

    def parse_log_file(self, filepath: Union[str, Path]) -> List[LogEntry]:
        """
        Parse a log file and return structured entries.

        Supports both plain-text format and JSON format.

        Args:
            filepath: Path to the log file.

        Returns:
            List of parsed LogEntry objects.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return []

        entries: List[LogEntry] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = self._parse_line(line)
                if entry is not None:
                    entries.append(entry)

        self._entries.extend(entries)
        return entries

    def parse_all_logs(self) -> List[LogEntry]:
        """Parse all .log files in the log directory."""
        self._entries.clear()
        if not self.log_dir.exists():
            return []
        for log_file in sorted(self.log_dir.glob("*.log")):
            self.parse_log_file(log_file)
        return self._entries

    def filter_entries(
        self,
        level: Optional[str] = None,
        component: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        message_contains: Optional[str] = None,
    ) -> List[LogEntry]:
        """
        Filter parsed log entries by various criteria.

        Args:
            level: Filter by log level (e.g., "ERROR").
            component: Filter by component name in logger.
            start_time: Include entries after this time.
            end_time: Include entries before this time.
            message_contains: Filter by substring in message.

        Returns:
            Filtered list of LogEntry objects.
        """
        results = self._entries

        if level:
            results = [e for e in results if e.level == level.upper()]
        if component:
            results = [e for e in results if component in e.logger_name]
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]
        if message_contains:
            results = [e for e in results if message_contains in e.message]

        return results

    def get_error_patterns(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Identify the most common error patterns.

        Groups errors by their message prefix (first 80 chars) and
        returns frequency counts.

        Args:
            top_n: Number of top patterns to return.

        Returns:
            List of dicts with pattern, count, and example entries.
        """
        error_entries = [e for e in self._entries if e.level in ("ERROR", "CRITICAL")]
        pattern_counts: Dict[str, List[LogEntry]] = defaultdict(list)

        for entry in error_entries:
            # Use first 80 chars as pattern key
            pattern_key = entry.message[:80]
            pattern_counts[pattern_key].append(entry)

        sorted_patterns = sorted(
            pattern_counts.items(), key=lambda x: len(x[1]), reverse=True
        )

        return [
            {
                "pattern": pattern,
                "count": len(entries),
                "first_seen": min(e.timestamp for e in entries).isoformat(),
                "last_seen": max(e.timestamp for e in entries).isoformat(),
                "example_message": entries[0].message,
            }
            for pattern, entries in sorted_patterns[:top_n]
        ]

    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of all parsed logs.

        Returns:
            Dictionary with level counts, errors per hour, most common
            warnings, and component activity breakdown.
        """
        if not self._entries:
            return {"status": "no_entries", "total": 0}

        # Level distribution
        level_counts: Dict[str, int] = defaultdict(int)
        for entry in self._entries:
            level_counts[entry.level] += 1

        # Time range
        timestamps = [e.timestamp for e in self._entries]
        time_range_hours = max(
            (max(timestamps) - min(timestamps)).total_seconds() / 3600, 0.001
        )

        # Errors per hour
        error_count = level_counts.get("ERROR", 0) + level_counts.get("CRITICAL", 0)
        errors_per_hour = error_count / time_range_hours

        # Component breakdown
        component_counts: Dict[str, int] = defaultdict(int)
        for entry in self._entries:
            parts = entry.logger_name.split(".")
            comp = parts[1] if len(parts) > 1 else "root"
            component_counts[comp] += 1

        # Most common warnings
        warnings = [e for e in self._entries if e.level == "WARNING"]
        warning_msgs: Dict[str, int] = defaultdict(int)
        for w in warnings:
            warning_msgs[w.message[:100]] += 1
        top_warnings = sorted(warning_msgs.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_entries": len(self._entries),
            "time_range": {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
                "hours": round(time_range_hours, 2),
            },
            "level_distribution": dict(level_counts),
            "errors_per_hour": round(errors_per_hour, 2),
            "top_warnings": [
                {"message": msg, "count": count} for msg, count in top_warnings
            ],
            "component_activity": dict(component_counts),
            "error_patterns": self.get_error_patterns(top_n=5),
        }

    def _parse_line(self, line: str) -> Optional[LogEntry]:
        """Parse a single log line (JSON or plain-text format)."""
        # Try JSON first
        if line.startswith("{"):
            return self._parse_json_line(line)
        return self._parse_text_line(line)

    def _parse_json_line(self, line: str) -> Optional[LogEntry]:
        """Parse a JSON-formatted log line."""
        try:
            data = json.loads(line)
            timestamp = datetime.fromisoformat(data.get("timestamp", ""))
            return LogEntry(
                timestamp=timestamp,
                level=data.get("level", "INFO"),
                logger_name=data.get("logger", ""),
                message=data.get("message", ""),
                module=data.get("module"),
                function=data.get("function"),
                line=data.get("line"),
                raw=line,
            )
        except (json.JSONDecodeError, ValueError):
            return None

    def _parse_text_line(self, line: str) -> Optional[LogEntry]:
        """Parse a plain-text formatted log line."""
        # Expected format: "2024-01-01 12:00:00 | INFO     | logger | message"
        parts = line.split(" | ", maxsplit=3)
        if len(parts) < 4:
            return None
        try:
            timestamp = datetime.strptime(parts[0].strip(), "%Y-%m-%d %H:%M:%S")
            level = parts[1].strip()
            logger_name = parts[2].strip()
            message = parts[3].strip()

            # Try to extract function:line from logger field
            func_info = None
            line_no = None
            if ":" in parts[2] and len(parts) > 3:
                # Format might be "funcName:lineno | message"
                pass

            return LogEntry(
                timestamp=timestamp,
                level=level,
                logger_name=logger_name,
                message=message,
                function=func_info,
                line=line_no,
                raw=line,
            )
        except (ValueError, IndexError):
            return None


# ---------------------------------------------------------------------------
# Utility Decorators
# ---------------------------------------------------------------------------

def log_execution(
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    log_args: bool = False,
    log_result: bool = False,
) -> Callable:
    """
    Decorator that logs function calls with execution timing.

    Logs function entry, exit, duration, and optionally arguments and
    return values.

    Args:
        logger: Logger instance to use (defaults to module logger).
        level: Log level for the messages.
        log_args: Whether to log function arguments.
        log_result: Whether to log the return value.

    Returns:
        Decorator function.

    Usage:
        @log_execution()
        def train_step(batch):
            ...

        @log_execution(log_args=True, log_result=True)
        def compute_loss(predictions, targets):
            ...
    """
    def decorator(func: Callable) -> Callable:
        _logger = logger or logging.getLogger(
            f"controlnet_pipeline.{func.__module__}"
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            start = time.time()

            msg_parts = [f"Calling {func_name}"]
            if log_args:
                args_str = ", ".join(
                    [repr(a)[:100] for a in args]
                    + [f"{k}={repr(v)[:100]}" for k, v in kwargs.items()]
                )
                msg_parts.append(f"args=({args_str})")
            _logger.log(level, " ".join(msg_parts))

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000

                exit_msg = f"Completed {func_name} in {duration_ms:.2f}ms"
                if log_result:
                    exit_msg += f" result={repr(result)[:200]}"
                _logger.log(level, exit_msg)

                return result
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                _logger.error(
                    f"Failed {func_name} after {duration_ms:.2f}ms: "
                    f"{type(e).__name__}: {e}"
                )
                raise

        return wrapper
    return decorator


def trace_memory(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
) -> Callable:
    """
    Decorator that logs memory usage before and after function execution.

    Reports GPU memory (if available) or process RSS memory, along with
    the delta caused by the function call.

    Args:
        logger: Logger instance to use.
        level: Log level for memory messages.

    Returns:
        Decorator function.

    Usage:
        @trace_memory()
        def load_model():
            ...
    """
    def decorator(func: Callable) -> Callable:
        _logger = logger or logging.getLogger(
            f"controlnet_pipeline.{func.__module__}"
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            mem_before = _get_current_memory_mb()

            _logger.log(
                level,
                f"[MEMORY] Before {func_name}: {mem_before:.1f}MB"
            )

            result = func(*args, **kwargs)

            mem_after = _get_current_memory_mb()
            delta = mem_after - mem_before
            sign = "+" if delta >= 0 else ""

            _logger.log(
                level,
                f"[MEMORY] After {func_name}: {mem_after:.1f}MB "
                f"(delta: {sign}{delta:.1f}MB)"
            )

            return result

        return wrapper
    return decorator


def _get_current_memory_mb() -> float:
    """Get current memory usage in MB (GPU preferred, then process RSS)."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)
    return 0.0
