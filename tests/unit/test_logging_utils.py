"""
Unit tests for logging and debugging utilities.

Tests cover:
- setup_logging() with various configurations
- DebugMode context manager and tracing
- LogAnalyzer parsing and filtering
- log_execution decorator
- trace_memory decorator
"""

import json
import logging
import tempfile
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import (
    setup_logging,
    get_logger,
    DebugMode,
    LogAnalyzer,
    LogEntry,
    log_execution,
    trace_memory,
    ColoredFormatter,
    JSONFormatter,
    COMPONENTS,
)


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_returns_logger(self):
        """setup_logging returns a configured Logger instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(log_dir=tmpdir, use_colors=False)
            assert isinstance(logger, logging.Logger)
            assert logger.name == "controlnet_pipeline"

    def test_creates_log_directory(self):
        """setup_logging creates the log directory if it does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "nested" / "logs"
            setup_logging(log_dir=str(log_dir), use_colors=False)
            assert log_dir.exists()

    def test_creates_component_log_files(self):
        """setup_logging creates separate log files for each component."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir, use_colors=False, component_logs=True)
            for component in COMPONENTS:
                comp_logger = get_logger("test", component=component)
                comp_logger.info(f"Test message for {component}")

            # Verify component log files exist
            for component in COMPONENTS:
                log_file = Path(tmpdir) / f"{component}.log"
                assert log_file.exists(), f"Missing log file for {component}"

    def test_json_format_output(self):
        """setup_logging with use_json=True produces valid JSON log lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(
                log_dir=tmpdir, use_colors=False, use_json=True
            )
            logger.info("JSON test message")

            main_log = Path(tmpdir) / "controlnet_pipeline.log"
            assert main_log.exists()
            content = main_log.read_text().strip()
            # Parse each line as JSON
            for line in content.split("\n"):
                if line.strip():
                    entry = json.loads(line)
                    assert "timestamp" in entry
                    assert "level" in entry
                    assert "message" in entry

    def test_log_level_filtering(self):
        """setup_logging respects the configured log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(
                log_level="WARNING", log_dir=tmpdir, use_colors=False
            )
            # Console handler should filter INFO
            console_handler = [
                h for h in logger.handlers
                if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
            ]
            assert len(console_handler) == 1
            assert console_handler[0].level == logging.WARNING

    def test_custom_log_filename(self):
        """setup_logging uses the specified log filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(
                log_dir=tmpdir, log_filename="custom.log", use_colors=False
            )
            assert (Path(tmpdir) / "custom.log").exists()


class TestGetLogger:
    """Tests for the get_logger helper."""

    def test_returns_namespaced_logger(self):
        """get_logger returns a logger under the pipeline namespace."""
        logger = get_logger("my_module")
        assert logger.name == "controlnet_pipeline.my_module"

    def test_component_logger_namespace(self):
        """get_logger with component routes to component namespace."""
        logger = get_logger("my_module", component="training")
        assert logger.name == "controlnet_pipeline.training.my_module"

    def test_invalid_component_falls_back(self):
        """get_logger with invalid component uses default namespace."""
        logger = get_logger("my_module", component="nonexistent")
        assert logger.name == "controlnet_pipeline.my_module"


class TestDebugMode:
    """Tests for the DebugMode context manager."""

    def test_context_manager_enables_and_disables(self):
        """DebugMode enables on enter and disables on exit."""
        assert not DebugMode.is_enabled()
        with DebugMode(enabled=True):
            assert DebugMode.is_enabled()
        assert not DebugMode.is_enabled()

    def test_disabled_mode_skips_tracing(self):
        """When disabled, trace_function does not record traces."""
        with DebugMode(enabled=False) as dbg:
            @dbg.trace_function
            def noop():
                return 42

            result = noop()
            assert result == 42
            assert len(dbg.get_traces()) == 0

    def test_traces_function_calls(self):
        """DebugMode records function entry/exit with timing."""
        with DebugMode(enabled=True, log_memory=False) as dbg:
            @dbg.trace_function
            def add(a, b):
                time.sleep(0.005)
                return a + b

            result = add(3, 7)
            assert result == 10

        traces = dbg.get_traces()
        assert len(traces) == 1
        assert traces[0]["duration_ms"] >= 5
        assert traces[0]["exception"] is None

    def test_traces_exceptions(self):
        """DebugMode records exceptions in traced functions."""
        with DebugMode(enabled=True, log_memory=False) as dbg:
            @dbg.trace_function
            def fail():
                raise RuntimeError("boom")

            try:
                fail()
            except RuntimeError:
                pass

        traces = dbg.get_traces()
        assert len(traces) == 1
        assert "RuntimeError" in traces[0]["exception"]

    def test_checkpoint_recording(self):
        """DebugMode records named checkpoints."""
        with DebugMode(enabled=True, log_memory=False) as dbg:
            @dbg.trace_function
            def noop():
                return 1

            noop()
            dbg.log_checkpoint("start")
            dbg.log_checkpoint("end")

        summary = dbg.get_summary()
        assert summary["checkpoints"] == 2

    def test_get_summary(self):
        """get_summary returns correct aggregate statistics."""
        with DebugMode(enabled=True, log_memory=False) as dbg:
            @dbg.trace_function
            def work():
                time.sleep(0.001)
                return True

            work()
            work()

        summary = dbg.get_summary()
        assert summary["total_traces"] == 2
        assert summary["error_count"] == 0
        assert summary["total_duration_ms"] > 0

    def test_class_level_enable_disable(self):
        """DebugMode.enable() and .disable() work globally."""
        DebugMode.enable()
        assert DebugMode.is_enabled()
        DebugMode.disable()
        assert not DebugMode.is_enabled()


class TestLogAnalyzer:
    """Tests for the LogAnalyzer class."""

    def _create_log_file(self, tmpdir: str, filename: str, content: str) -> Path:
        path = Path(tmpdir) / filename
        path.write_text(content)
        return path

    def test_parse_text_log(self):
        """LogAnalyzer parses plain-text log format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = (
                "2024-01-15 10:00:00 | INFO     | root | Hello world\n"
                "2024-01-15 10:00:01 | ERROR    | root | Something failed\n"
            )
            self._create_log_file(tmpdir, "test.log", content)

            analyzer = LogAnalyzer(log_dir=tmpdir)
            entries = analyzer.parse_all_logs()
            assert len(entries) == 2
            assert entries[0].level == "INFO"
            assert entries[1].level == "ERROR"
            assert entries[0].message == "Hello world"

    def test_parse_json_log(self):
        """LogAnalyzer parses JSON-formatted log lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lines = [
                json.dumps({
                    "timestamp": "2024-01-15T10:00:00",
                    "level": "WARNING",
                    "logger": "test.module",
                    "message": "Low memory",
                    "module": "mod",
                    "function": "fn",
                    "line": 42,
                }),
            ]
            self._create_log_file(tmpdir, "json.log", "\n".join(lines))

            analyzer = LogAnalyzer(log_dir=tmpdir)
            entries = analyzer.parse_all_logs()
            assert len(entries) == 1
            assert entries[0].level == "WARNING"
            assert entries[0].message == "Low memory"
            assert entries[0].line == 42

    def test_filter_by_level(self):
        """filter_entries correctly filters by log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = (
                "2024-01-15 10:00:00 | INFO     | root | msg1\n"
                "2024-01-15 10:00:01 | ERROR    | root | msg2\n"
                "2024-01-15 10:00:02 | INFO     | root | msg3\n"
            )
            self._create_log_file(tmpdir, "test.log", content)

            analyzer = LogAnalyzer(log_dir=tmpdir)
            analyzer.parse_all_logs()
            errors = analyzer.filter_entries(level="ERROR")
            assert len(errors) == 1
            assert errors[0].message == "msg2"

    def test_filter_by_component(self):
        """filter_entries correctly filters by component name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = (
                "2024-01-15 10:00:00 | INFO     | pipeline.training | train msg\n"
                "2024-01-15 10:00:01 | INFO     | pipeline.data | data msg\n"
            )
            self._create_log_file(tmpdir, "test.log", content)

            analyzer = LogAnalyzer(log_dir=tmpdir)
            analyzer.parse_all_logs()
            training = analyzer.filter_entries(component="training")
            assert len(training) == 1
            assert "train msg" in training[0].message

    def test_error_patterns_grouping(self):
        """get_error_patterns groups identical error messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = (
                "2024-01-15 10:00:00 | ERROR    | root | OOM error\n"
                "2024-01-15 10:00:01 | ERROR    | root | OOM error\n"
                "2024-01-15 10:00:02 | ERROR    | root | OOM error\n"
                "2024-01-15 10:00:03 | ERROR    | root | Timeout error\n"
            )
            self._create_log_file(tmpdir, "test.log", content)

            analyzer = LogAnalyzer(log_dir=tmpdir)
            analyzer.parse_all_logs()
            patterns = analyzer.get_error_patterns()
            assert len(patterns) == 2
            # OOM should be first (most frequent)
            assert patterns[0]["count"] == 3
            assert patterns[1]["count"] == 1

    def test_summary_report(self):
        """get_summary_report returns correct statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = (
                "2024-01-15 10:00:00 | INFO     | root | msg1\n"
                "2024-01-15 10:00:01 | WARNING  | root | warn1\n"
                "2024-01-15 10:00:02 | ERROR    | root | err1\n"
                "2024-01-15 10:30:00 | INFO     | root | msg2\n"
            )
            self._create_log_file(tmpdir, "test.log", content)

            analyzer = LogAnalyzer(log_dir=tmpdir)
            analyzer.parse_all_logs()
            report = analyzer.get_summary_report()

            assert report["total_entries"] == 4
            assert report["level_distribution"]["INFO"] == 2
            assert report["level_distribution"]["WARNING"] == 1
            assert report["level_distribution"]["ERROR"] == 1
            assert report["errors_per_hour"] > 0

    def test_empty_log_directory(self):
        """LogAnalyzer handles empty log directory gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = LogAnalyzer(log_dir=tmpdir)
            entries = analyzer.parse_all_logs()
            assert entries == []

    def test_nonexistent_directory(self):
        """LogAnalyzer handles nonexistent directory gracefully."""
        analyzer = LogAnalyzer(log_dir="/nonexistent/path")
        entries = analyzer.parse_all_logs()
        assert entries == []


class TestLogExecutionDecorator:
    """Tests for the @log_execution decorator."""

    def test_preserves_return_value(self):
        """Decorated function returns the correct value."""
        @log_execution()
        def multiply(a, b):
            return a * b

        assert multiply(3, 4) == 12

    def test_preserves_exceptions(self):
        """Decorated function re-raises exceptions."""
        @log_execution()
        def fail():
            raise ValueError("test")

        try:
            fail()
            assert False, "Should have raised"
        except ValueError as e:
            assert str(e) == "test"

    def test_preserves_function_metadata(self):
        """Decorated function preserves __name__ and __doc__."""
        @log_execution()
        def documented_func():
            """This is documented."""
            pass

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is documented."


class TestTraceMemoryDecorator:
    """Tests for the @trace_memory decorator."""

    def test_preserves_return_value(self):
        """Decorated function returns the correct value."""
        @trace_memory()
        def create_list():
            return [1, 2, 3]

        assert create_list() == [1, 2, 3]

    def test_preserves_function_metadata(self):
        """Decorated function preserves __name__."""
        @trace_memory()
        def my_func():
            pass

        assert my_func.__name__ == "my_func"


class TestColoredFormatter:
    """Tests for the ColoredFormatter."""

    def test_adds_colors_when_enabled(self):
        """ColoredFormatter adds ANSI codes when use_colors=True."""
        formatter = ColoredFormatter(
            fmt="%(levelname)s %(message)s", use_colors=True
        )
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "\033[" in output  # Contains ANSI escape

    def test_no_colors_when_disabled(self):
        """ColoredFormatter produces plain text when use_colors=False."""
        formatter = ColoredFormatter(
            fmt="%(levelname)s %(message)s", use_colors=False
        )
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "\033[" not in output


class TestJSONFormatter:
    """Tests for the JSONFormatter."""

    def test_produces_valid_json(self):
        """JSONFormatter output is valid JSON with expected fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger", level=logging.ERROR, pathname="test.py",
            lineno=42, msg="error occurred", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "ERROR"
        assert data["message"] == "error occurred"
        assert data["logger"] == "test.logger"
        assert data["line"] == 42

    def test_includes_exception_info(self):
        """JSONFormatter includes exception details when present."""
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="",
            lineno=0, msg="failed", args=(), exc_info=exc_info,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert "test error" in data["exception"]["message"]
