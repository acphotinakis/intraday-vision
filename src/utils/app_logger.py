import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import json
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.syntax import Syntax
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class LoggerTheme:
    panel_border: str = "red"
    info_color: str = "cyan"
    warning_color: str = "yellow"
    error_color: str = "bold red"


console = Console()


class AppLogger:
    """Reusable logger using Rich for console output + optional file logging."""

    def __init__(
        self,
        name: str,
        log_dir: Optional[Path] = None,
        filename: Optional[str] = None,
        level: int = logging.INFO,
        json_format: bool = False,  # still supports JSON for files
        rotate: bool = True,
        when: str = "midnight",
        backup_count: int = 7,
        clear_existing: bool = True,
        theme: LoggerTheme | None = None,
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # avoid double logging

        self.theme = theme or LoggerTheme()

        # File logging setup
        self.log_dir = log_dir or Path("./logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = filename or f"{name}.log"
        log_path = self.log_dir / log_file

        if clear_existing and log_path.exists():
            log_path.unlink()

        # File handler (optional rotation)
        if rotate:
            file_handler = TimedRotatingFileHandler(
                filename=log_path, when=when, backupCount=backup_count, utc=True
            )
        else:
            file_handler = logging.FileHandler(log_path)

        # File formatter (JSON or plain)
        if json_format:
            file_formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s"}'
            )
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Rich console handler
        console_handler = RichHandler(
            console=Console(),
            markup=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
        self.logger.addHandler(console_handler)

    # @staticmethod
    def _format_msg(self, msg: str, extra: Optional[dict] = None) -> str:
        """Return the message text only for logging, render extra separately."""
        if extra:
            # Render extra metadata as a Rich Panel directly to console
            json_str = json.dumps(extra, indent=2, default=str)
            # panel = Panel(
            #     json_str, title="Extra Metadata", expand=False, border_style="red"
            # )
            panel = Panel(
                json_str,
                title="Extra Metadata",
                border_style=self.theme.panel_border,
            )

            console.print(panel)  # render the panel immediately
        return msg  # log only the plain message

    # def _format_msg(msg: str, extra: Optional[dict] = None) -> str:
    #     """Merge extra dict into message for logging."""
    #     if extra:
    #         # Pretty print extra using JSON syntax highlighting
    #         json_str = json.dumps(extra, indent=2, default=str)
    #         syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
    #         panel = Panel(
    #             syntax, title="Extra Metadata", expand=False, border_style="red"
    #         )
    #         return f"{msg}\n{panel}"
    #     return msg

    # Logging convenience methods
    def info(self, msg: str, extra: Optional[dict] = None):
        self.logger.info(self._format_msg(msg, extra))

    def warning(self, msg: str, extra: Optional[dict] = None):
        self.logger.warning(self._format_msg(msg, extra))

    def error(self, msg: str, extra: Optional[dict] = None):
        self.logger.error(self._format_msg(msg, extra))

    def debug(self, msg: str, extra: Optional[dict] = None):
        self.logger.debug(self._format_msg(msg, extra))
