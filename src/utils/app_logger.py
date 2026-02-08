import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import json
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.syntax import Syntax
from dataclasses import dataclass
from datetime import datetime
from rich.console import Group


# ===== Theme for styling =====
@dataclass(frozen=True)
class LoggerTheme:
    panel_border: str = "red"
    info_color: str = "cyan"
    warning_color: str = "yellow"
    error_color: str = "bold red"
    debug_color: str = "magenta"
    header_bg: str = "dark_blue"
    footer_bg: str = "dark_green"
    panel_bg: str = "grey11"


# Icons for each log level
LEVEL_STYLES = {
    "info": {"color": "cyan", "icon": "ℹ️"},
    "warning": {"color": "yellow", "icon": "⚠️"},
    "error": {"color": "bold red", "icon": "❌"},
    "debug": {"color": "magenta", "icon": "🐛"},
}


console = Console()


class AppLogger:
    """Enhanced logger using Rich for console + optional file logging with JSON support."""

    def __init__(
        self,
        name: str,
        log_dir: Optional[Path] = None,
        filename: Optional[str] = None,
        level: int = logging.INFO,
        json_format: bool = False,  # file logging can still be JSON
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

        # ===== File logging setup =====
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

        # File formatter
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

        # ===== Rich console handler =====
        console_handler = RichHandler(
            console=Console(),
            markup=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
        self.logger.addHandler(console_handler)

    # ===== Helper to format message with extra metadata =====
    def _format_msg(self, msg: str, extra: Optional[dict] = None) -> str:
        """Return plain message for file logging; print Rich panels for console if extra is provided."""
        if extra:
            json_str = json.dumps(extra, indent=2, default=str)
            syntax = Syntax(json_str, "json", theme="monokai", word_wrap=True)
            panel = Panel(
                syntax,
                title="Extra Metadata",
                border_style=self.theme.panel_border,
                expand=False,
                style=self.theme.panel_bg,
            )
            console.print(panel)
        return msg

    # ===== Block logging with panels =====
    def info_block(
        self,
        messages: list[str],
        header: Optional[str] = None,
        footer: Optional[str] = None,
        levels: Optional[list[str]] = None,
        extra: Optional[dict] = None,
    ):
        if levels and len(levels) != len(messages):
            raise ValueError("Length of levels must match length of messages")

        # Create a list of renderables (strings + Syntax)
        panel_renderables = []

        # Timestamp
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        panel_renderables.append(f"[dim]{timestamp}[/dim]")

        # Header
        if header:
            panel_renderables.append(
                f"[bold white on {self.theme.header_bg}]{header}[/bold white on {self.theme.header_bg}]"
            )

        # Messages
        for i, msg in enumerate(messages):
            level = levels[i] if levels else "info"
            style = LEVEL_STYLES.get(level.lower(), LEVEL_STYLES["info"])
            panel_renderables.append(
                f"{style['icon']} [{style['color']}]{msg}[/{style['color']}]"
            )

        # Extra metadata (as Syntax)
        if extra:
            json_str = json.dumps(extra, indent=2, default=str)
            syntax = Syntax(json_str, "json", theme="monokai", word_wrap=True)
            panel_renderables.append(syntax)

        # Footer
        if footer:
            panel_renderables.append(
                f"[bold white on {self.theme.footer_bg}]{footer}[/bold white on {self.theme.footer_bg}]"
            )

        # Use Group to combine renderables
        panel = Panel(
            Group(*panel_renderables),
            border_style=self.theme.panel_border,
            expand=False,
            style=self.theme.panel_bg,
        )
        console.print(panel)

        # Also log plain messages + extra to file
        log_text = " | ".join(messages)
        if extra:
            log_text += " | EXTRA: " + json.dumps(extra, default=str)
        for i, msg in enumerate(messages):
            level = levels[i] if levels else "info"
            getattr(self.logger, level.lower())(self._format_msg(msg, extra=None))

    # ===== Convenience logging methods =====
    def info(self, msg: str, extra: Optional[dict] = None):
        self.logger.info(self._format_msg(msg, extra))

    def warning(self, msg: str, extra: Optional[dict] = None):
        self.logger.warning(self._format_msg(msg, extra))

    def error(self, msg: str, extra: Optional[dict] = None):
        self.logger.error(self._format_msg(msg, extra))

    def debug(self, msg: str, extra: Optional[dict] = None):
        self.logger.debug(self._format_msg(msg, extra))
