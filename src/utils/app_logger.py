import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import json
from typing import Optional


class AppLogger:
    """Dynamic reusable logger with file + console output, optionally JSON structured."""

    def __init__(
        self,
        name: str,
        log_dir: Optional[Path] = None,
        filename: Optional[str] = None,
        level: int = logging.INFO,
        json_format: bool = False,
        rotate: bool = True,
        when: str = "midnight",
        backup_count: int = 7,
        clear_existing: bool = True,
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        self.log_dir = log_dir or Path("./logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = filename or f"{name}.log"
        log_path = self.log_dir / log_file

        if clear_existing and log_path.exists():
            log_path.unlink()

        if json_format:
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        if rotate:
            file_handler = TimedRotatingFileHandler(
                filename=log_path, when=when, backupCount=backup_count, utc=True
            )
        else:
            file_handler = logging.FileHandler(log_path)

        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    # Helper to merge extra into message
    @staticmethod
    def _format_msg(msg: str, extra: Optional[dict] = None) -> str:
        if extra:
            return f"{msg} | {json.dumps(extra, default=str)}"
        return msg

    # Convenience methods
    def info(self, msg: str, extra: Optional[dict] = None):
        self.logger.info(self._format_msg(msg, extra))

    def warning(self, msg: str, extra: Optional[dict] = None):
        self.logger.warning(self._format_msg(msg, extra))

    def error(self, msg: str, extra: Optional[dict] = None):
        self.logger.error(self._format_msg(msg, extra))

    def debug(self, msg: str, extra: Optional[dict] = None):
        self.logger.debug(self._format_msg(msg, extra))
