"""Unified logging helpers.

All modules obtain loggers via :func:`get_logger` so that formatting, level and
optional file sinks are controlled in one place (``setup_logging``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT_NAME = "msp"
_CONFIGURED = False


def setup_logging(level: int | str = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """Configure the package root logger.

    Args:
        level: Logging level (int or name such as ``"DEBUG"``).
        log_file: Optional file path; when given, records are also appended to this file.

    Returns:
        The configured root logger for the package.
    """
    global _CONFIGURED
    root = logging.getLogger(_ROOT_NAME)
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    root.setLevel(level)

    if not _CONFIGURED:
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(fmt)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        already = any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve() for h in root.handlers)
        if not already:
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
            root.addHandler(fh)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the package namespace.

    Args:
        name: Usually ``__name__`` of the calling module.
    """
    if not _CONFIGURED:
        setup_logging()
    short = name.replace("mall_space_planner.", "")
    return logging.getLogger(f"{_ROOT_NAME}.{short}")
