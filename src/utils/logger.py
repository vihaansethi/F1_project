"""Single source of truth for project logging."""
from __future__ import annotations

import logging
import sys


def setup_logger(
    name: str = "f1ml",
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
) -> logging.Logger:
    """Return a configured logger; safe to call multiple times."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger
