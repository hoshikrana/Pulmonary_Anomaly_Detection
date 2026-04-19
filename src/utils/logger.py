"""
src/utils/logger.py
───────────────────
Structured logging for the entire project.

Production code never uses bare print() statements.
This logger writes to both console and a log file simultaneously,
with timestamps and severity levels — essential for long training
runs on Colab where you want a persistent record.
"""

import logging
import os
import sys
from datetime import datetime


def get_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    Create (or retrieve) a named logger that writes to console + file.

    Args:
        name    : Logger name, typically __name__ of the calling module.
        log_dir : Directory to save the .log file. If None, file logging
                  is skipped (useful for lightweight scripts).

    Returns:
        logging.Logger instance.

    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Training started.")
        logger.warning("Val loss not improving.")
        logger.error("Checkpoint save failed.")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler — DEBUG and above (captures everything)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path  = os.path.join(log_dir, f"{timestamp}_{name.replace('.', '_')}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.debug(f"Log file: {log_path}")

    return logger