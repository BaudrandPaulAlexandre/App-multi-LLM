"""
logger.py
---------
Configuration centralisée du logging pour tout le pipeline.

Usage depuis n'importe quel module :
    from eloquent.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Message")
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """
    Configure le logging global : console + fichier optionnel.
    À appeler une seule fois au démarrage (dans run.py).

    Args:
        log_file : si fourni, les logs sont aussi écrits dans ce fichier
        level    : niveau de log (logging.DEBUG, INFO, WARNING, ERROR)
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout)
    ]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,  # Écrase une config précédente si appelé plusieurs fois
    )


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger nommé. À utiliser en haut de chaque module."""
    return logging.getLogger(name)
