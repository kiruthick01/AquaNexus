"""Logging setup.

``get_logger`` is the only entry point; it configures the root handler once and
returns a namespaced child logger.
"""

import logging
import sys

from aquanexus.config import settings

_CONFIGURED = False

_FORMAT = "%(asctime)s  %(levelname)-8s %(name)-28s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger("aquanexus")
    root.setLevel(settings.LOG_LEVEL)
    root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``aquanexus`` namespace."""
    _configure()
    return logging.getLogger(f"aquanexus.{name}" if not name.startswith("aquanexus") else name)
