"""Logging setup.

``get_logger`` is the only entry point; it configures the root handler once and
returns a namespaced child logger.
"""

import contextlib
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

    stream = sys.stderr
    # Station and river names are Japanese throughout. A Windows console
    # defaults to cp1252, which renders them as \uXXXX escapes; reconfigure to
    # UTF-8 and replace anything the terminal still cannot draw.
    if hasattr(stream, "reconfigure"):
        # Suppressed: the stream may be detached or non-reconfigurable, in which
        # case the default encoding is the best available and still usable.
        with contextlib.suppress(OSError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")

    handler = logging.StreamHandler(stream)
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
