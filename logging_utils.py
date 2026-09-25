"""Central logging configuration for the project."""

from __future__ import annotations

import logging


def setup_logging(level: int | None = None) -> None:
    """Configure consistent project logging output."""
    if level is None:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,
    )
