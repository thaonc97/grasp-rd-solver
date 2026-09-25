"""Load all example instance folders and solve them with the GRASP heuristic."""

from __future__ import annotations

import logging

from config import DEFAULT_CONFIG, Config
from examples.loader import solve_all_examples
from logging_utils import setup_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    config = Config(
        c6=500.0,
        min_site_distance=1e-8,
        alpha=0.15,
        seed=42,
        max_iterations=10,
        verbose=False,
        log_level=logging.INFO,
    )
    setup_logging(config.log_level)
    for record in solve_all_examples(config=config, verbose=False):
        name = record["instance"]
        solutions = record["solutions"]
        best = min(solutions, key=lambda item: item["cost"])
        logger.info("Instance %s: best cost=%.6f, layer=%d", name, best["cost"], best["layer_index"])
        logger.info("  assignment=%s", best["assignment"])
