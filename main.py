"""Example entry point for the fixed-layout register-design heuristic."""

import logging

import numpy as np

from config import DEFAULT_CONFIG
from examples.loader import solve_example_instance
from heuristics import GRASPSolver
from logging_utils import setup_logging
from models import RegisterInstance

logger = logging.getLogger(__name__)


def example() -> None:
    atom_count = 8
    grid_x, grid_y = np.meshgrid(np.linspace(0, 12, 4), np.linspace(0, 12, 4))
    layout = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    rng = np.random.default_rng(42)
    raw = rng.uniform(0.1, 2.0, size=(atom_count, atom_count))
    target = (raw + raw.T) / 2.0  # Q matrix (symmetric)
    np.fill_diagonal(target, DEFAULT_CONFIG.target_diagonal)

    instance = RegisterInstance(target, layout, config=DEFAULT_CONFIG)
    solution, history = GRASPSolver(instance, config=DEFAULT_CONFIG).solve(
        DEFAULT_CONFIG.max_iterations,
        DEFAULT_CONFIG.verbose,
    )
    logger.info("Best cost: %.6f", solution.cost)
    logger.info("Assignment: %s", solution.pi)
    logger.info("History length: %d", len(history))


if __name__ == "__main__":
    setup_logging(DEFAULT_CONFIG.log_level)
    example()

    results = solve_example_instance(
        "examples/instances/instance_4_atoms_9_sites",
        max_iterations=20,
        verbose=False
    )
    print(results)