"""Example entry point for the fixed-layout register-design heuristic."""

import logging

import numpy as np

from config import DEFAULT_CONFIG
from examples import loader
from heuristics import GRASPSolver, RandomAssignmentSolver
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
    result = GRASPSolver(instance, config=DEFAULT_CONFIG).solve()
    logger.info("Best cost: %.6f", result.objective_value)
    logger.info("Assignment: %s", result.solution.pi)
    logger.info("History length: %d", len(result.history))


if __name__ == "__main__":
    setup_logging(DEFAULT_CONFIG.log_level)
    # example()
    atoms_example = 4 # number of atoms
    layout_size_example = 9 # number trapping sites
    instance_location = f"examples/instances/instance_{atoms_example}_atoms_{layout_size_example}_sites"

    q, layouts = loader.load_example_instance(instance_location)
    solve_results = RandomAssignmentSolver.solve_layouts(
        layouts,
        q,
        max_iterations=DEFAULT_CONFIG.max_iterations,
    )