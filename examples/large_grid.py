"""Run GRASP on 81 atoms and a 100-site calibrated grid.

Run from the repository root with:
    python -m examples.large_grid
"""

from config import Config
from examples.common import ATOM_COUNT, make_grid, make_target
from heuristics import GRASPSolver
from models import RegisterInstance


if __name__ == "__main__":
    config = Config(
        c6=500.0,
        min_site_distance=4.0,
        alpha=0.20,
        seed=2026,
        max_iterations=3,
        verbose=True,
    )
    target = make_target(seed=config.seed, atom_count=ATOM_COUNT)
    sites = make_grid(spacing=4.0)
    instance = RegisterInstance(target, sites, config=config)
    solution, history = GRASPSolver(instance, config=config).solve(
        config.max_iterations,
        config.verbose,
    )
    print(f"Atoms: {instance.N}; sites: {instance.M}")
    print(f"Best cost: {solution.cost:.6f}")
    print(f"History: {history}")
