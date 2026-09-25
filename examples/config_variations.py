"""Run the same 81-atom instance with two GRASP configurations.

Run from the repository root with:
    python -m examples.config_variations
"""

from config import Config
from examples.common import ATOM_COUNT, make_grid, make_target
from heuristics import GRASPSolver
from models import RegisterInstance


if __name__ == "__main__":
    target = make_target(seed=17, atom_count=ATOM_COUNT)
    sites = make_grid(spacing=4.0)
    configurations = [
        Config(c6=500.0, min_site_distance=4.0, alpha=0.10, seed=11, max_iterations=2, verbose=False),
        Config(c6=500.0, min_site_distance=4.0, alpha=0.50, seed=22, max_iterations=2, verbose=False),
    ]

    for index, config in enumerate(configurations, start=1):
        instance = RegisterInstance(target, sites, config=config)
        solution, _ = GRASPSolver(instance, config=config).solve(
            config.max_iterations,
            config.verbose,
        )
        print(f"Configuration {index}: alpha={config.alpha:.2f}, seed={config.seed}, cost={solution.cost:.6f}")
