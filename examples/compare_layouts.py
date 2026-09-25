"""Compare GRASP results over several 100-site calibrated layouts.

Run from the repository root with:
    python -m examples.compare_layouts
"""

from config import Config
from examples.common import ATOM_COUNT, make_layouts, make_target
from heuristics import GRASPSolver


if __name__ == "__main__":
    config = Config(
        c6=500.0,
        min_site_distance=4.0,
        alpha=0.20,
        seed=2026,
        max_iterations=2,
        verbose=False,
    )
    target = make_target(seed=config.seed, atom_count=ATOM_COUNT)
    layout_index, solution, history = GRASPSolver.solve_layouts(
        make_layouts(),
        target,
        config=config,
    )
    print(f"Best calibrated layout: {layout_index}")
    print(f"Best cost: {solution.cost:.6f}")
    print(f"History of winning layout: {history}")
