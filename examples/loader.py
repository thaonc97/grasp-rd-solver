"""Helpers for reading example instances and solving them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import DEFAULT_CONFIG, Config
from heuristics import GRASPSolver, RandomAssignmentSolver
from models import RegisterInstance

EXAMPLES_DIR = Path(__file__).resolve().parent / "instances"


def _load_matrix(path: Path) -> np.ndarray:
    return np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=float)


def _load_layout(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "sites" not in payload:
            raise ValueError(f"Layout file does not contain a 'sites' array: {path}")
        payload = payload["sites"]
    return np.asarray(payload, dtype=float)


def _find_layout_files(directory: Path) -> List[Path]:
    return sorted(directory.glob("layout_*.json"))


def list_example_instances(base_dir: str | Path = EXAMPLES_DIR) -> List[Path]:
    """Return all folders under the example-instance directory, sorted by name."""
    directory = Path(base_dir)
    if not directory.exists():
        return []
    return sorted([path for path in directory.iterdir() if path.is_dir()], key=lambda p: p.name)


def load_example_instance(instance_dir: str | Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Load Q and all layout files from a single example instance folder."""
    directory = Path(instance_dir)
    q_path = directory / "Q.json"
    if not q_path.exists():
        raise FileNotFoundError(f"No Q.json file found in {directory}")

    q = _load_matrix(q_path)
    layouts = [_load_layout(path) for path in _find_layout_files(directory)]
    if not layouts:
        raise FileNotFoundError(f"No layout_*.json files found in {directory}")
    return q, layouts


def _resolve_solver(solver: str | type):
    """Resolve a solver name or class into a concrete solver implementation."""
    if isinstance(solver, str):
        key = solver.lower()
        if key == "grasp":
            return GRASPSolver
        if key == "random":
            return RandomAssignmentSolver
        raise ValueError(f"Unsupported solver: {solver}")
    if isinstance(solver, type) and hasattr(solver, "solve"):
        return solver
    raise TypeError("solver must be 'grasp' or 'random'")


def solve_example_instance(
    instance_dir: str | Path,
    config: Config = DEFAULT_CONFIG,
    max_iterations: int | None = None,
    solver: str | type = "grasp",
    solver_kwargs: dict | None = None,
) -> List[dict]:
    """Solve each layout in an example instance folder and return results.

    Parameters
    ----------
    solver:
        Either 'grasp', 'random', or a solver class implementing solve().
    solver_kwargs:
        Extra constructor arguments forwarded to the selected solver.
    """
    q, layouts = load_example_instance(instance_dir)
    effective_iterations = config.max_iterations if max_iterations is None else max_iterations
    solver_cls = _resolve_solver(solver)
    solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)
    results: List[dict] = []
    for idx, layout in enumerate(layouts):
        instance = RegisterInstance(q, layout, config=config)
        solver_instance = solver_cls(instance, config=config, **solver_kwargs)
        solve_result = solver_instance.solve(effective_iterations)
        results.append(
            {
                "layer_index": idx,
                "cost": solve_result.objective_value,
                "assignment": list(solve_result.solution.pi),
                "history": list(solve_result.history),
                "elapsed_seconds": solve_result.elapsed_seconds,
            }
        )
    return results


def solve_all_examples(
    base_dir: str | Path = EXAMPLES_DIR,
    config: Config = DEFAULT_CONFIG,
    max_iterations: int | None = None,
    solver: str | type = "grasp",
    solver_kwargs: dict | None = None,
) -> List[dict]:
    """Solve every example instance folder in the repository."""
    results: List[dict] = []
    for folder in list_example_instances(base_dir):
        results.append({
            "instance": folder.name,
            "solutions": solve_example_instance(folder, config, max_iterations, solver, solver_kwargs),
        })
    return results
