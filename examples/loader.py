"""Helpers for reading example instances and solving them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from config import DEFAULT_CONFIG, Config
from heuristics import GRASPSolver
from models import RegisterInstance

EXAMPLES_DIR = Path(__file__).resolve().parent / "instances"


def _as_path(path: str | Path) -> Path:
    return Path(path)


def _load_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=float)
    if path.suffix.lower() == ".json":
        return np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=float)
    raise ValueError(f"Unsupported example matrix format: {path}")


def _load_layout(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=float)
    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported example layout format: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("sites", "layout", "coordinates", "positions"):
            if key in payload:
                return np.asarray(payload[key], dtype=float)
        raise ValueError(f"Layout file does not contain a site array: {path}")
    return np.asarray(payload, dtype=float)


def _candidate_files(directory: Path, names: Sequence[str]) -> List[Path]:
    return [directory / name for name in names if (directory / name).exists()]


def _find_layout_files(directory: Path) -> List[Path]:
    files = []
    for filename in sorted(directory.iterdir()):
        if not filename.is_file():
            continue
        lower = filename.name.lower()
        if lower.startswith("layer_") or lower.startswith("layout_") or lower.startswith("sites_"):
            files.append(filename)
    if files:
        return files

    layers_dir = directory / "layers"
    if layers_dir.is_dir():
        for filename in sorted(layers_dir.iterdir()):
            if filename.is_file() and filename.suffix.lower() in {".json", ".npy"}:
                files.append(filename)
    return files


def list_example_instances(base_dir: str | Path = EXAMPLES_DIR) -> List[Path]:
    """Return all folders under the example-instance directory, sorted by name."""
    directory = _as_path(base_dir)
    if not directory.exists():
        return []
    return sorted([path for path in directory.iterdir() if path.is_dir()], key=lambda p: p.name)


def load_example_instance(instance_dir: str | Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Load Q and all layout files from a single example instance folder."""
    directory = _as_path(instance_dir)
    q_candidates = _candidate_files(directory, ["Q.json", "Q.npy", "target.json", "target.npy"])
    if not q_candidates:
        raise FileNotFoundError(f"No Q matrix file found in {directory}")

    q = _load_matrix(q_candidates[0])
    layouts = []
    for path in _find_layout_files(directory):
        layouts.append(_load_layout(path))
    if not layouts:
        single_layout = directory / "layout.json"
        if single_layout.exists():
            layouts = [_load_layout(single_layout)]
    if not layouts:
        raise FileNotFoundError(f"No layer or layout files found in {directory}")
    return q, layouts


def solve_example_instance(
    instance_dir: str | Path,
    config: Config = DEFAULT_CONFIG,
    max_iterations: int | None = None,
    verbose: bool | None = None,
) -> List[dict]:
    """Solve each layout in an example instance folder and return results."""
    q, layouts = load_example_instance(instance_dir)
    effective_iterations = config.max_iterations if max_iterations is None else max_iterations
    effective_verbose = config.verbose if verbose is None else verbose
    results: List[dict] = []
    for idx, layout in enumerate(layouts):
        instance = RegisterInstance(q, layout, config=config)
        solution, history = GRASPSolver(instance, config=config).solve(effective_iterations, effective_verbose)
        results.append(
            {
                "layer_index": idx,
                "cost": float(solution.cost),
                "assignment": list(solution.pi),
                "history": list(history),
            }
        )
    return results


def solve_all_examples(
    base_dir: str | Path = EXAMPLES_DIR,
    config: Config = DEFAULT_CONFIG,
    max_iterations: int | None = None,
    verbose: bool | None = None,
) -> List[dict]:
    """Solve every example instance folder in the repository."""
    results: List[dict] = []
    for folder in list_example_instances(base_dir):
        results.append({"instance": folder.name, "solutions": solve_example_instance(folder, config, max_iterations, verbose)})
    return results
