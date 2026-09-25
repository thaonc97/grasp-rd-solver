"""Shared data builders for the larger examples."""

from __future__ import annotations

from typing import List

import numpy as np

from config import DEFAULT_CONFIG


ATOM_COUNT = 81
SITE_COUNT = 100
GRID_SIDE = 10


def make_target(
    seed: int = 42,
    atom_count: int = ATOM_COUNT,
    diagonal: float = DEFAULT_CONFIG.target_diagonal,
) -> np.ndarray:
    """Create a deterministic symmetric target matrix for demonstrations."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.1, 2.0, size=(atom_count, atom_count))
    target = (raw + raw.T) / 2.0
    np.fill_diagonal(target, diagonal)
    return target


def make_grid(spacing: float = 4.0, scale: float = 1.0) -> np.ndarray:
    """Create a centered 10x10 calibrated layout with 100 trapping sites."""
    coordinates = np.linspace(-4.5 * spacing, 4.5 * spacing, GRID_SIDE)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    return np.column_stack([grid_x.ravel(), grid_y.ravel()]) * scale


def make_layouts() -> List[np.ndarray]:
    """Return three distinct calibrated layouts for comparison experiments."""
    rectangular = make_grid(spacing=4.0)
    stretched = make_grid(spacing=4.0, scale=1.15)
    anisotropic = rectangular.copy()
    anisotropic[:, 0] *= 1.10
    anisotropic[:, 1] *= 0.95
    return [rectangular, stretched, anisotropic]
