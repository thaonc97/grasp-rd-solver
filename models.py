"""Validated data structures for fixed-layout neutral-atom register design."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from config import DEFAULT_CONFIG, Config


class RegisterInstance:
    """A symmetric target matrix together with one calibrated site layout."""

    def __init__(
        self,
        target_W: np.ndarray,
        site_positions: np.ndarray,
        C6: Optional[float] = None,
        min_site_distance: Optional[float] = None,
        max_radius: Optional[float] = None,
        config: Config = DEFAULT_CONFIG,
    ) -> None:
        """Validate one calibrated layout and precompute its interactions."""
        target_W = np.asarray(target_W, dtype=float)
        site_positions = np.asarray(site_positions, dtype=float)
        C6, min_site_distance, max_radius = self._resolve_hardware_config(
            C6, min_site_distance, max_radius, config
        )
        self._validate_inputs(target_W, site_positions, C6, min_site_distance, max_radius)

        self.W = target_W.copy()
        self.sites = site_positions.copy()
        self.C6 = float(C6)
        self.N = target_W.shape[0]
        self.M = site_positions.shape[0]
        self.min_site_distance = float(min_site_distance)
        self.max_radius = max_radius
        distances = self._validate_geometry()
        self.U = self._precompute_site_interactions(distances)

    @staticmethod
    def _resolve_hardware_config(
        C6: Optional[float],
        min_site_distance: Optional[float],
        max_radius: Optional[float],
        config: Config,
    ) -> tuple[float, float, Optional[float]]:
        """Use explicit hardware values, falling back to project configuration."""
        return (
            config.c6 if C6 is None else C6,
            config.min_site_distance if min_site_distance is None else min_site_distance,
            config.max_radius if max_radius is None else max_radius,
        )

    @classmethod
    def _validate_inputs(
        cls,
        target_W: np.ndarray,
        site_positions: np.ndarray,
        C6: float,
        min_site_distance: float,
        max_radius: Optional[float],
    ) -> None:
        """Run all input validations required before instance state is created."""
        cls._validate_target(target_W)
        cls._validate_sites(site_positions)
        cls._validate_hardware_limits(C6, min_site_distance, max_radius)

    @staticmethod
    def _validate_target(target_W: np.ndarray) -> None:
        """Ensure the target interaction matrix is finite, square, and symmetric."""
        if target_W.ndim != 2 or target_W.shape[0] != target_W.shape[1]:
            raise ValueError("target_W must be a square matrix")
        if not np.allclose(target_W, target_W.T):
            raise ValueError("target_W must be symmetric")
        if not np.all(np.isfinite(target_W)):
            raise ValueError("target_W must be finite")

    @staticmethod
    def _validate_sites(site_positions: np.ndarray) -> None:
        """Ensure trapping-site coordinates are finite two-dimensional points."""
        if site_positions.ndim != 2 or site_positions.shape[1] != 2:
            raise ValueError("site_positions must have shape (M, 2)")
        if not np.all(np.isfinite(site_positions)):
            raise ValueError("site_positions must be finite")

    @staticmethod
    def _validate_hardware_limits(
        C6: float, min_site_distance: float, max_radius: Optional[float]
    ) -> None:
        """Ensure physical coefficients and distance limits are positive."""
        if C6 <= 0:
            raise ValueError("C6 must be positive")
        if min_site_distance <= 0:
            raise ValueError("min_site_distance must be positive")
        if max_radius is not None and max_radius <= 0:
            raise ValueError("max_radius must be positive")

    def _validate_geometry(self) -> np.ndarray:
        """Validate site capacity and geometry, returning all pairwise distances."""
        if self.M < self.N:
            raise ValueError(f"trapping sites M ({self.M}) must be >= atoms N ({self.N})")
        distances = np.linalg.norm(self.sites[:, None] - self.sites[None, :], axis=2)
        pair_distances = distances[np.triu_indices(self.M, k=1)]
        if pair_distances.size and np.any(pair_distances < self.min_site_distance):
            raise ValueError("site positions violate min_site_distance")
        if self.max_radius is not None and np.any(np.linalg.norm(self.sites, axis=1) > self.max_radius):
            raise ValueError("site positions violate max_radius")
        return distances

    def _precompute_site_interactions(self, distances: np.ndarray) -> np.ndarray:
        """Build the Van der Waals matrix using U[k,l] = C6 / distance[k,l]^6."""
        with np.errstate(divide="raise", over="raise"):
            U = np.divide(self.C6, distances**6, out=np.zeros_like(distances), where=distances > 0)
        np.fill_diagonal(U, 0.0)
        return U


class RegisterLayout:
    """An assignment mapping each atom to a distinct trapping site."""

    def __init__(self, instance: RegisterInstance, pi: Optional[Sequence[int]] = None) -> None:
        """Create a validated atom-to-site assignment.

        Parameters
        ----------
        instance : RegisterInstance
            Register problem containing the target matrix and site interactions.
        pi : sequence of int, optional
            Site index for each atom. If omitted, all atoms start unassigned and
            are represented by ``-1``.

        Attributes
        ----------
        pi : list of int
            Atom-to-site assignment. Each assigned site index is unique.
        cost : float
            Cached objective value for a complete assignment, or infinity for a
            partial assignment.

        Examples
        --------
        For three atoms, ``pi=[4, 8, 2]`` assigns atom 0 to site 4, atom 1
        to site 8, and atom 2 to site 2. A partial assignment can use ``-1``::

            layout = RegisterLayout(instance, pi=[4, -1, 2])
        """
        self.instance = instance
        self.N = instance.N
        self.M = instance.M
        self.pi = list(pi) if pi is not None else [-1] * self.N  
        self._validate_assignment()
        self.cost = self.compute_full_cost() if self.is_complete else float("inf")

    @property
    def is_complete(self) -> bool:
        return all(site >= 0 for site in self.pi)

    def _validate_assignment(self) -> None:
        if len(self.pi) != self.N:
            raise ValueError(f"assignment must contain exactly {self.N} entries")
        if any(site < -1 or site >= self.M for site in self.pi):
            raise ValueError("assignment contains an invalid site index")
        assigned = [site for site in self.pi if site >= 0]
        if len(set(assigned)) != len(assigned):
            raise ValueError("each trapping site can be assigned to at most one atom")

    def compute_full_cost(self) -> float:
        if not self.is_complete:
            raise ValueError("a complete assignment is required to compute cost")
        indices = np.asarray(self.pi, dtype=int)
        residual = self.instance.U[np.ix_(indices, indices)] - self.instance.W
        return float(np.sum(np.triu(residual * residual, k=1)))

    def evaluate_swap_delta(self, i: int, j: int) -> float:
        if not self.is_complete:
            raise ValueError("a complete assignment is required for swaps")
        if not (0 <= i < self.N and 0 <= j < self.N) or i == j:
            raise IndexError("swap indices must be distinct atom indices")
        site_i, site_j = self.pi[i], self.pi[j]
        delta = 0.0
        for atom in range(self.N):
            if atom in (i, j):
                continue
            site = self.pi[atom]
            old_i = (self.instance.U[site_i, site] - self.instance.W[i, atom]) ** 2
            old_j = (self.instance.U[site_j, site] - self.instance.W[j, atom]) ** 2
            new_i = (self.instance.U[site_j, site] - self.instance.W[i, atom]) ** 2
            new_j = (self.instance.U[site_i, site] - self.instance.W[j, atom]) ** 2
            delta += new_i + new_j - old_i - old_j
        return float(delta)

    def apply_swap(self, i: int, j: int, delta: float) -> None:
        self.pi[i], self.pi[j] = self.pi[j], self.pi[i]
        self.cost += float(delta)

    def copy(self) -> "RegisterLayout":
        copied = RegisterLayout(self.instance, self.pi)
        copied.cost = self.cost
        return copied
