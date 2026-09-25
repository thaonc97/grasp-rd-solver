"""GRASP heuristic for one or more fixed trapping-site layouts."""

from __future__ import annotations

import logging
import random
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from config import DEFAULT_CONFIG, Config
from logging_utils import setup_logging
from models import RegisterInstance, RegisterLayout

logger = logging.getLogger(__name__)


class GreedyRandomizedBuilder:
    """First phase of the GRASP heuristic: construct a feasible solution using a greedy randomized approach"""
    def __init__(self, instance: RegisterInstance, alpha: float = 0.15, rng: Optional[random.Random] = None) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.instance = instance
        self.alpha = alpha
        self.rng = rng or random.Random()

    def build(self) -> RegisterLayout:
        assignment = [-1] * self.instance.N
        unassigned_atoms = list(range(self.instance.N))
        unassigned_sites = list(range(self.instance.M))
        while unassigned_atoms:
            candidates: List[Tuple[int, int, float]] = []
            for atom in unassigned_atoms:
                for site in unassigned_sites:
                    marginal = sum(
                        (self.instance.U[site, assignment[other]] - self.instance.W[atom, other]) ** 2
                        for other in range(self.instance.N)
                        if assignment[other] >= 0
                    ) # Compute the marginal cost of assigning the current atom to the current site, deltaC in the document
                    candidates.append((atom, site, float(marginal)))
            minimum = min(value for _, _, value in candidates)
            maximum = max(value for _, _, value in candidates)
            threshold = minimum + self.alpha * (maximum - minimum)
            rcl = [(atom, site) for atom, site, value in candidates if value <= threshold + 1e-12]
            atom, site = self.rng.choice(rcl)
            assignment[atom] = site
            unassigned_atoms.remove(atom)
            unassigned_sites.remove(site)
        return RegisterLayout(self.instance, assignment)


class LocalSearchOptimizer:
    """Second phase of the GRASP heuristic: Local search optimizer for improving a given atom-to-site assignment."""
    def __init__(self, instance: RegisterInstance, strategy: str = "first_improvement") -> None:
        if strategy not in {"first_improvement", "best_improvement"}:
            raise ValueError("strategy must be first_improvement or best_improvement")
        self.instance = instance
        self.strategy = strategy

    def optimize(self, layout: RegisterLayout) -> RegisterLayout:
        current = layout.copy()
        while True:
            best_move = None
            for i in range(self.instance.N):
                for j in range(i + 1, self.instance.N):
                    delta = current.evaluate_swap_delta(i, j)
                    if delta < -1e-9:
                        if self.strategy == "first_improvement":
                            current.apply_swap(i, j, delta)
                            best_move = True
                            break
                        if best_move is None or delta < best_move[2]:
                            best_move = (i, j, delta)
                if best_move is True:
                    break
            if best_move is None:
                return current
            if best_move is not True:
                current.apply_swap(*best_move)


class GRASPSolver:
    """GRASP solver that iteratively constructs and improves atom-to-site assignments."""
    def __init__(self, instance: RegisterInstance, alpha: Optional[float] = None, seed: Optional[int] = None, local_search_strategy: Optional[str] = None, config: Config = DEFAULT_CONFIG) -> None:
        if alpha is None:
            alpha = config.alpha
        if local_search_strategy is None:
            local_search_strategy = config.local_search_strategy
        self.instance = instance
        self.rng = random.Random(config.seed if seed is None else seed)
        self.builder = GreedyRandomizedBuilder(instance, alpha, self.rng)
        self.local_search = LocalSearchOptimizer(instance, local_search_strategy)

    def solve(self, max_iterations: int = 50, verbose: bool = True) -> Tuple[RegisterLayout, List[float]]:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        best_layout: Optional[RegisterLayout] = None
        history: List[float] = []
        started = time.perf_counter()
        logger.info("Starting GRASP solve: iterations=%d, alpha=%.3f", max_iterations, self.builder.alpha)
        for iteration in range(1, max_iterations + 1):
            candidate = self.local_search.optimize(self.builder.build())
            if best_layout is None or candidate.cost < best_layout.cost:
                best_layout = candidate.copy()
            history.append(best_layout.cost)
            logger.debug("GRASP iteration=%d: candidate_cost=%.6f, best_cost=%.6f", iteration, candidate.cost, best_layout.cost)
            if verbose and (iteration == 1 or iteration % 10 == 0):
                logger.info("GRASP %3d/%d: candidate=%.6f, best=%.6f", iteration, max_iterations, candidate.cost, best_layout.cost)
        if verbose:
            logger.info("Optimization completed in %.3f seconds", time.perf_counter() - started)
        return best_layout, history

    @staticmethod
    def solve_layouts(
        layouts: Sequence[np.ndarray], target_W: np.ndarray, C6: Optional[float] = None,
        alpha: Optional[float] = None, max_iterations: Optional[int] = None,
        seed: Optional[int] = None, verbose: Optional[bool] = None,
        config: Config = DEFAULT_CONFIG,
    ) -> Tuple[int, RegisterLayout, List[float]]:
        if not layouts:
            raise ValueError("layouts must contain at least one layout")
        seeds = random.Random(config.seed if seed is None else seed)
        best = None
        for index, sites in enumerate(layouts):
            solver = GRASPSolver(RegisterInstance(target_W, sites, C6, config=config), alpha, seeds.randrange(2**63), config=config)
            solution, history = solver.solve(max_iterations or config.max_iterations, config.verbose if verbose is None else verbose)
            result = (index, solution, history)
            if best is None or solution.cost < best[1].cost:
                best = result
        return best


class RandomAssignmentSolver:
    """Baseline that evaluates uniformly random feasible assignments."""

    def __init__(self, instance: RegisterInstance, seed: Optional[int] = None, config: Config = DEFAULT_CONFIG) -> None:
        self.instance = instance
        self.rng = random.Random(config.seed if seed is None else seed)

    def _build_random_layout(self) -> RegisterLayout:
        """Assign distinct randomly sampled sites to atoms."""
        assignment = self.rng.sample(range(self.instance.M), self.instance.N)
        return RegisterLayout(self.instance, assignment)

    def solve(
        self,
        max_iterations: Optional[int] = None,
        verbose: Optional[bool] = None,
        config: Config = DEFAULT_CONFIG,
    ) -> Tuple[RegisterLayout, List[float]]:
        """Return the best assignment found by repeated random sampling."""
        iterations = config.max_iterations if max_iterations is None else max_iterations
        show_progress = config.verbose if verbose is None else verbose
        if iterations <= 0:
            raise ValueError("max_iterations must be positive")

        best_layout: Optional[RegisterLayout] = None
        history: List[float] = []
        logger.info("Starting random baseline solve: iterations=%d", iterations)
        for iteration in range(1, iterations + 1):
            candidate = self._build_random_layout()
            if best_layout is None or candidate.cost < best_layout.cost:
                best_layout = candidate
            history.append(best_layout.cost)
            logger.debug("Random iteration=%d: candidate_cost=%.6f, best_cost=%.6f", iteration, candidate.cost, best_layout.cost)
            if show_progress and (iteration == 1 or iteration % 10 == 0):
                logger.info("Random %3d/%d: best=%.6f", iteration, iterations, best_layout.cost)
        return best_layout, history

    @staticmethod
    def solve_layouts(
        layouts: Sequence[np.ndarray],
        target_W: np.ndarray,
        C6: Optional[float] = None,
        max_iterations: Optional[int] = None,
        seed: Optional[int] = None,
        verbose: Optional[bool] = None,
        config: Config = DEFAULT_CONFIG,
    ) -> Tuple[int, RegisterLayout, List[float]]:
        """Run the random baseline independently on each calibrated layout."""
        if not layouts:
            raise ValueError("layouts must contain at least one layout")
        seeds = random.Random(config.seed if seed is None else seed)
        best = None
        for index, sites in enumerate(layouts):
            solver = RandomAssignmentSolver(
                RegisterInstance(target_W, sites, C6, config=config),
                seed=seeds.randrange(2**63),
                config=config,
            )
            solution, history = solver.solve(max_iterations, verbose, config)
            result = (index, solution, history)
            if best is None or solution.cost < best[1].cost:
                best = result
        return best
