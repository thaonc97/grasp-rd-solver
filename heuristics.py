"""GRASP heuristic for one or more fixed trapping-site layouts."""

from __future__ import annotations

import logging
import random
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from config import DEFAULT_CONFIG, Config
from models import RDSolveResult, RegisterInstance, RegisterLayout, SolveResult

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
        self.config = config
        self.rng = random.Random(config.seed if seed is None else seed)
        self.builder = GreedyRandomizedBuilder(instance, alpha, self.rng)
        self.local_search = LocalSearchOptimizer(instance, local_search_strategy)

    def solve(self, max_iterations: Optional[int] = None) -> SolveResult:
        iterations = self.config.max_iterations if max_iterations is None else max_iterations
        if iterations <= 0:
            raise ValueError("max_iterations must be positive")
        best_layout: Optional[RegisterLayout] = None
        history: List[float] = []
        started = time.perf_counter()
        logger.info("Starting GRASP solve: iterations=%d, alpha=%.3f", iterations, self.builder.alpha)
        for iteration in range(1, iterations + 1):
            candidate = self.local_search.optimize(self.builder.build())
            if best_layout is None or candidate.cost < best_layout.cost:
                best_layout = candidate.copy()
            history.append(best_layout.cost)
            logger.debug("GRASP iteration=%d: candidate_cost=%.6f, best_cost=%.6f", iteration, candidate.cost, best_layout.cost)
            if iteration == 1 or iteration % 10 == 0:
                logger.info("GRASP %3d/%d: candidate=%.6f, best=%.6f", iteration, iterations, candidate.cost, best_layout.cost)
        elapsed_seconds = time.perf_counter() - started
        logger.info("Optimization completed in %.3f seconds", elapsed_seconds)
        assert best_layout is not None
        return SolveResult(
            solution=best_layout,
            objective_value=best_layout.cost,
            history=history,
            elapsed_seconds=elapsed_seconds,
        )

    @staticmethod
    def solve_rd(
        layouts: Sequence[np.ndarray], target_Q: np.ndarray, C6: Optional[float] = None,
        alpha: Optional[float] = None, max_iterations: Optional[int] = None,
        seed: Optional[int] = None,
        config: Config = DEFAULT_CONFIG,
    ) -> RDSolveResult:
        if not layouts:
            raise ValueError("layouts must contain at least one layout")
        started = time.perf_counter()
        seeds = random.Random(config.seed if seed is None else seed)
        results: List[SolveResult] = []
        best_result: Optional[SolveResult] = None
        best_layout_index: Optional[int] = None
        for index, sites in enumerate(layouts):
            logger.info("Solving layout %d/%d", index + 1, len(layouts))
            solver = GRASPSolver(RegisterInstance(target_Q, sites, C6, config=config), alpha, seeds.randrange(2**63), config=config)
            result = solver.solve(max_iterations)
            result = SolveResult(
                solution=result.solution,
                objective_value=result.objective_value,
                history=result.history,
                elapsed_seconds=result.elapsed_seconds,
                layout_index=index,
            )
            results.append(result)
            if best_result is None or result.objective_value < best_result.objective_value:
                best_result = result
                best_layout_index = index
        elapsed_seconds = time.perf_counter() - started
        assert best_result is not None and best_layout_index is not None
        return RDSolveResult(
            results=results,
            best_layout_index=best_layout_index,
            best_result=best_result,
            total_elapsed_seconds=elapsed_seconds,
        )


class RandomAssignmentSolver:
    """Baseline that evaluates uniformly random feasible assignments."""

    def __init__(self, instance: RegisterInstance, seed: Optional[int] = None, config: Config = DEFAULT_CONFIG) -> None:
        self.instance = instance
        self.config = config
        self.rng = random.Random(config.seed if seed is None else seed)

    def _build_random_layout(self) -> RegisterLayout:
        """Assign distinct randomly sampled sites to atoms."""
        assignment = self.rng.sample(range(self.instance.M), self.instance.N)
        return RegisterLayout(self.instance, assignment)

    def solve(
        self,
        max_iterations: Optional[int] = None,
    ) -> SolveResult:
        """Return the best assignment found by repeated random sampling."""
        iterations = self.config.max_iterations if max_iterations is None else max_iterations
        if iterations <= 0:
            raise ValueError("max_iterations must be positive")

        best_layout: Optional[RegisterLayout] = None
        history: List[float] = []
        started = time.perf_counter()
        logger.info("Starting random assignment solve: iterations=%d", iterations)
        for iteration in range(1, iterations + 1):
            candidate = self._build_random_layout()
            if best_layout is None or candidate.cost < best_layout.cost:
                best_layout = candidate
            history.append(best_layout.cost)
            logger.debug("Random iteration=%d: candidate_cost=%.6f, best_cost=%.6f", iteration, candidate.cost, best_layout.cost)
            if iteration == 1 or iteration % 10 == 0:
                logger.info("Random %3d/%d: best=%.6f", iteration, iterations, best_layout.cost)
        elapsed_seconds = time.perf_counter() - started
        logger.info("Random optimization completed in %.3f seconds", elapsed_seconds)
        assert best_layout is not None
        return SolveResult(
            solution=best_layout,
            objective_value=best_layout.cost,
            history=history,
            elapsed_seconds=elapsed_seconds,
        )

    @staticmethod
    def solve_layouts(
        layouts: Sequence[np.ndarray],
        target_Q: np.ndarray,
        C6: Optional[float] = None,
        max_iterations: Optional[int] = None,
        seed: Optional[int] = None,
        config: Config = DEFAULT_CONFIG,
    ) -> RDSolveResult:
        """Run the random baseline independently on each calibrated layout."""
        if not layouts:
            raise ValueError("layouts must contain at least one layout")
        started = time.perf_counter()
        seeds = random.Random(config.seed if seed is None else seed)
        results: List[SolveResult] = []
        best_result: Optional[SolveResult] = None
        best_layout_index: Optional[int] = None
        for index, sites in enumerate(layouts):
            logger.info("Solving layout %d/%d", index + 1, len(layouts))
            solver = RandomAssignmentSolver(
                RegisterInstance(target_Q, sites, C6, config=config),
                seed=seeds.randrange(2**63),
                config=config,
            )
            result = solver.solve(max_iterations)
            result = SolveResult(
                solution=result.solution,
                objective_value=result.objective_value,
                history=result.history,
                elapsed_seconds=result.elapsed_seconds,
                layout_index=index,
            )
            results.append(result)
            if best_result is None or result.objective_value < best_result.objective_value:
                best_result = result
                best_layout_index = index
        elapsed_seconds = time.perf_counter() - started
        assert best_result is not None and best_layout_index is not None
        return RDSolveResult(
            results=results,
            best_layout_index=best_layout_index,
            best_result=best_result,
            total_elapsed_seconds=elapsed_seconds,
        )
