import json

import numpy as np
import pytest

from examples.loader import load_example_instance, solve_example_instance
from heuristics import GRASPSolver, LocalSearchOptimizer, RandomAssignmentSolver
from models import RegisterInstance, RegisterLayout


def make_instance(atom_count=4):
    target = np.ones((atom_count, atom_count), dtype=float)
    np.fill_diagonal(target, 0.0)
    sites = np.array([[x, y] for x in range(3) for y in range(3)], dtype=float)
    return RegisterInstance(target, sites, C6=1.0)


def test_swap_delta_matches_full_cost():
    instance = make_instance()
    layout = RegisterLayout(instance, [0, 1, 3, 4])
    delta = layout.evaluate_swap_delta(0, 2)
    swapped = layout.copy()
    swapped.apply_swap(0, 2, delta)
    assert swapped.cost == pytest.approx(swapped.compute_full_cost())


def test_invalid_assignment_is_rejected():
    with pytest.raises(ValueError):
        RegisterLayout(make_instance(), [0, 0, 1, 2])


def test_grasp_returns_complete_solution():
    result = GRASPSolver(make_instance(), seed=3).solve(4)
    assert result.solution.is_complete
    assert len(set(result.solution.pi)) == result.solution.N
    assert len(result.history) == 4
    assert result.history[-1] == pytest.approx(result.objective_value)


def test_best_improvement_is_supported():
    instance = make_instance()
    layout = RegisterLayout(instance, [0, 1, 3, 4])
    result = LocalSearchOptimizer(instance, "best_improvement").optimize(layout)
    assert result.cost == pytest.approx(result.compute_full_cost())


def test_all_fixed_layouts_are_compared():
    instance = make_instance()
    layouts = [instance.sites, instance.sites[[0, 1, 2, 3, 4, 5, 6, 8, 7]]]
    result = GRASPSolver.solve_rd(layouts, instance.W, seed=5, max_iterations=2)
    assert result.best_layout_index in {0, 1}
    assert result.best_result.solution.is_complete
    assert len(result.results) == 2
    assert result.total_elapsed_seconds >= 0.0


def test_random_baseline_returns_feasible_reproducible_solution():
    instance = make_instance()
    first = RandomAssignmentSolver(instance, seed=9).solve(4)
    second = RandomAssignmentSolver(instance, seed=9).solve(4)
    assert first.solution.is_complete
    assert len(set(first.solution.pi)) == first.solution.N
    assert first.solution.pi == second.solution.pi
    assert first.history == second.history


def test_example_loader_accepts_random_solver(tmp_path):
    target = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    layout = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    (tmp_path / "Q.json").write_text(str(target.tolist()).replace("'", '"'))
    (tmp_path / "layout_0.json").write_text(str({"sites": layout.tolist()}).replace("'", '"'))

    results = load_example_instance(tmp_path)
    assert results[0].shape == (2, 2)

    solver_results = solve_example_instance(tmp_path, solver="random", solver_kwargs={"seed": 7}, max_iterations=2)
    assert len(solver_results) == 1
    assert solver_results[0]["cost"] >= 0.0


def test_example_instance_loader_reads_matrix_and_layers(tmp_path):
    target = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.9], [0.5, 0.9, 0.0]], dtype=float)
    layer_0 = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=float)
    layer_1 = np.array([[0.5, 0.5], [2.5, 0.5], [0.5, 2.5]], dtype=float)

    (tmp_path / "Q.json").write_text(json.dumps(target.tolist()))
    (tmp_path / "layout_0.json").write_text(json.dumps({"sites": layer_0.tolist()}))
    (tmp_path / "layout_1.json").write_text(json.dumps({"sites": layer_1.tolist()}))

    loaded_Q, loaded_layers = load_example_instance(tmp_path)

    assert np.allclose(loaded_Q, target)
    assert len(loaded_layers) == 2
    assert loaded_layers[0].shape == (3, 2)
    assert loaded_layers[1].shape == (3, 2)
