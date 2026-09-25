import json

import numpy as np
import pytest

from examples.loader import load_example_instance
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
    solution, history = GRASPSolver(make_instance(), seed=3).solve(4, verbose=False)
    assert solution.is_complete
    assert len(set(solution.pi)) == solution.N
    assert len(history) == 4
    assert history[-1] == pytest.approx(solution.cost)


def test_best_improvement_is_supported():
    instance = make_instance()
    layout = RegisterLayout(instance, [0, 1, 3, 4])
    result = LocalSearchOptimizer(instance, "best_improvement").optimize(layout)
    assert result.cost == pytest.approx(result.compute_full_cost())


def test_all_fixed_layouts_are_compared():
    instance = make_instance()
    layouts = [instance.sites, instance.sites[[0, 1, 2, 3, 4, 5, 6, 8, 7]]]
    index, solution, _ = GRASPSolver.solve_layouts(layouts, instance.W, seed=5, max_iterations=2)
    assert index in {0, 1}
    assert solution.is_complete


def test_random_baseline_returns_feasible_reproducible_solution():
    instance = make_instance()
    first, first_history = RandomAssignmentSolver(instance, seed=9).solve(4, verbose=False)
    second, second_history = RandomAssignmentSolver(instance, seed=9).solve(4, verbose=False)
    assert first.is_complete
    assert len(set(first.pi)) == first.N
    assert first.pi == second.pi
    assert first_history == second_history


def test_example_instance_loader_reads_matrix_and_layers(tmp_path):
    target = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.9], [0.5, 0.9, 0.0]], dtype=float)
    layer_0 = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=float)
    layer_1 = np.array([[0.5, 0.5], [2.5, 0.5], [0.5, 2.5]], dtype=float)

    (tmp_path / "Q.json").write_text(json.dumps(target.tolist()))
    (tmp_path / "layer_0.json").write_text(json.dumps({"sites": layer_0.tolist()}))
    (tmp_path / "layer_1.json").write_text(json.dumps({"sites": layer_1.tolist()}))

    loaded_Q, loaded_layers = load_example_instance(tmp_path)

    assert np.allclose(loaded_Q, target)
    assert len(loaded_layers) == 2
    assert loaded_layers[0].shape == (3, 2)
    assert loaded_layers[1].shape == (3, 2)
