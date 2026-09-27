 # Fixed-layout neutral-atom register design

This repository implements a GRASP heuristic for Version 2 of the register-design problem. Each calibrated layout is optimized independently, and the best layout is returned.

The algorithm detail (GRASP) can be found in attached answer file.

The UML class diagrams are given in `UML-high-level.md`  and `UML.md` files.

Runtime defaults live in `config.py`; change `DEFAULT_CONFIG` for a normal experiment or create a separate `Config` for an experiment-specific run.

## Files

- `models.py`: validated `RegisterInstance` and `RegisterLayout` classes.
- `heuristics.py`: implementation of GRASP algorithm `GRASPSolver`.
- `heuristics.py`: also contains `RandomAssignmentSolver`, a pure random baseline for comparison.
- `main.py`: runnable example.
- `tests.py`: regression tests for feasibility, swap deltas, GRASP, and layout comparison.
- `config.py`: physical, reproducibility, and heuristic defaults.
- `examples/`: example-instance folders and loading helpers.
- `examples/loader.py`: discovers instance folders and loads `Q`, along with one or more trapping site layouts.
- `experiments.ipynb` notebook, an example of using GRASP to solve instances in `examples/instances` included for quickly understand the codebase.

## Run

```text
python -m pytest tests.py -q
python main.py
```

To solve an example JSON instance directly, in root:

```python
from examples import loader
from heuristics import GRASPSolver

instance_dir = "examples/instances/instance_4_atoms_9_sites"
q, layouts = loader.load_example_instance(instance_dir)

result = GRASPSolver.solve_rd(
	layouts,
	q,
	max_iterations=50,
)

print(f"Best layout: {result.best_layout_index}")
print(f"Best cost: {result.objective_value:.6f}")
print(f"Total time: {result.total_elapsed_seconds:.3f} seconds")
```

. Each `result.results` entry contains the objective value, iteration history, and elapsed time for one layout.

To override settings without editing the heuristic:

```python
from config import Config

experiment = Config(c6=750.0, min_site_distance=2.0, seed=7, max_iterations=100)
```


`Q` is assumed symmetric. Site coordinates are validated for finite values, minimum pairwise distance, and optionally maximum radius.
