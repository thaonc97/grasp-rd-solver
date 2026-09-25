 # Fixed-layout neutral-atom register design

This repository implements a GRASP heuristic for Version 2 of the register-design problem. Each calibrated layout is optimized independently, and the best layout is returned.

Runtime defaults live in `config.py`; change `DEFAULT_CONFIG` for a normal experiment or create a separate `Config` for an experiment-specific run.

## Files

- `models.py`: validated `RegisterInstance` and `RegisterLayout` classes.
- `heuristics.py`: randomized construction, first/best-improvement local search, and GRASP orchestration.
- `heuristics.py`: also contains `RandomAssignmentSolver`, a pure random baseline for comparison.
- `main.py`: runnable example.
- `tests.py`: regression tests for feasibility, swap deltas, GRASP, and layout comparison.
- `config.py`: physical, reproducibility, and heuristic defaults.
- `examples/`: example-instance folders, loaders, and runnable demos.
- `examples/loader.py`: discovers instance folders and loads `Q` plus one or more layer layouts.
- `examples/solve_instances.py`: solves every example folder in the repository.

## Run

```text
python -m pytest tests.py -q
python main.py
```

Larger examples can be run from the repository root with:

```text
python -m examples.large_grid
python -m examples.compare_layouts
python -m examples.config_variations
```

The examples use 81 atoms and 100 trapping sites. Their iteration counts are intentionally small so that the demonstrations remain practical; increase `max_iterations` in each example for longer experiments.

To override settings without editing the heuristic:

```python
from config import Config

experiment = Config(c6=750.0, min_site_distance=2.0, seed=7, max_iterations=100)
```

For multiple calibrated layouts:

```python
from heuristics import GRASPSolver

layout_index, solution, history = GRASPSolver.solve_layouts(
	layouts, target_W=Q, C6=500.0, max_iterations=50, seed=123
)
```

The target matrix diagonal is configurable through `Config.target_diagonal` and defaults to `-1.0`. The objective is the squared error over unordered atom pairs:

```text
sum((U[pi[i], pi[j]] - Q[i, j]) ** 2 for i < j)
```

`Q` is assumed symmetric. Site coordinates are validated for finite values, minimum pairwise distance, and optionally maximum radius.
