# Example instance folders

Each test instance lives in its own folder under `examples/instances/`.

## Folder convention

Use a descriptive folder name that includes the problem size, for example:

- `instance_4_atoms_9_sites/`
- `instance_6_atoms_12_sites/`
- `instance_50_atoms_100_sites/`

Inside each folder, keep a single input matrix and one or more layer files:

- `Q.json` describes the target interaction matrix.
- `layout_0.json`, `layout_1.json`, ... describe site coordinates for each layout.

## Supported file formats

The loader accepts:

- `Q.json`
- `layout_*.json`

Each layout file may be either a raw NumPy-style array or a dictionary containing a `sites` key.

## Loading and solving

The repository includes helper code to discover and solve example instances. The main entry point is `main.py`:

```bash
python main.py
```

You can also load and solve an instance directly:

```python
from config import Config
from examples.loader import load_example_instance
from heuristics import GRASPSolver

q, layouts = load_example_instance(
	"examples/instances/instance_4_atoms_9_sites"
)
config = Config(seed=7, max_iterations=50)

result = GRASPSolver.solve_rd(layouts, q, config=config)

for layout_result in result.results:
	print(
		layout_result.layout_index,
		layout_result.objective_value,
		layout_result.elapsed_seconds,
	)

print("Best layout:", result.best_layout_index)
print("Best cost:", result.objective_value)
print("Total time:", result.total_elapsed_seconds)
```

For the random baseline, replace `GRASPSolver` with `RandomAssignmentSolver`.
