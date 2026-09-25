# Example instance folders

Each test instance lives in its own folder under `examples/instances/`.

## Folder convention

Use a descriptive folder name that includes the problem size, for example:

- `instance_4_atoms_9_sites/`
- `instance_6_atoms_12_sites/`
- `instance_8_atoms_16_sites/`

Inside each folder, keep a single input matrix and one or more layer files:

- `Q.json` or `Q.npy` describes the target interaction matrix.
- `layer_0.json`, `layer_1.json`, ... or `layout_0.json`, `layout_1.json`, ... describe site coordinates for each layer.

## Supported file formats

The loader accepts:

- `Q.json`, `Q.npy`, `target.json`, `target.npy`
- `layer_*.json`, `layout_*.json`, or `layers/*.json`
- `sites_*.json` and `layout.json`

Each layout file may be either a raw NumPy-style array or a dictionary containing a `sites`/`layout` key.

## Loading and solving

The repository includes helper code to discover and solve every example instance:

```bash
python -m examples.solve_instances
```

The loader API is available in `examples/loader.py`.
