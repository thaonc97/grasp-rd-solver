"""Generate a larger benchmark instance with 50 atoms."""

import json
from pathlib import Path

import numpy as np


base = Path(__file__).resolve().parent / "instances" / "instance_50_atoms_100_sites"
base.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(1234)
raw = rng.uniform(0.1, 2.0, size=(50, 50))
Q = (raw + raw.T) / 2.0
np.fill_diagonal(Q, -1.0)
(base / "Q.json").write_text(json.dumps(Q.tolist()))

coords = np.linspace(-9.0, 9.0, 10)
X, Y = np.meshgrid(coords, coords)
layout = np.column_stack([X.ravel(), Y.ravel()])
(base / "layer_0.json").write_text(json.dumps({"sites": layout.tolist()}))

print(f"Created {base} with Q shape={Q.shape} and {layout.shape[0]} sites")
