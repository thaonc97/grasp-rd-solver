
"""Project-level defaults for register design experiments."""

import logging
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
	"""Runtime settings; edit this file or create a custom Config for experiments."""

	c6: float = 500.0
	target_diagonal: float = -1.0
	min_site_distance: float = 1e-8
	max_radius: Optional[float] = None
	alpha: float = 0.15
	seed: Optional[int] = 42
	max_iterations: int = 50
	local_search_strategy: str = "first_improvement"
	verbose: bool = True
	log_level: int = logging.DEBUG # Choose logging.DEBUG or logging.INFO


DEFAULT_CONFIG = Config()

