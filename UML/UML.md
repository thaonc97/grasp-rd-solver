# UML : Question 6

The diagrams below describe the main domain objects and solver flow. They use Mermaid syntax and render in GitHub and VS Code Markdown previews that support Mermaid.

## Class Diagram

```mermaid
classDiagram
    direction LR

    class Config {
        <<dataclass>>
        +float c6
        +float target_diagonal
        +float min_site_distance
        +float max_radius
        +float alpha
        +int seed
        +int max_iterations
        +str local_search_strategy
        +int log_level
    }

    class RegisterInstance {
        +ndarray W
        +ndarray sites
        +float C6
        +int N
        +int M
        +float min_site_distance
        +float max_radius
        +ndarray U
        +__init__(target_Q, site_positions, C6, min_site_distance, max_radius, config)
    }

    class RegisterLayout {
        +RegisterInstance instance
        +int N
        +int M
        +list pi
        +float cost
        +bool is_complete
        +compute_full_cost() float
        +evaluate_swap_delta(i, j) float
        +apply_swap(i, j, delta) None
        +copy() RegisterLayout
    }

    class SolveResult {
        <<dataclass>>
        +RegisterLayout solution
        +float objective_value
        +list history
        +float elapsed_seconds
        +int layout_index
    }

    class RDSolveResult {
        <<dataclass>>
        +list results
        +int best_layout_index
        +SolveResult best_result
        +float total_elapsed_seconds
        +objective_value() float
    }

    class GreedyRandomizedBuilder {
        +RegisterInstance instance
        +float alpha
        +Random rng
        +build() RegisterLayout
    }

    class LocalSearchOptimizer {
        +RegisterInstance instance
        +str strategy
        +optimize(layout) RegisterLayout
    }

    class GRASPSolver {
        +RegisterInstance instance
        +Config config
        +Random rng
        +GreedyRandomizedBuilder builder
        +LocalSearchOptimizer local_search
        +solve(max_iterations) SolveResult
        +solve_rd(layouts, target_Q, ...) RDSolveResult
    }

    class Loader {
        <<examples.loader module>>
        +load_example_instance(instance_dir) tuple
        +solve_example_instance(instance_dir, ...) list
        +solve_all_examples(base_dir, ...) list
    }

    Config ..> RegisterInstance : hardware defaults
    Config ..> GRASPSolver : heuristic defaults

    RegisterInstance "1" o-- "0..*" RegisterLayout : provides problem data
    RegisterLayout --> RegisterInstance : references

    GRASPSolver *-- GreedyRandomizedBuilder : creates
    GRASPSolver *-- LocalSearchOptimizer : creates
    GreedyRandomizedBuilder --> RegisterInstance : reads W/U
    GreedyRandomizedBuilder ..> RegisterLayout : builds
    LocalSearchOptimizer --> RegisterInstance : reads N
    LocalSearchOptimizer --> RegisterLayout : improves
    GRASPSolver --> RegisterInstance : solves
    GRASPSolver --> SolveResult : returns
    GRASPSolver --> RDSolveResult : returns multi-layout (RD) result

    SolveResult *-- RegisterLayout : contains solution
    RDSolveResult *-- SolveResult : contains results
    Loader ..> RegisterInstance : constructs
    Loader ..> GRASPSolver : selects
```

## Module Responsibilities

- `config.py`: immutable runtime and heuristic defaults.
- `models.py`: input validation, site interactions, assignments, objective values, and result records.
- `heuristics.py`: GRASP construction and local search.
- `examples/loader.py`: JSON instance discovery, loading, and solver dispatch.
- `main.py` and `scripts/`: runnable experiment entry points and plotting/comparison utilities.
