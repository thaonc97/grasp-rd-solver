# High-Level UML: Question 6

The RD problem consists of a target QUBO matrix and multiple calibrated layouts. The GRASP heuristic solves each layout and keeps the assignment with the lowest cost. Its two main phases are randomized greedy construction and swap-based local search. `RDProblem` is a conceptual input bundle; the current API passes its target matrix and layouts separately to `solve_rd`.

```mermaid
classDiagram
    direction LR

    class RDProblem {
        <<conceptual; not implemented>>
        Multi-layout RD input
    }

    class RegisterInstance {
        <<fixed-layout problem>>
    }

    class RegisterLayout {
        <<atom-to-site assignment>>
    }

    class GRASPSolver {
        <<multi-layout heuristic>>
    }

    class GreedyRandomizedBuilder {
        <<construction phase>>
    }

    class LocalSearchOptimizer {
        <<swap improvement phase>>
    }

    RDProblem "1" *-- "1..*" RegisterInstance : contains candidate layouts
    GRASPSolver --> RDProblem : solves full RD problem
    GRASPSolver --> RegisterInstance : solves each layout
    GRASPSolver *-- GreedyRandomizedBuilder : constructs
    GRASPSolver *-- LocalSearchOptimizer : improves
    GreedyRandomizedBuilder ..> RegisterLayout : builds feasible assignment
    LocalSearchOptimizer ..> RegisterLayout : improves by swaps
    RegisterLayout --> RegisterInstance : evaluated against
```