## API — evrp.ga_solver

Source: `src/evrp/ga_solver.py`

- `solve_ga(nodes, links, requests, fleet, population_size=50, generations=200, mutation_rate=0.1) -> (solution, cost)`
  - Permutation-based GA; route builder respects capacity/energy greedily; penalizes unassigned then repairs via station insertion; feasibility enforcement and cost recomputation.

