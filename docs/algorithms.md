## Algorithms

### ALNS (Adaptive Large Neighborhood Search)

Implementation: `src/evrp/alns_solver.py`

Key components:
- Initial solution: greedy, capacity- and energy-aware with station recharge
- Destroy operators: random removal of customers (`n_remove` in {2,4})
- Repair operators: greedy insertion, swap between routes
- Acceptance: Simulated Annealing with exponential cooling
- Operator selection: RouletteWheel adaptive scheme

Objective (evaluate_solution):
- Expected travel time (using `mu` matrix; falls back to distance/40)
- Optional variance (balance) of tour durations
- Heavy penalties for unserved customers and duplicates
- Capacity and energy feasibility considered; station insertion when needed

### OR-Tools Baseline

Implementation: `src/evrp/ortools_solver.py`

- Builds a reduced VRP on depot + required customers
- Capacitated VRP with distance cost; battery is enforced post‑hoc via constraints module (station insertion and checks)
- Uses PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH; 10s time limit

### Genetic Algorithm Baseline

Implementation: `src/evrp/ga_solver.py`

- Permutation-based GA on customers
- Route builder packs customers into vehicles based on capacity/energy
- Penalizes unassigned; then post-processes to insert missing via station insertion; feasibility enforced

