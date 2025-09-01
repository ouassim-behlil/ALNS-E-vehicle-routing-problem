## API — evrp.ortools_solver

Source: `src/evrp/ortools_solver.py`

- `solve_ortools(nodes, links, requests, fleet) -> (solution, cost)`
  - Capacitated VRP on depot + required customers; uses distance cost. Battery is handled by post‑hoc feasibility enforcement (station insertion and checks). Returns solution and total distance.

