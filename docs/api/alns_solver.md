## API — evrp.alns_solver

Source: `src/evrp/alns_solver.py`

- `parse_instance_file(path) -> (nodes, links, requests, fleet, drivers)`
  - Parses legacy XML or `.evrp` format into internal structures.

- `build_matrices(nodes, links, sigma_alpha=0.2) -> (dist, mu, sigma)`
  - Distance, expected travel time, and stddev matrices from links. Falls back to `mu = dist/40` when missing.

- `build_initial_solution(nodes, requests, fleet, dist, drivers=None) -> solution`
  - Greedy seeding with capacity/energy checks and station recharge.

- `compute_route_distance(route, dist) -> float`
  - Sum of edge distances.

- `assign_drivers_to_routes(routes, drivers, dist, mu_matrix=None) -> (assignments, driver_loads)`
  - LPT-style balancing based on route durations.

- `evaluate_solution(solution, dist, demands, ..., mu_matrix=None, sigma_matrix=None, monte_carlo=False, ...) -> float`
  - Combines expected time and balance; penalizes infeasibility/unserved/duplicates; considers energy and station insertion logic.

- Destroy/Repair operators:
  - `destroy_remove_random(state, rng, n_remove=2)`
  - `repair_insert_greedy(state, rng)`
  - `repair_swap_routes(state, rng)`

- `solve_alns(nodes, links, requests, fleet, drivers=None, iterations=200, weight_time=1.0, weight_balance=0.0, monte_carlo=False, ...) -> (solution, cost)`
  - Runs ALNS with SA acceptance, RouletteWheel selection and MaxIterations stopping.

- `print_routes(solution, dist, prefix="")`
- `save_solution(solution, nodes, filename='output/solution.json', prefix="")`

CLI entry: `python -m evrp.alns_solver --help`

