EVRP — Electric Vehicle Routing Problem Toolkit
==============================================

EVRP is a small, self‑contained toolkit for experimenting with Electric Vehicle Routing Problem (EVRP) solvers and utilities. It includes an ALNS metaheuristic, Google OR‑Tools and Genetic Algorithm baselines, plotting helpers, and robustness analysis utilities. Instances can be loaded from simple `.evrp` files (TSPLIB‑like).

Project highlights
- ALNS solver with pluggable destroy/repair operators
- OR‑Tools baseline for VRP with capacities
- Simple GA baseline for comparison
- Shared feasibility enforcement with charging‑station insertion
- Plotting for instances and solutions
- Robustness and sensitivity analysis via Monte Carlo

Quick start
- Python: 3.10+
- Install deps: `pip install -r requirements.txt` (or your env)

Run ALNS on an instance
  python -m evrp.alns_solver --instance data/E-n29-k4-s7.evrp --iterations 400

Compare solvers on a folder of instances
  python -m evrp.benchmark --data-dir data --solvers alns,ortools,ga --iterations 400

Solve a single .evrp file and save JSON
  python -m evrp.problem_io data/E-n29-k4-s7.evrp --iterations 500 --output output/solution.json

Plot a solution JSON
  python -m evrp.plotting --solution output/solution.json --out output/solution_plot.png

Plot an instance map
  python -m evrp.plotting --instance data/E-n29-k4-s7.evrp --out output/E-n29-k4-s7_nodes.png

Robustness analysis
  python -m evrp.robustness data/E-n29-k4-s7.evrp --iterations 400 --scenarios 200

Package layout
- src/evrp/__init__.py: public API exports
- src/evrp/alns_solver.py: ALNS solver and helpers
- src/evrp/ga_solver.py: Genetic Algorithm baseline
- src/evrp/ortools_solver.py: OR‑Tools baseline
- src/evrp/problem_io.py: `.evrp` file parsing and problem builder
- src/evrp/constraints.py: shared feasibility and station insertion
- src/evrp/plotting.py: plotting of instances and solutions
- src/evrp/benchmark.py: batch benchmarking across instances
- src/evrp/robustness.py: robustness and sensitivity analysis

Public API (most useful functions)
- evrp.solve_alns(nodes, links, requests, fleet, ...): run ALNS
- evrp.solve_ortools(nodes, links, requests, fleet): run OR‑Tools
- evrp.solve_ga(nodes, links, requests, fleet, ...): run GA
- evrp.build_problem_from_evrp(path): parse `.evrp` file into (nodes, links, requests, fleet)
- evrp.build_matrices(nodes, links): distance and travel‑time matrices
- evrp.compute_route_distance(route, dist): route distance helper
- evrp.evaluate_solution(solution, dist, demands, ...): objective evaluation
- evrp.enforce_solution_feasibility(solution, nodes, dist, demands): fix infeasible routes
- evrp.insert_charging_stations(route, vehicles, dist, nodes, demands): add stations
- evrp.plot_sol(path, out_path=None): plot a saved solution JSON
- evrp.plot_inst(path, out_path=None): plot an instance
- evrp.robustness_analysis(...), evrp.sensitivity_analysis(...)

Data model
- nodes: list of dicts with keys: `id`, `type` in {`depot`, `customer`, `charging_station`}, `lat`, `lon`
- links: dict mapping `(i, j)` to `{distance: float, travel_time: float?}`
- requests: list of dicts `{node_id: int, load: int}` (only for positive load customers)
- fleet: list of dicts `{max_load_capacity, max_energy_capacity, consumption_per_km, quantity}`

Solution format (JSON)
- routes: list of routes, each a list of node ids: `[depot, ..., depot]`
- coords: map of node id to `[x, y]` (added on save for plotting convenience)
- unassigned: optional list of unserved customer ids
- vehicles: optional list of expanded vehicle dicts
- driver_assignments: optional mapping `route_index -> driver_id`

Migration guide (old → new names)
- Module `src/solver.py` → `evrp/alns_solver.py`
  - `parse_instance` → `parse_instance_file`
  - `build_mats` → `build_matrices`
  - `init_solution` → `build_initial_solution`
  - `route_dist` → `compute_route_distance`
  - `assign_drivers` → `assign_drivers_to_routes`
  - `score` → `evaluate_solution`
  - `remove_random` → `destroy_remove_random`
  - `insert_greedy` → `repair_insert_greedy`
  - `swap_routes` → `repair_swap_routes`
  - `solve` → `solve_alns`
  - `print_solution` → `print_routes`
  - `save` → `save_solution`
- Module `src/constraints.py` → `evrp/constraints.py`
  - `expand_fleet` → `expand_vehicle_fleet`
  - `distance_matrix` → `build_distance_matrix`
  - `is_route_feasible` → `check_route_feasibility`
  - `insert_stations_for_route` → `insert_charging_stations`
  - `enforce_feasible_solution` → `enforce_solution_feasibility`
- `src/file_solver.py` → `evrp/problem_io.py`
  - `parse_file` → `parse_evrp_file`
  - `build_problem` → `build_problem_from_evrp`
- `src/compare_solvers.py` → `evrp/benchmark.py`
- `src/plot.py` → `evrp/plotting.py`
- `src/robustness.py` → `evrp/robustness.py`
- `src/ga_solver.py` → `evrp/ga_solver.py`
- `src/ortools_solver.py` → `evrp/ortools_solver.py`

Notes
- The ALNS solver prints initial and best objective values.
- The OR‑Tools baseline focuses on capacities; battery recharging is enforced post‑hoc via shared feasibility utilities.
- The GA baseline is intentionally simple; use for quick comparisons.

License
This project is licensed under the terms in LICENSE.

Further documentation
- Full docs with installation, usage, data formats, algorithms, benchmarking, robustness, plotting, and per-module API: see `docs/index.md`
