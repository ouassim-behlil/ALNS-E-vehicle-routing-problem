## Overview

The EVRP toolkit is a small, self-contained codebase for experimenting with Electric Vehicle Routing Problem solvers and utilities. It provides:

- Metaheuristic ALNS solver with destroy/repair operators
- Baselines using Google OR-Tools and a simple Genetic Algorithm
- Shared feasibility checks including charging-station insertion
- Plotting for instances and solutions
- Robustness and sensitivity analysis via Monte Carlo
- A benchmarking driver for batches of instances

## Concepts

- Nodes: depot, customers, charging stations
- Requests: customer demands (positive loads only)
- Fleet: vehicle types with quantity, capacity, energy and consumption
- Links: directed arcs with distance and optional travel time
- Solution: list of routes (each from depot back to depot) plus metadata

## Code Layout

- src/evrp/alns_solver.py — ALNS and helpers
- src/evrp/ortools_solver.py — OR-Tools baseline
- src/evrp/ga_solver.py — GA baseline
- src/evrp/problem_io.py — .evrp parsing and builder
- src/evrp/constraints.py — feasibility checks and station insertion
- src/evrp/plotting.py — plotting helpers
- src/evrp/benchmark.py — benchmarking utility
- src/evrp/robustness.py — robustness utilities

