## FAQ

- How are travel times handled?
  - Links may include `travel_time` (hours). When missing, times default to `distance/40`. See `build_matrices` in `src/evrp/alns_solver.py`.

- Are charging stations required in the input?
  - If present in `.evrp`, they are used. Feasibility routines can insert station visits automatically when needed.

- Does OR-Tools model battery explicitly?
  - Not in this baseline; battery is enforced post-hoc by shared feasibility checks. Consider extending with a state-dependent dimension for full modeling.

- What is the solution JSON schema?
  - See docs/data-formats.md and `save_solution` in `src/evrp/alns_solver.py`.

