## API — evrp.constraints

Source: `src/evrp/constraints.py`

- `expand_vehicle_fleet(fleet) -> List[Dict]`
  - Expands vehicle types with `quantity` to individual normalized vehicles: `capacity`, `energy`, `consumption`.

- `build_distance_matrix(n, links) -> np.ndarray`
  - Dense symmetric distance matrix from `(i,j)->distance` links.

- `check_route_feasibility(route, vehicles, dist, nodes, demands) -> bool`
  - At least one vehicle can serve the route under load and energy constraints with recharging at stations.

- `insert_charging_stations(route, vehicles, dist, nodes, demands) -> Optional[List[int]]`
  - Inserts station visits to enforce battery range using a Dijkstra over allowed nodes.

- `enforce_solution_feasibility(solution, nodes, dist, demands) -> solution`
  - Fixes routes by inserting stations or splitting to single-customer trips; moves impossible customers to `unassigned`.

