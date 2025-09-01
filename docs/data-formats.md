## Data Formats

### Instance (.evrp)

TSPLIB-like format with header fields and sections:

- Header keys commonly used: `DIMENSION`, `VEHICLES`, `CAPACITY`, `ENERGY_CAPACITY`, `ENERGY_CONSUMPTION`
- Sections: `NODE_COORD_SECTION`, `DEMAND_SECTION`, `STATIONS_COORD_SECTION`, `DEPOT_SECTION`

Parsing implementation: `src/evrp/problem_io.py:parse_evrp_file`

### In-memory model

- Nodes: list of dicts with keys: `id`, `type` in {`depot`, `customer`, `charging_station`}, `lat`, `lon`
- Links: dict mapping `(i, j)` to `{distance: float, travel_time: float?}`
- Requests: list of dicts `{node_id: int, load: int}` (positive loads only)
- Fleet: list of dicts `{max_load_capacity, max_energy_capacity, consumption_per_km, quantity}`

Builder: `src/evrp/problem_io.py:build_problem_from_evrp`

### Solution (JSON)

Saved by `evrp.alns_solver.save_solution`:

- `routes`: list of routes, each a list of node ids `[depot, ..., depot]`
- `coords`: map of node id to `[x, y]` (added on save)
- `unassigned`: optional list of unserved customer ids
- `vehicles`: optional list of expanded vehicle dicts
- `driver_assignments`: optional mapping `route_index -> driver_id`

