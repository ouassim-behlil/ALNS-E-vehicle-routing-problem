## API — evrp.problem_io

Source: `src/evrp/problem_io.py`

- `parse_evrp_file(path) -> (header, node_coords, demands, stations, depot_ids)`
  - Reads TSPLIB-like `.evrp` files; extracts metadata, coordinates, demands, stations and depots.

- `build_problem_from_evrp(path) -> (nodes, links, requests, fleet)`
  - Builds the in-memory model: nodes with types, complete graph links with `distance` and `travel_time` (assumes 40 units/hour), positive-demand requests, single-type fleet.

CLI entry:

```
python -m evrp.problem_io <instance.evrp> --iterations 1000 --output output/solution_from_evrp.json
```

