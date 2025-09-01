## API — evrp.plotting

Source: `src/evrp/plotting.py`

- `plot_sol(sol_path, out_path=None, show=False, figsize=(10,8)) -> str`
  - Plots routes, start/end markers and step annotations; returns saved image path.

- `plot_inst(evrp_path, out_path=None, show=False, figsize=(10,8)) -> str`
  - Plots instance nodes with IDs, stations and depot.

Module provides a convenient CLI when invoked via `python -m evrp.plotting`.

