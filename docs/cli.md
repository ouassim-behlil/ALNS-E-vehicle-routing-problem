## CLI Usage

### ALNS Solver

File: `src/evrp/alns_solver.py:main`

```
python -m evrp.alns_solver \
  --instance data/E-n29-k4-s7.evrp \
  --iterations 400 \
  --weight-time 1.0 \
  --weight-balance 0.6 \
  --solver alns
```

Arguments:
- `--instance`: `.xml` or `.evrp` instance file
- `--iterations`: ALNS/GA iterations
- `--weight-time`: weight for expected travel time objective
- `--weight-balance`: variance (balance) weight
- `--solver`: `alns`, `ortools`, `ga`, or `all`

Outputs: prints cost and routes, writes `output/solution_{solver}.json`.

### Benchmark

File: `src/evrp/benchmark.py:main`

```
python -m evrp.benchmark --data-dir data --solvers alns,ortools,ga --iterations 400 --limit 10
```

Arguments:
- `--data-dir`: directory of `.evrp` files
- `--solvers`: comma-separated list: `alns,ortools,ga`
- `--iterations`: iterations for ALNS/GA
- `--limit`: optional number of instances to benchmark

### Single-file Solve

File: `src/evrp/problem_io.py:main`

```
python -m evrp.problem_io data/E-n29-k4-s7.evrp --iterations 500 --output output/solution.json
```

### Plotting

File: `src/evrp/plotting.py` (module-level CLI)

```
python -m evrp.plotting --solution output/solution.json --out output/solution_plot.png
python -m evrp.plotting --instance data/E-n29-k4-s7.evrp --out output/E-n29-k4-s7_nodes.png
```

### Robustness

File: `src/evrp/robustness.py:main`

```
python -m evrp.robustness data/E-n29-k4-s7.evrp --iterations 400 --scenarios 200 --demand-std 0.1 --time-std 0.1
```

