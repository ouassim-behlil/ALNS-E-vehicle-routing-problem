## Benchmarking

Use `src/evrp/benchmark.py` to run one or more solvers over a directory of `.evrp` instances and report per-instance metrics.

Command:

```
python -m evrp.benchmark --data-dir data --solvers alns,ortools,ga --iterations 400 --limit 20
```

Metrics per (instance, solver):
- `cost`: total distance using distance matrix
- `variance`: variance of route expected times (from `mu`) across routes
- `time`: wall-clock seconds for the solver run

Reference: `src/evrp/benchmark.py`

