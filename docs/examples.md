## Examples

### Solve and plot a small instance

```
python -m evrp.problem_io data/E-n29-k4-s7.evrp --iterations 400 --output output/solution.json
python -m evrp.plotting --solution output/solution.json --out output/solution_plot.png
```

### Compare three solvers on a dataset

```
python -m evrp.benchmark --data-dir data --solvers alns,ortools,ga --iterations 300
```

### Robustness metrics for a solution

```
python -m evrp.robustness data/E-n29-k4-s7.evrp --iterations 500 --scenarios 200 --demand-std 0.15 --time-std 0.1
```

