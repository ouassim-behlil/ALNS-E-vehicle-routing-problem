## Quick Start

Run ALNS on a single instance and save JSON:

```
python -m evrp.alns_solver --instance data/E-n29-k4-s7.evrp --iterations 400
```

Benchmark multiple solvers over a folder of `.evrp` files:

```
python -m evrp.benchmark --data-dir data --solvers alns,ortools,ga --iterations 400
```

Build a problem from a `.evrp` file and solve via ALNS:

```
python -m evrp.problem_io data/E-n29-k4-s7.evrp --iterations 500 --output output/solution.json
```

Plot a solution JSON:

```
python -m evrp.plotting --solution output/solution.json --out output/solution_plot.png
```

Plot an instance map:

```
python -m evrp.plotting --instance data/E-n29-k4-s7.evrp --out output/E-n29-k4-s7_nodes.png
```

