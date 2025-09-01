## Plotting

Module: `src/evrp/plotting.py`

### Plot a solution JSON

```
python -m evrp.plotting --solution output/solution.json --out output/solution_plot.png
```

- Draws each route in a different color, annotates step numbers
- Marks depot with a red star and start/end markers
- Auto-saves to `_plot.png` if `--out` is omitted

### Plot an instance map

```
python -m evrp.plotting --instance data/E-n29-k4-s7.evrp --out output/E-n29-k4-s7_nodes.png
```

- Plots node identifiers, stations (blue P marker) and depot (red star)
- Saves to `_nodes.png` by default

### Comparer plusieurs solutions dans une seule image

```
python -m evrp.plotting \
  --compare output/solution_alns.json output/solution_ortools.json output/solution_ga.json \
  --labels ALNS OR-Tools GA \
  --out output/solutions_compare.png
```

API Python:

```
from evrp.plotting import plot_compare_solutions
plot_compare_solutions([
    'output/solution_alns.json',
    'output/solution_ortools.json',
    'output/solution_ga.json',
], labels=['ALNS', 'OR-Tools', 'GA'], out_path='output/solutions_compare.png')
```
