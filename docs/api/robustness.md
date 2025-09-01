## API — evrp.robustness

Source: `src/evrp/robustness.py`

- `robustness_analysis(solution, nodes, links, requests, fleet, scenarios=100, demand_std=0.1, travel_time_std=0.1) -> Dict`
  - Monte Carlo evaluation of solution under perturbed demands and travel times; returns mean/std/min/max and failure rate.

- `sensitivity_analysis(solution, nodes, links, requests, fleet, scenarios=100, demand_std=0.1, travel_time_std=0.1) -> Dict`
  - Average deltas in cost for demand-only and time-only perturbations.

CLI entry: `python -m evrp.robustness --help`

