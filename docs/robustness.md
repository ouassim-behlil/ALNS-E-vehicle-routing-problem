## Robustness Guide

Module: `src/evrp/robustness.py`

### robustness_analysis

Evaluates stability of a solution under perturbations of demand and travel times using Monte Carlo. Returns mean/std/min/max cost and failure rate.

Key args:
- `scenarios`: number of randomized scenarios
- `demand_std`: relative stddev for demand noise
- `travel_time_std`: relative stddev for travel time noise

### sensitivity_analysis

Measures average cost increase when changing demand only and travel time only.

CLI example:

```
python -m evrp.robustness data/E-n29-k4-s7.evrp --iterations 400 --scenarios 200
```

