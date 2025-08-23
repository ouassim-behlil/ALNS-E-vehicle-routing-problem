import random
import math
import os
import sys
from typing import Dict, Any

# Enable execution both as part of the ``src`` package and as a standalone script
if __package__:
    from .solver import build_mats, score
else:  # pragma: no cover - runtime path fix
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.solver import build_mats, score


def robustness_analysis(solution: Dict[str, Any],
                        nodes,
                        links,
                        requests,
                        fleet,
                        scenarios: int = 100,
                        demand_std: float = 0.1,
                        travel_time_std: float = 0.1) -> Dict[str, float]:
    """Evaluate solution robustness via Monte Carlo simulation.

    Parameters
    ----------
    solution: dict
        Solution returned by :func:`solve`.
    nodes, links, requests, fleet:
        Original problem data.
    scenarios: int
        Number of random scenarios to simulate.
    demand_std: float
        Standard deviation for Gaussian perturbation of customer demand
        (relative to original load).
    travel_time_std: float
        Standard deviation for Gaussian perturbation of travel times
        (relative to original time).

    Returns
    -------
    dict with keys ``mean_cost``, ``std_cost``, ``min_cost``, ``max_cost``
    and ``infeasible_rate`` representing the fraction of scenarios where the
    solution becomes infeasible.
    """
    base_dist, base_mu, _ = build_mats(nodes, links)
    base_demands = {r['node_id']: r['load'] for r in requests}

    costs = []
    infeasible = 0
    penalty = 1e5

    for _ in range(scenarios):
        perturbed_demands = {}
        for nid, load in base_demands.items():
            noise = random.gauss(0.0, demand_std)
            perturbed = max(0, load + int(round(load * noise)))
            perturbed_demands[nid] = perturbed

        perturbed_links = {}
        for (i, j), data in links.items():
            dist = data['distance'] if isinstance(data, dict) else data
            tt = data.get('travel_time') if isinstance(data, dict) else None
            if tt is None:
                tt = dist / 40.0
            tt *= max(0.0, 1.0 + random.gauss(0.0, travel_time_std))
            perturbed_links[(i, j)] = {'distance': dist, 'travel_time': tt}

        dist_mat, mu_mat, _ = build_mats(nodes, perturbed_links)
        c = score(solution, dist_mat, perturbed_demands, mu_matrix=mu_mat,
                  penalty_unserved=penalty, vehicles=fleet, nodes=nodes)
        if c >= penalty:
            infeasible += 1
        costs.append(c)

    mean_cost = sum(costs) / len(costs)
    var = sum((c - mean_cost) ** 2 for c in costs) / len(costs)
    std_cost = math.sqrt(var)

    return {
        'mean_cost': mean_cost,
        'std_cost': std_cost,
        'min_cost': min(costs),
        'max_cost': max(costs),
        'infeasible_rate': infeasible / scenarios,
    }


def main(argv=None):
    """Simple CLI for robustness analysis."""
    import argparse
    if __package__:
        from .file_solver import build_problem
        from .solver import solve
    else:  # pragma: no cover - runtime path fix
        from src.file_solver import build_problem
        from src.solver import solve

    parser = argparse.ArgumentParser(description='Robustness analysis for EVRP solutions')
    parser.add_argument('instance', help='Path to .evrp instance file')
    parser.add_argument('--iterations', type=int, default=500, help='ALNS iterations to generate solution')
    parser.add_argument('--scenarios', type=int, default=100, help='Number of perturbation scenarios')
    parser.add_argument('--demand-std', type=float, default=0.1, help='Std dev for demand perturbation (relative)')
    parser.add_argument('--time-std', type=float, default=0.1, help='Std dev for travel time perturbation (relative)')
    args = parser.parse_args(argv)

    nodes, links, requests, fleet = build_problem(args.instance)
    solution, _ = solve(nodes, links, requests, fleet, iterations=args.iterations)
    metrics = robustness_analysis(solution, nodes, links, requests, fleet,
                                  scenarios=args.scenarios,
                                  demand_std=args.demand_std,
                                  travel_time_std=args.time_std)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
