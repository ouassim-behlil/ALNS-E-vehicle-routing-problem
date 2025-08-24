"""Utilities to evaluate robustness of an EVRP solution.

The module provides two high level helpers:

``robustness_analysis``
    Estimates the stability of a solution under random demand and travel
    time perturbations using a Monte Carlo approach.

``sensitivity_analysis``
    Computes simple measures of how sensitive the solution cost is to
    changes in demand or travel time individually.
"""

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


def robustness_analysis(
    solution: Dict[str, Any],
    nodes,
    links,
    requests,
    fleet,
    scenarios: int = 100,
    demand_std: float = 0.1,
    travel_time_std: float = 0.1,
) -> Dict[str, float]:
    """Evaluate solution robustness via Monte Carlo simulation.

    This function perturbs customer demands and travel times according to
    independent Gaussian noises and re-evaluates the solution cost.  The
    distribution of the resulting costs is a measure for the robustness of
    the provided solution.
    """

    base_demands = {r['node_id']: r['load'] for r in requests}
    dist_base, mu_base, _ = build_mats(nodes, links)

    costs = []
    infeasible_count = 0
    penalty = 1e5

    for _ in range(scenarios):
        # --- perturb demands ---
        demands = {}
        for nid, load in base_demands.items():
            noise = random.gauss(0.0, demand_std)
            demands[nid] = max(0, load + int(round(load * noise)))

        # --- perturb travel times ---
        mu_mat = mu_base.copy()
        for (i, j) in links.keys():
            noise = random.gauss(0.0, travel_time_std)
            mu_mat[i, j] = mu_base[i, j] * max(0.0, 1.0 + noise)

        c = score(
            solution,
            dist_base,
            demands,
            mu_matrix=mu_mat,
            penalty_unserved=penalty,
            vehicles=fleet,
            nodes=nodes,
        )
        if c >= penalty:
            infeasible_count += 1  # count scenarios that violate constraints
        costs.append(c)

    mean_cost = sum(costs) / len(costs)
    var = sum((c - mean_cost) ** 2 for c in costs) / len(costs)
    std_cost = math.sqrt(var)

    failure_rate = infeasible_count / scenarios
    success_rate = 1.0 - failure_rate

    return {
        "mean_cost": mean_cost,
        "std_cost": std_cost,
        "min_cost": min(costs),
        "max_cost": max(costs),
        "failure_rate": failure_rate,
        "success_rate": success_rate,
    }


def sensitivity_analysis(
    solution: Dict[str, Any],
    nodes,
    links,
    requests,
    fleet,
    scenarios: int = 100,
    demand_std: float = 0.1,
    travel_time_std: float = 0.1,
) -> Dict[str, float]:
    """Quantify solution sensitivity to demand and travel time changes.

    Two separate Monte Carlo experiments are run: one perturbing only the
    demands and another perturbing only the travel times.  The returned
    values represent the average increase in cost compared to the base
    scenario.
    """

    base_demands = {r['node_id']: r['load'] for r in requests}
    dist_base, mu_base, _ = build_mats(nodes, links)
    penalty = 1e5

    base_cost = score(
        solution,
        dist_base,
        base_demands,
        mu_matrix=mu_base,
        penalty_unserved=penalty,
        vehicles=fleet,
        nodes=nodes,
    )

    def mean_cost(perturb_demand: bool, perturb_time: bool) -> float:
        costs = []
        for _ in range(scenarios):
            demands = base_demands.copy()
            if perturb_demand:
                for nid, load in base_demands.items():
                    noise = random.gauss(0.0, demand_std)
                    demands[nid] = max(0, load + int(round(load * noise)))

            mu_mat = mu_base.copy()
            if perturb_time:
                for (i, j) in links.keys():
                    noise = random.gauss(0.0, travel_time_std)
                    mu_mat[i, j] = mu_base[i, j] * max(0.0, 1.0 + noise)

            c = score(
                solution,
                dist_base,
                demands,
                mu_matrix=mu_mat,
                penalty_unserved=penalty,
                vehicles=fleet,
                nodes=nodes,
            )
            costs.append(c)
        return sum(costs) / len(costs)

    demand_mean = mean_cost(True, False)
    time_mean = mean_cost(False, True)

    return {
        "base_cost": base_cost,
        "demand_sensitivity": demand_mean - base_cost,
        "time_sensitivity": time_mean - base_cost,
    }


def main(argv=None):
    """Simple CLI for robustness and sensitivity analysis."""
    import argparse
    if __package__:
        from .file_solver import build_problem
        from .solver import solve
    else:  # pragma: no cover - runtime path fix
        from src.file_solver import build_problem
        from src.solver import solve

    parser = argparse.ArgumentParser(
        description="Robustness analysis for EVRP solutions"
    )
    parser.add_argument("instance", help="Path to .evrp instance file")
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="ALNS iterations to generate solution",
    )
    parser.add_argument(
        "--scenarios", type=int, default=100, help="Number of perturbation scenarios"
    )
    parser.add_argument(
        "--demand-std",
        type=float,
        default=0.1,
        help="Std dev for demand perturbation (relative)",
    )
    parser.add_argument(
        "--time-std",
        type=float,
        default=0.1,
        help="Std dev for travel time perturbation (relative)",
    )
    args = parser.parse_args(argv)

    nodes, links, requests, fleet = build_problem(args.instance)
    solution, _ = solve(
        nodes, links, requests, fleet, iterations=args.iterations
    )

    metrics = robustness_analysis(
        solution,
        nodes,
        links,
        requests,
        fleet,
        scenarios=args.scenarios,
        demand_std=args.demand_std,
        travel_time_std=args.time_std,
    )

    sens = sensitivity_analysis(
        solution,
        nodes,
        links,
        requests,
        fleet,
        scenarios=args.scenarios,
        demand_std=args.demand_std,
        travel_time_std=args.time_std,
    )

    for k, v in metrics.items():
        print(f"{k}: {v}")
    for k, v in sens.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
