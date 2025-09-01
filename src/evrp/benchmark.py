import argparse
import time
import os
import sys
from pathlib import Path
from typing import List, Dict
# Flexible imports: work as package or standalone script
if __package__:
    from .problem_io import build_problem_from_evrp
    from .alns_solver import build_matrices, compute_route_distance
else:  # pragma: no cover - runtime path fix
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from evrp.problem_io import build_problem_from_evrp
    from evrp.alns_solver import build_matrices, compute_route_distance


def _solution_cost(solution: Dict, dist) -> float:
    """Compute total distance of all routes in solution using distance matrix."""
    total = 0.0
    for route in solution.get("routes", []):
        total += compute_route_distance(route, dist)
    return total


def _route_time(route: List[int], dist, mu) -> float:
    """Compute expected time of a route using mu (fallback to dist/40)."""
    t = 0.0
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        try:
            t += float(mu[a, b])
        except Exception:
            # Fallback to distance-based time at 40 units/hour
            t += float(dist[a, b]) / 40.0
    return t


def _solution_time_variance(solution: Dict, dist, mu) -> float:
    """Compute variance of route durations (expected time) across all routes."""
    routes = solution.get("routes", [])
    if not routes:
        return 0.0
    times = [_route_time(r, dist, mu) for r in routes]
    mean_t = sum(times) / len(times)
    var = sum((t - mean_t) ** 2 for t in times) / len(times)
    return var


def run_instance(path: Path, solvers: List[str], iterations: int) -> List[Dict]:
    """Run selected solvers on a single instance file and return metrics."""
    nodes, links, requests, fleet = build_problem_from_evrp(str(path))
    dist, mu, _ = build_matrices(nodes, links)

    results = []
    for name in solvers:
        start = time.perf_counter()
        if name == "alns":
            from .alns_solver import solve_alns  # lazy import
            sol, _ = solve_alns(nodes, links, requests, fleet, iterations=iterations)
        elif name == "ga":
            from .ga_solver import solve_ga  # lazy import
            sol, _ = solve_ga(nodes, links, requests, fleet, generations=iterations)
        elif name == "ortools":
            from .ortools_solver import solve_ortools  # lazy import
            sol, _ = solve_ortools(nodes, links, requests, fleet)
        else:
            continue
        elapsed = time.perf_counter() - start
        cost = _solution_cost(sol, dist)
        variance = _solution_time_variance(sol, dist, mu)
        results.append({
            "instance": path.name,
            "solver": name,
            "cost": cost,
            "variance": variance,
            "time": elapsed,
        })
    return results


def benchmark(data_dir: str = "data", solvers: List[str] = None, iterations: int = 50,
              limit: int = None) -> List[Dict]:
    if solvers is None:
        solvers = ["alns", "ortools", "ga"]
    files = sorted(Path(data_dir).glob("*.evrp"))
    if limit is not None:
        files = files[:limit]
    all_results: List[Dict] = []
    for inst in files:
        all_results.extend(run_instance(inst, solvers, iterations))
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark EVRP solvers across instances")
    parser.add_argument("--data-dir", default="data", help="Directory containing .evrp files")
    parser.add_argument("--solvers", default="alns,ortools,ga",
                        help="Comma-separated list of solvers to run")
    parser.add_argument("--iterations", type=int, default=400,
                        help="Iterations for ALNS and GA solvers")
    parser.add_argument("--limit", type=int, help="Limit number of instances")
    args = parser.parse_args()

    solvers = [s.strip().lower() for s in args.solvers.split(",") if s.strip()]
    results = benchmark(args.data_dir, solvers, args.iterations, args.limit)
    if not results:
        print("No results")
        return

    header = f"{'instance':30} {'solver':10} {'cost':>10} {'variance':>10} {'time(s)':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['instance']:30} {r['solver']:10} {r['cost']:10.2f} {r['variance']:10.2f} {r['time']:10.2f}")


if __name__ == "__main__":
    main()
