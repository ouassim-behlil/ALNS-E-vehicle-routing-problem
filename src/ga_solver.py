import random
from typing import List, Dict, Tuple, Optional

import numpy as np


def _expand_fleet(fleet: List[Dict]) -> List[Dict]:
    """Expand fleet definitions into individual vehicles."""
    vehicles = []
    for v in fleet:
        qty = int(v.get("quantity", 1))
        for _ in range(qty):
            vehicles.append(
                {
                    "capacity": float(v.get("max_load_capacity", 0)),
                    "energy": float(v.get("max_energy_capacity", 0)),
                    "consumption": float(v.get("consumption_per_km", 0.0)),
                }
            )
    if not vehicles:
        vehicles.append({"capacity": 0.0, "energy": 0.0, "consumption": 0.0})
    return vehicles


def _distance_matrix(n: int, links: Dict[Tuple[int, int], Dict]) -> np.ndarray:
    """Build dense distance matrix from links mapping."""
    big = 1e9
    dist = np.full((n, n), big, dtype=float)
    for i in range(n):
        dist[i, i] = 0.0
    for (i, j), data in links.items():
        d = data["distance"] if isinstance(data, dict) else float(data)
        dist[i, j] = d
        if dist[j, i] >= big:
            dist[j, i] = d
    return dist


def _build_routes(
    perm: List[int],
    vehicles: List[Dict],
    depot: int,
    dist: np.ndarray,
    demand: Dict[int, int],
) -> Tuple[List[List[int]], List[int], float]:
    """Construct routes from a customer permutation.

    Returns routes, unassigned customers and total distance.
    """
    routes: List[List[int]] = []
    unassigned: List[int] = []
    idx = 0
    customers = len(perm)

    for veh in vehicles:
        load = 0.0
        energy = veh["energy"]
        cons = veh["consumption"]
        route = [depot]
        current = depot
        while idx < customers:
            cust = perm[idx]
            d = demand[cust]
            dist_to = dist[current, cust]
            dist_back = dist[cust, depot]
            need_energy = cons * (dist_to + dist_back)
            if load + d <= veh["capacity"] and energy >= need_energy:
                # accept customer
                route.append(cust)
                load += d
                energy -= cons * dist_to
                current = cust
                idx += 1
            else:
                break
        if len(route) > 1:
            route.append(depot)
            routes.append(route)
        if idx >= customers:
            break

    if idx < customers:
        unassigned.extend(perm[idx:])

    cost = 0.0
    for r in routes:
        for i in range(len(r) - 1):
            cost += dist[r[i], r[i + 1]]
    return routes, unassigned, cost


def solve_ga(
    nodes: List[Dict],
    links: Dict[Tuple[int, int], Dict],
    requests: List[Dict],
    fleet: List[Dict],
    population_size: int = 50,
    generations: int = 200,
    mutation_rate: float = 0.1,
) -> Tuple[Dict, float]:
    """Solve the EVRP using a simple genetic algorithm.

    The returned solution has the same structure as solver.solve or
    ortools_solver.solve_ortools.
    """
    n = len(nodes)
    depot = next((n["id"] for n in nodes if n.get("type") == "depot"), 0)
    dist = _distance_matrix(n, links)
    vehicles = _expand_fleet(fleet)

    demand = {r["node_id"]: int(r.get("load", 0)) for r in requests}
    customers = list(demand.keys())
    if not customers:
        sol = {"routes": [], "unassigned": [], "vehicles": vehicles, "depot": depot}
        return sol, 0.0

    def evaluate(perm: List[int]):
        routes, unassigned, cost = _build_routes(perm, vehicles, depot, dist, demand)
        penalty = 1e6 * len(unassigned)
        return cost + penalty, routes, unassigned

    # initial population
    population: List[List[int]] = []
    for _ in range(population_size):
        p = customers[:]
        random.shuffle(p)
        population.append(p)

    best_cost = float("inf")
    best_routes: Optional[List[List[int]]] = None
    best_unassigned: Optional[List[int]] = None

    for _ in range(generations):
        scored = []
        for perm in population:
            cost, routes, unassigned = evaluate(perm)
            scored.append((cost, perm, routes, unassigned))
            if cost < best_cost:
                best_cost = cost
                best_routes = routes
                best_unassigned = unassigned
        scored.sort(key=lambda x: x[0])
        survivors = [perm for _, perm, _, _ in scored[: max(2, population_size // 2)]]

        offspring: List[List[int]] = []
        while len(offspring) + len(survivors) < population_size:
            parent1, parent2 = random.sample(survivors, 2)
            if len(customers) < 2:
                child = parent1[:]
            else:
                cut1, cut2 = sorted(random.sample(range(len(customers)), 2))
                child = [None] * len(customers)
                child[cut1:cut2] = parent1[cut1:cut2]
                fill = [c for c in parent2 if c not in child]
                it = iter(fill)
                for i in range(len(customers)):
                    if child[i] is None:
                        child[i] = next(it)
            if random.random() < mutation_rate and len(customers) >= 2:
                i, j = random.sample(range(len(customers)), 2)
                child[i], child[j] = child[j], child[i]
            offspring.append(child)
        population = survivors + offspring

    if best_routes is None:
        best_routes = []
    if best_unassigned is None:
        best_unassigned = customers
    solution = {
        "routes": best_routes,
        "unassigned": best_unassigned,
        "vehicles": vehicles,
        "depot": depot,
    }
    return solution, best_cost
