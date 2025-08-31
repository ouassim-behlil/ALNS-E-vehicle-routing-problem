from typing import List, Dict, Tuple, Optional
import numpy as np


def expand_vehicle_fleet(fleet: List[Dict]) -> List[Dict]:
    """Expand a fleet definition into individual vehicle entries.

    Each item in ``fleet`` may define a vehicle type with a ``quantity``.
    This function returns a list with one dict per physical vehicle, using
    normalized keys: ``capacity``, ``energy``, ``consumption``.
    """
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


def build_distance_matrix(n: int, links: Dict[Tuple[int, int], Dict]) -> np.ndarray:
    """Create a dense distance matrix from link data."""
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


def check_route_feasibility(
    route: List[int],
    vehicles: List[Dict],
    dist: np.ndarray,
    nodes: List[Dict],
    demands: Dict[int, int],
) -> bool:
    """Return True if at least one vehicle can serve the route.

    Checks load capacity and battery feasibility with recharging at
    charging stations.
    """
    depot = route[0] if route else 0
    load = sum(demands.get(c, 0) for c in route if c != depot)
    n = len(nodes)
    for v in vehicles:
        cap = v.get("capacity", v.get("max_load_capacity", float("inf")))
        if load > cap:
            continue
        energy_cap = v.get("energy", v.get("max_energy_capacity", float("inf")))
        consumption = v.get("consumption", v.get("consumption_per_km", 0.2))
        battery = energy_cap
        feasible = True
        for i in range(len(route) - 1):
            a = route[i]
            b = route[i + 1]
            energy_needed = dist[a, b] * consumption
            battery -= energy_needed
            if 0 <= b < n and nodes[b].get("type") == "charging_station":
                battery = energy_cap
            if battery < -1e-6:
                feasible = False
                break
        if feasible:
            return True
    return False


def insert_charging_stations(
    route: List[int],
    vehicles: List[Dict],
    dist: np.ndarray,
    nodes: List[Dict],
    demands: Dict[int, int],
) -> Optional[List[int]]:
    """Insert charging stations into a route to satisfy battery constraints.

    Returns a new route with stations added for at least one feasible
    vehicle, or ``None`` if no such adjustment exists.
    """
    if len(route) <= 2:
        return route
    stations = [n["id"] for n in nodes if n.get("type") == "charging_station"]
    depot = route[0]

    # precompute route load
    load = sum(demands.get(c, 0) for c in route if c != depot)

    def shortest_station_path(a: int, b: int, max_range: float) -> Optional[List[int]]:
        allowed = [a] + stations + [b]
        idx_map = {nid: i for i, nid in enumerate(allowed)}
        n_allowed = len(allowed)
        # adjacency: edges i->j if within range
        adj = [[] for _ in range(n_allowed)]
        for i in range(n_allowed):
            u = allowed[i]
            for j in range(n_allowed):
                if i == j:
                    continue
                v = allowed[j]
                if dist[u, v] <= max_range + 1e-9:
                    adj[i].append((j, dist[u, v]))
        # Dijkstra
        import heapq
        src = idx_map[a]
        tgt = idx_map[b]
        INF = 1e18
        dist_cost = [INF] * n_allowed
        prev = [-1] * n_allowed
        dist_cost[src] = 0.0
        heap = [(0.0, src)]
        while heap:
            dcur, uidx = heapq.heappop(heap)
            if dcur > dist_cost[uidx] + 1e-12:
                continue
            if uidx == tgt:
                break
            for vidx, w in adj[uidx]:
                nd = dcur + w
                if nd + 1e-12 < dist_cost[vidx]:
                    dist_cost[vidx] = nd
                    prev[vidx] = uidx
                    heapq.heappush(heap, (nd, vidx))
        if dist_cost[tgt] >= INF:
            return None
        # reconstruct
        path = []
        cur = tgt
        while cur != -1:
            path.append(allowed[cur])
            cur = prev[cur]
        path.reverse()
        return path

    for v in vehicles:
        cap = v.get("capacity", v.get("max_load_capacity", float("inf")))
        if load > cap:
            continue
        energy_cap = v.get("energy", v.get("max_energy_capacity", 0.0))
        cons = v.get("consumption", v.get("consumption_per_km", 0.2))
        if energy_cap <= 0 or cons <= 0:
            continue
        max_range = energy_cap / cons
        aug = [route[0]]
        feasible = True
        for i in range(len(route) - 1):
            a = aug[-1]
            b = route[i + 1]
            path = shortest_station_path(a, b, max_range)
            if path is None:
                feasible = False
                break
            # append path without duplicating the start node
            aug.extend(path[1:])
        if feasible:
            return aug
    return None


def enforce_solution_feasibility(
    solution: Dict,
    nodes: List[Dict],
    dist: np.ndarray,
    demands: Dict[int, int],
) -> Dict:
    """Ensure all routes satisfy capacity/energy checks; adjust if needed.

    Infeasible routes are split into single-customer trips if possible;
    otherwise customers are moved to ``unassigned``.
    """
    routes = solution.get("routes", [])
    vehicles = solution.get("vehicles", [])
    depot = solution.get("depot", 0)
    new_routes: List[List[int]] = []
    unassigned: List[int] = []

    for r in routes:
        if len(r) <= 2:
            continue
        # Try to augment with charging stations if needed
        fixed = insert_charging_stations(r, vehicles, dist, nodes, demands)
        if fixed is None:
            # Not fixable as a whole; try single-customer routes with station insertion
            for c in r:
                if c == depot:
                    continue
                single = [depot, int(c), depot]
                fixed_single = insert_charging_stations(single, vehicles, dist, nodes, demands)
                if fixed_single is not None and check_route_feasibility(
                    fixed_single, vehicles, dist, nodes, demands
                ):
                    new_routes.append(fixed_single)
                else:
                    unassigned.append(int(c))
            continue
        if check_route_feasibility(fixed, vehicles, dist, nodes, demands):
            new_routes.append(fixed)
        else:
            # final guard; if still infeasible, try singles
            for c in r:
                if c == depot:
                    continue
                single = [depot, int(c), depot]
                fixed_single = insert_charging_stations(single, vehicles, dist, nodes, demands)
                if fixed_single is not None and check_route_feasibility(
                    fixed_single, vehicles, dist, nodes, demands
                ):
                    new_routes.append(fixed_single)
                else:
                    unassigned.append(int(c))
            continue

    solution["routes"] = new_routes
    solution["unassigned"] = unassigned
    return solution
