import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import os
import sys
if __package__:
    from .constraints import build_distance_matrix as build_full_dist, enforce_solution_feasibility
else:  # pragma: no cover - runtime path fix
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from evrp.constraints import build_distance_matrix as build_full_dist, enforce_solution_feasibility


def solve_ortools(nodes, links, requests, fleet):
    """Solve EVRP using Google OR-Tools.

    Parameters
    ----------
    nodes : list of dict
        Nodes with 'id' and 'type'.
    links : dict
        Mapping (i, j) -> {'distance': float} at minimum.
    requests : list of dict
        Each request has 'node_id' and 'load'.
    fleet : list of dict
        Vehicle types with capacities and quantities.

    Returns
    -------
    solution : dict
        Same structure as solver.solve output: contains 'routes', 'unassigned',
        'vehicles' and 'depot'.
    cost : float
        Objective value (total distance).
    """
    # Determine depot and reduce problem to only required customers (demand > 0)
    depot = next((n['id'] for n in nodes if n['type'] == 'depot'), 0)
    # Full distance matrix for feasibility enforcement later
    full_dist = build_full_dist(len(nodes), links)
    demand_map = {r['node_id']: int(r['load']) for r in requests}
    required = [nid for nid, load in demand_map.items() if load > 0]
    # Active set includes depot at index 0, then required customers
    active_nodes = [depot] + required
    id_to_active = {nid: idx for idx, nid in enumerate(active_nodes)}
    n = len(active_nodes)

    # build distance matrix on active nodes only
    big = 1e9
    dist = np.full((n, n), big, dtype=float)
    for i in range(n):
        dist[i, i] = 0.0
    for (i_id, j_id), data in links.items():
        if i_id in id_to_active and j_id in id_to_active:
            i = id_to_active[i_id]
            j = id_to_active[j_id]
            d = data['distance'] if isinstance(data, dict) else data
            dist[i, j] = d
            if dist[j, i] >= big:
                dist[j, i] = d

    # expand fleet into individual vehicles
    vehicles = []
    for v in fleet:
        for _ in range(int(v.get('quantity', 1))):
            vehicles.append({
                'capacity': float(v.get('max_load_capacity', 0)),
                'energy': float(v.get('max_energy_capacity', 0)),
                'consumption': float(v.get('consumption_per_km', 0.0)),
            })
    if not vehicles:
        vehicles.append({'capacity': 0, 'energy': 0, 'consumption': 0.0})

    vehicle_count = len(vehicles)

    # Demands vector aligned with active nodes (0 for depot)
    demands = [0] * n
    for nid, load in demand_map.items():
        if nid in id_to_active:
            demands[id_to_active[nid]] = load

    # Build RoutingIndexManager and RoutingModel
    # depot is at active index 0
    manager = pywrapcp.RoutingIndexManager(n, vehicle_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback used for cost and energy consumption
    def distance_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(dist[i, j] * 1000)

    transit_cb = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # Load dimension
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return demands[node]

    demand_cb = routing.RegisterUnaryTransitCallback(demand_callback)
    load_caps = [int(v['capacity']) for v in vehicles]
    routing.AddDimensionWithVehicleCapacity(
        demand_cb, 0, load_caps, True, 'load')

    # Energy dimension (battery)
    # The OR-Tools baseline does not model battery constraints. Vehicles are
    # assumed to have sufficient energy for their routes. A more accurate
    # formulation would require a state-dependent dimension or additional
    # variables to reset the battery at charging stations.

    # Search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_params.time_limit.FromSeconds(10)

    solution = routing.SolveWithParameters(search_params)

    routes = []
    if solution:
        for v in range(vehicle_count):
            idx = routing.Start(v)
            route = [depot]
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node != 0:
                    # map back to original node id
                    route.append(int(active_nodes[node]))
                idx = solution.Value(routing.NextVar(idx))
            route.append(depot)
            if len(route) > 2:
                routes.append(route)
        # identify missing requested customers not visited by routes
        assigned = set()
        for r in routes:
            for x in r:
                if x != depot:
                    assigned.add(int(x))
        missing = [nid for nid in required if nid not in assigned]

        # try to insert missing customers into existing routes (with charging support)
        if missing:
            try:
                from .constraints import insert_charging_stations, check_route_feasibility
            except Exception:
                from evrp.constraints import insert_charging_stations, check_route_feasibility
            for cust in missing:
                best = None
                best_delta = float('inf')
                for ri, r in enumerate(routes):
                    for pos in range(1, len(r)):
                        trial = r[:pos] + [int(cust)] + r[pos:]
                        fixed = insert_charging_stations(trial, vehicles, full_dist, nodes, demand_map)
                        if fixed is None:
                            continue
                        # OR-Tools already satisfies capacity, but recheck against shared constraints
                        if not check_route_feasibility(fixed, vehicles, full_dist, nodes, demand_map):
                            continue
                        delta = sum(full_dist[fixed[i], fixed[i+1]] for i in range(len(fixed)-1)) - \
                                sum(full_dist[r[i], r[i+1]] for i in range(len(r)-1))
                        if delta < best_delta:
                            best_delta = delta
                            best = (ri, fixed)
                if best is not None:
                    ri, fixed = best
                    routes[ri] = fixed
                else:
                    routes.append([depot, int(cust), depot])

        # enforce common feasibility/energy constraints
        sol = {
            'routes': routes,
            'unassigned': [],
            'vehicles': vehicles,
            'depot': depot,
        }
        sol = enforce_solution_feasibility(sol, nodes, full_dist, demand_map)
        # compute distance cost after enforcement
        cost = 0.0
        for r in sol.get('routes', []):
            for i in range(len(r) - 1):
                cost += full_dist[r[i], r[i + 1]]
        return sol, cost

    # If no solution, mark all requests as unassigned
    sol = {
        'routes': [],
        'unassigned': [r['node_id'] for r in requests],
        'vehicles': vehicles,
        'depot': depot,
    }
    return sol, float('inf')
