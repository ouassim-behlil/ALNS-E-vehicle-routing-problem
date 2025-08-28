import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


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
    # number of nodes and depot index
    n = len(nodes)
    depot = next((n['id'] for n in nodes if n['type'] == 'depot'), 0)

    # build distance matrix
    big = 1e9
    dist = np.full((n, n), big, dtype=float)
    for i in range(n):
        dist[i, i] = 0.0
    for (i, j), data in links.items():
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

    # Demands for load dimension
    demand = {r['node_id']: int(r['load']) for r in requests}
    demands = [demand.get(i, 0) for i in range(n)]

    # Build RoutingIndexManager and RoutingModel
    manager = pywrapcp.RoutingIndexManager(n, vehicle_count, depot)
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
    # Consumption is proportional to distance. At this level of abstraction we
    # simply cap the total energy used by each vehicle; visiting charging
    # stations does not currently reset the cumulative energy.
    consumption = vehicles[0]['consumption'] if vehicles else 0.0
    # Ignore battery capacity for feasibility in this basic model
    energy_caps = [10**9 for _ in vehicles]
    energy_cost = dist * consumption

    def energy_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(energy_cost[i, j] * 1000)

    energy_cb = routing.RegisterTransitCallback(energy_callback)
    routing.AddDimensionWithVehicleCapacity(
        energy_cb, 0, energy_caps, True, 'energy')

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
                if node != depot:
                    route.append(int(node))
                idx = solution.Value(routing.NextVar(idx))
            route.append(depot)
            if len(route) > 2:
                routes.append(route)
        obj = solution.ObjectiveValue() / 1000.0
        sol = {
            'routes': routes,
            'unassigned': [],
            'vehicles': vehicles,
            'depot': depot,
        }
        return sol, obj

    # If no solution, mark all requests as unassigned
    sol = {
        'routes': [],
        'unassigned': [r['node_id'] for r in requests],
        'vehicles': vehicles,
        'depot': depot,
    }
    return sol, float('inf')
