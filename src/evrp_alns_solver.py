import xml.etree.ElementTree as ET
import os
import copy
import math
import random
import numpy as np
from alns import ALNS, State
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations


# --- 1. Parse EVRP instance from XML ---
def parse_evrp_instance(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    nodes = []
    for node in root.find('nodes'):
        nodes.append({
            'id': int(node.find('id').text),
            'type': node.find('node_type').text,
            'lat': float(node.find('latitude').text),
            'lon': float(node.find('longitude').text)
        })

    # links will store both distance and optional mean travel time (mu)
    links = {}
    for link in root.find('links'):
        i = int(link.find('from').text)
        j = int(link.find('to').text)
        dist = float(link.find('distance').text)
        # travel_time may be present (in hours) or not
        travel_time_elem = link.find('travel_time')
        travel_time = None
        if travel_time_elem is not None and travel_time_elem.text is not None:
            try:
                travel_time = float(travel_time_elem.text)
            except Exception:
                travel_time = None
        links[(i, j)] = {'distance': dist, 'travel_time': travel_time}

    requests = []
    for req in root.find('requests'):
        requests.append({
            'node_id': int(req.find('node_id').text),
            'load': int(req.find('requested_load').text)
        })

    # Ensure every customer node has a request (default load=1) -- requirement: all customers must be visited
    customer_ids = [n['id'] for n in nodes if n['type'] == 'customer']
    existing = set(r['node_id'] for r in requests)
    for cid in customer_ids:
        if cid not in existing:
            requests.append({'node_id': cid, 'load': 1})

    fleet = []
    for v in root.find('fleet'):
        fleet.append({
            'type': v.find('type').text if v.find('type') is not None else 'vehicle',
            'max_load_capacity': float(v.find('max_load_capacity').text),
            'max_energy_capacity': float(v.find('max_energy_capacity').text) if v.find('max_energy_capacity') is not None else 0.0,
            'consumption_per_km': float(v.find('consumption_per_km').text) if v.find('consumption_per_km') is not None else 0.2,
            'quantity': int(v.find('quantity').text)
        })

    # parse drivers if present
    drivers = []
    drivers_elem = root.find('drivers')
    if drivers_elem is not None:
        for d in drivers_elem:
            try:
                did = int(d.find('id').text)
            except Exception:
                did = None
            drivers.append({
                'id': did,
                'name': d.find('name').text if d.find('name') is not None else None,
                'shift_hours': float(d.find('shift_hours').text) if d.find('shift_hours') is not None else None,
                'skill_level': int(d.find('skill_level').text) if d.find('skill_level') is not None else None,
            })

    return nodes, links, requests, fleet, drivers


# --- 2. Build distance matrix ---
def build_distance_matrix(nodes, links):
    n = len(nodes)
    big = 1e6
    dist = np.full((n, n), big)
    mu = np.full((n, n), big)  # expected travel times (in hours or km-based proxy)
    for i in range(n):
        dist[i, i] = 0.0
        mu[i, i] = 0.0
    for (i, j), data in links.items():
        if 0 <= i < n and 0 <= j < n:
            d = data['distance'] if isinstance(data, dict) else data
            dist[i, j] = d
            # if travel_time provided, use that as mu (convert if necessary), otherwise use distance/avg_speed heuristic
            tt = None
            if isinstance(data, dict):
                tt = data.get('travel_time')
            if tt is None:
                # fallback: assume average speed 40 km/h to convert distance (km) to hours
                mu[i, j] = dist[i, j] / 40.0
            else:
                # travel_time may already be in hours or seconds; try to detect
                if tt > 24:  # likely seconds -> convert to hours
                    mu[i, j] = tt / 3600.0
                else:
                    mu[i, j] = tt
    # symmetrize when appropriate
    for i in range(n):
        for j in range(n):
            if dist[i, j] < big and dist[j, i] >= big:
                dist[j, i] = dist[i, j]
            if mu[i, j] < big and mu[j, i] >= big:
                mu[j, i] = mu[i, j]
    return dist, mu


# --- 3. Simple initial solution (greedy capacity-respecting) ---
def initial_solution(nodes, requests, fleet, dist, drivers=None):
    depot_id = next((n['id'] for n in nodes if n['type'] == 'depot'), 0)
    customer_demands = {r['node_id']: r['load'] for r in requests}
    customers = list(customer_demands.keys())

    # expand fleet into vehicles (each vehicle type may have many identical units)
    vehicles = []
    for v in fleet:
        for _ in range(int(v['quantity'])):
            vehicles.append({'capacity': v['max_load_capacity'], 'energy': v['max_energy_capacity'], 'consumption': v['consumption_per_km']})

    routes = []
    unassigned = set(customers)

    # Helper to check if a route can be served by at least one vehicle (capacity + energy)
    def route_can_be_served(route):
        load = sum(customer_demands.get(c, 0) for c in route if c != depot_id)
        distance = route_distance(route, dist)
        for v in vehicles:
            if load <= v['capacity']:
                # energy needed depends on consumption_per_km (kWh/km) and distance (km)
                energy_needed = distance * v.get('consumption', v.get('consumption_per_km', 0.2))
                if energy_needed <= v['energy']:
                    return True
        return False

    # assign customers greedily; allow opening as many routes as needed (vehicles may do multiple trips)
    while unassigned:
        route = [depot_id]
        load = 0
        # greedily append nearest feasible customer until no more fit
        while True:
            last = route[-1]
            candidates = []
            for c in list(unassigned):
                demand = customer_demands[c]
                if load + demand <= max(v['capacity'] for v in vehicles) if vehicles else True:
                    candidates.append((dist[last, c], c))
            if not candidates:
                break
            candidates.sort()
            chosen = candidates[0][1]
            route.append(chosen)
            load += customer_demands[chosen]
            # if route cannot be served by any vehicle after adding, undo and stop
            if not route_can_be_served(route):
                route.pop()
                unassigned.add(chosen)
                break
            unassigned.remove(chosen)
        route.append(depot_id)
        # only add route if it serves at least one customer
        if len(route) > 2:
            routes.append(route)
        else:
            # couldn't add any customer (maybe no vehicle types) -> break to avoid infinite loop
            break

    # assign drivers to routes in round-robin if drivers provided
    driver_assignments = {}
    if drivers and len(routes) > 0:
        for idx, route in enumerate(routes):
            driver = drivers[idx % len(drivers)]
            driver_assignments[idx] = driver.get('id')

    solution = {'routes': routes, 'unassigned': list(unassigned), 'vehicles': vehicles, 'depot': depot_id, 'driver_assignments': driver_assignments, 'drivers': drivers}
    return solution


# --- 4. Objective and helpers ---
def route_distance(route, dist):
    d = 0.0
    for i in range(len(route) - 1):
        a = route[i]
        b = route[i + 1]
        d += dist[a, b]
    return d


def assign_drivers_to_routes(routes, drivers, dist, mu_matrix=None):
    """Assign drivers to routes trying to balance total route durations (LPT heuristic).

    Returns a dict mapping route_index -> driver_id and a dict driver_id -> total_time_hours
    """
    if not drivers:
        return {}
    # compute route durations
    route_times = []
    for idx, r in enumerate(routes):
        t = 0.0
        for i in range(len(r) - 1):
            a = r[i]
            b = r[i + 1]
            if mu_matrix is not None:
                t += mu_matrix[a, b]
            else:
                # fallback convert distance to time
                # assume 40 km/h
                t += dist[a, b] / 40.0
        route_times.append((idx, t))

    # sort routes by decreasing time
    route_times.sort(key=lambda x: x[1], reverse=True)

    # initialize driver loads
    driver_loads = {d['id']: 0.0 for d in drivers}
    assignments = {}
    driver_ids = list(driver_loads.keys())
    for idx, t in route_times:
        # choose driver with smallest current load
        best_driver = min(driver_ids, key=lambda did: driver_loads[did])
        assignments[idx] = best_driver
        driver_loads[best_driver] += t

    return assignments, driver_loads

def evaluate(solution, dist, customer_demands, weight_time=1.0, weight_balance=0.0, mu_matrix=None, penalty_unserved=1e5):
    total = 0.0
    routes = solution.get('routes', [])
    unassigned = set(solution.get('unassigned', []))
    depot = solution.get('depot', 0)
    drivers = solution.get('drivers', None)
    driver_assignments = solution.get('driver_assignments', None)

    # Objective 1: expected total travel time (using mu if available, otherwise convert distance)
    time_cost = 0.0
    for r in routes:
        # sum mu along route
        for i in range(len(r) - 1):
            a = r[i]
            b = r[i + 1]
            if mu_matrix is not None:
                time_cost += mu_matrix[a, b]
            else:
                # fallback converting distance to time assuming 40 km/h
                time_cost += dist[a, b] / 40.0

    # Objective 2: variance of tour durations (T_k)
    Tks = []
    for r in routes:
        T_k = 0.0
        for i in range(len(r) - 1):
            a = r[i]
            b = r[i + 1]
            if mu_matrix is not None:
                T_k += mu_matrix[a, b]
            else:
                T_k += dist[a, b] / 40.0
        Tks.append(T_k)

    # combine objectives as weighted sum
    # time component
    total += weight_time * time_cost

    # balance component: variance
    balance = 0.0
    if weight_balance > 0 and len(Tks) > 0:
        mean_T = sum(Tks) / len(Tks)
        balance = sum((Tk - mean_T) ** 2 for Tk in Tks) / len(Tks)
        total += weight_balance * balance

    # penalize unassigned customers heavily
    total += penalty_unserved * len(unassigned)

    # check that each customer appears exactly once across routes
    all_assigned = [c for r in routes for c in r if c != depot]
    from collections import Counter
    cnt = Counter(all_assigned)
    # penalize duplicates heavily
    duplicates = sum(v - 1 for v in cnt.values() if v > 1)
    total += penalty_unserved * duplicates

    # capacity violations: compare route loads to vehicle capacities (if provided)
    vehicles = solution.get('vehicles', [])
    # helper: check if any vehicle can serve this route (capacity + energy)
    def route_feasible_by_fleet(route):
        load = sum(customer_demands.get(c, 0) for c in route if c != depot)
        distance = route_distance(route, dist)
        for v in vehicles:
            cap = v.get('capacity', v.get('max_load_capacity', float('inf')))
            if load > cap:
                continue
            consumption = v.get('consumption', v.get('consumption_per_km', 0.2))
            energy_needed = distance * consumption
            energy_cap = v.get('energy', v.get('max_energy_capacity', float('inf')))
            if energy_needed <= energy_cap:
                return True
        return False
    for i, r in enumerate(routes):
        load = sum(customer_demands.get(c, 0) for c in r if c != depot)
        if i < len(vehicles):
            cap = vehicles[i].get('capacity', float('inf'))
            if load > cap:
                # heavy penalty proportional to excess
                total += penalty_unserved * (load - cap)
        else:
            # route without vehicle (shouldn't happen) -> heavy penalty
            total += penalty_unserved * len([c for c in r if c != depot])
        # penalize route if no vehicle can serve it (regardless of route count)
        if not route_feasible_by_fleet(r):
            total += penalty_unserved * len([c for c in r if c != depot])

    # --- Driver shift and balance penalties ---
    # If drivers present, ensure assignments exist (recompute with LPT heuristic)
    if drivers is not None:
        if not driver_assignments:
            # compute assignments
            assignments, driver_loads = assign_drivers_to_routes(routes, drivers, dist, mu_matrix)
        else:
            assignments = driver_assignments
            # compute loads
            _, driver_loads = assign_drivers_to_routes(routes, drivers, dist, mu_matrix)

        # penalty for drivers exceeding their shift_hours
        for d in drivers:
            did = d.get('id')
            shift = d.get('shift_hours', 8)
            load = driver_loads.get(did, 0.0)
            # if load (hours) exceeds shift, add heavy penalty proportional to excess hours
            if load > shift:
                total += penalty_unserved * (load - shift)

        # balance across drivers: variance of loads (hours) added scaled by weight_balance
        if weight_balance > 0 and driver_loads:
            vals = list(driver_loads.values())
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            total += weight_balance * var
    return total


# --- 5. ALNS operators (destroy & repair) ---
class EVRPSolution:
    """Concrete State implementing objective() for ALNS."""

    def __init__(self, solution, dist, demands):
        # solution is a dict containing 'routes', 'unassigned', 'vehicles', 'depot'
        self.solution = solution
        self.dist = dist
        self.demands = demands
        self.vehicles = solution.get('vehicles', [])
        self.depot = solution.get('depot', 0)
        # optional advanced fields (set by run_alns)
        self.mu_matrix = None
        self.weight_time = 1.0
        self.weight_balance = 0.0

    def objective(self) -> float:
        return evaluate(
            self.solution,
            self.dist,
            self.demands,
            weight_time=self.weight_time,
            weight_balance=self.weight_balance,
            mu_matrix=self.mu_matrix,
        )


def random_removal(state, rng, n_remove=2, **kwargs):
    sol = state.solution
    depot = sol.get('depot', 0)
    all_customers = [c for r in sol['routes'] for c in r if c != depot]
    if not all_customers:
        return state
    n_remove = min(n_remove, len(all_customers))
    removed = rng.choice(all_customers, size=n_remove, replace=False).tolist()
    new_sol = copy.deepcopy(sol)
    new_unassigned = set(new_sol.get('unassigned', []))
    for cust in removed:
        for r in new_sol['routes']:
            while cust in r:
                r.remove(cust)
        new_unassigned.add(cust)
    new_sol['unassigned'] = list(new_unassigned)
    # recompute driver assignments if drivers present
    if new_sol.get('drivers'):
        assigns, loads = assign_drivers_to_routes(new_sol['routes'], new_sol['drivers'], state.dist, state.mu_matrix if hasattr(state, 'mu_matrix') else None)
        new_sol['driver_assignments'] = assigns

    new_state = EVRPSolution(new_sol, state.dist, state.demands)
    # propagate mu and objective weights from current state
    try:
        new_state.mu_matrix = state.mu_matrix
        new_state.weight_time = state.weight_time
        new_state.weight_balance = state.weight_balance
    except Exception:
        pass
    return new_state


def greedy_insertion(state, rng, **kwargs):
    sol = state.solution
    dist = state.dist
    demands = state.demands
    depot = sol.get('depot', 0)
    unassigned = list(sol.get('unassigned', []))
    if not unassigned:
        return state
    new_sol = copy.deepcopy(sol)
    vehicles = new_sol.get('vehicles', [])
    for cust in unassigned:
        best = None
        best_cost = float('inf')
        # try inserting into existing routes while respecting capacity
        for i, r in enumerate(new_sol['routes']):
            # compute current load
            curr_load = sum(demands.get(c, 0) for c in r if c != depot)
            cap = vehicles[i]['capacity'] if i < len(vehicles) else float('inf')
            if curr_load + demands.get(cust, 0) > cap:
                continue
            for pos in range(1, len(r)):
                r_copy = r[:pos] + [cust] + r[pos:]
                cost = route_distance(r_copy, dist) - route_distance(r, dist)
                if cost < best_cost:
                    best_cost = cost
                    best = (i, pos)
        if best is not None:
            i, pos = best
            new_sol['routes'][i].insert(pos, cust)
        else:
            # open a new route; vehicles can perform multiple trips so we don't limit by vehicle count
            new_sol['routes'].append([depot, cust, depot])
    new_sol['unassigned'] = []
    # recompute driver assignments if drivers info present
    if new_sol.get('drivers'):
        assigns, loads = assign_drivers_to_routes(new_sol['routes'], new_sol['drivers'], dist, state.mu_matrix if hasattr(state, 'mu_matrix') else None)
        new_sol['driver_assignments'] = assigns

    new_state = EVRPSolution(new_sol, dist, demands)
    try:
        new_state.mu_matrix = state.mu_matrix
        new_state.weight_time = state.weight_time
        new_state.weight_balance = state.weight_balance
    except Exception:
        pass
    return new_state


def swap_between_routes(state, rng, **kwargs):
    sol = state.solution
    depot = sol.get('depot', 0)
    new_sol = copy.deepcopy(sol)
    routes = [r for r in new_sol['routes'] if len(r) > 2]
    if len(routes) < 2:
        return state
    # pick two distinct routes by index in the full routes list
    idxs = [i for i, r in enumerate(new_sol['routes']) if len(r) > 2]
    r1_idx, r2_idx = rng.choice(idxs, size=2, replace=False).tolist()
    route1 = new_sol['routes'][r1_idx]
    route2 = new_sol['routes'][r2_idx]
    c1_candidates = [c for c in route1 if c != depot]
    c2_candidates = [c for c in route2 if c != depot]
    if not c1_candidates or not c2_candidates:
        return EVRPSolution(new_sol, state.dist, state.demands)
    c1 = rng.choice(c1_candidates)
    c2 = rng.choice(c2_candidates)
    # swap occurrences
    for r in new_sol['routes']:
        for idx, v in enumerate(r):
            if v == c1:
                r[idx] = c2
            elif v == c2:
                r[idx] = c1
    # recompute driver assignments if drivers present
    if new_sol.get('drivers'):
        assigns, loads = assign_drivers_to_routes(new_sol['routes'], new_sol['drivers'], state.dist, state.mu_matrix if hasattr(state, 'mu_matrix') else None)
        new_sol['driver_assignments'] = assigns

    new_state = EVRPSolution(new_sol, state.dist, state.demands)
    try:
        new_state.mu_matrix = state.mu_matrix
        new_state.weight_time = state.weight_time
        new_state.weight_balance = state.weight_balance
    except Exception:
        pass
    return new_state


# --- 6. Run ALNS ---
def run_alns(nodes, links, requests, fleet, drivers=None, iterations=200, weight_time=1.0, weight_balance=0.0):
    dist, mu = build_distance_matrix(nodes, links)
    sol0 = initial_solution(nodes, requests, fleet, dist, drivers=drivers)
    depot = next((n['id'] for n in nodes if n['type'] == 'depot'), 0)
    sol0['depot'] = depot

    demands = {r['node_id']: r['load'] for r in requests}

    initial_cost = evaluate(sol0, dist, demands)
    print(f"Initial solution cost: {initial_cost:.2f}")

    initial_state = EVRPSolution(sol0, dist, demands)
    # attach mu matrix and objective weights to the state so operators can propagate them
    initial_state.mu_matrix = mu
    initial_state.weight_time = weight_time
    initial_state.weight_balance = weight_balance

    alns = ALNS()
    # register operators (destroy then repair)
    alns.add_destroy_operator(random_removal, name='rand_rem_2')
    # register another destroy with larger removal via a small wrapper
    def rand_rem_4(state, rng, **kwargs):
        return random_removal(state, rng, n_remove=4)

    alns.add_destroy_operator(rand_rem_4, name='rand_rem_4')
    alns.add_repair_operator(greedy_insertion, name='greedy_insert')
    alns.add_repair_operator(swap_between_routes, name='swap')

    # Create SA acceptor. autofit can produce start_temp < end_temp for small initial_cost;
    # instantiate a safe SA with start temp scaled from initial cost instead.
    try:
        start_temp = max(1.0, initial_cost * 10.0)
        acceptor = SimulatedAnnealing(start_temp, 1e-3, 0.995, method='exponential')
    except Exception:
        # fallback to a conservative default
        acceptor = SimulatedAnnealing(10.0, 1e-3, 0.99, method='exponential')
    # selection scheme: scores for [best, better, accept, reject]
    scores = [100, 50, 10, 0]
    decay = 0.8
    num_destroy = len(alns.destroy_operators)
    num_repair = len(alns.repair_operators)
    selector = RouletteWheel(scores, decay, num_destroy, num_repair)
    stop = MaxIterations(iterations)

    result = alns.iterate(initial_state, op_select=selector, accept=acceptor, stop=stop)

    best_state = result.best_state
    best_cost = best_state.objective()
    print(f"ALNS best cost: {best_cost:.2f}")

    # Post-process: ensure all requests are assigned. Compute missing requested nodes by comparing
    # the set of requested node_ids to the nodes present in the solution routes (excluding depot).
    solution = best_state.solution
    depot = solution.get('depot', 0)
    all_requested = set(r['node_id'] for r in requests)
    assigned_nodes = set()
    for r in solution.get('routes', []):
        for n in r:
            if n != depot:
                assigned_nodes.add(int(n))

    missing = sorted(list(all_requested - assigned_nodes))
    if missing:
        print(f"Post-processing: {len(missing)} missing requested nodes will be assigned as single-customer routes...")
        for cust in missing:
            solution.setdefault('routes', []).append([depot, int(cust), depot])
    # ensure unassigned is empty after postprocessing
    solution['unassigned'] = []

    return solution, best_cost


def pretty_print_solution(solution, dist):
    print("\nSolution routes:")
    for i, r in enumerate(solution['routes']):
        d = route_distance(r, dist)
        print(f" Route {i+1}: {r}  distance={d:.2f}")


def save_solution(solution, nodes, filename='output/solution.json'):
    import json
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # map node id -> lat/lon
    coords = {int(n['id']): (float(n['lat']), float(n['lon'])) for n in nodes}
    # normalize routes to native Python ints (convert numpy types if present)
    normalized_routes = []
    for r in solution.get('routes', []):
        normalized_routes.append([int(x) for x in r])

    out = {
        'routes': normalized_routes,
        'coords': coords,
    }
    # include unassigned if present
    if 'unassigned' in solution:
        out['unassigned'] = [int(x) for x in solution.get('unassigned', [])]
    # include driver assignments if present
    if 'driver_assignments' in solution:
        out['driver_assignments'] = {int(k): int(v) if v is not None else None for k, v in solution['driver_assignments'].items()}

    # include vehicle summary if present
    if 'vehicles' in solution:
        out['vehicles'] = solution['vehicles']
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"Saved solution to {filename}")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description='Run ALNS EVRP solver')
    parser.add_argument('--xml', default='evrp_rabat_data.xml', help='Input EVRP XML file')
    parser.add_argument('--iterations', type=int, default=400, help='ALNS iterations')
    parser.add_argument('--weight-time', type=float, default=1.0, help='Weight for expected travel time objective')
    parser.add_argument('--weight-balance', type=float, default=0.6, help='Weight for balance (variance) objective')
    args = parser.parse_args(argv)

    xml = args.xml
    nodes, links, requests, fleet, drivers = parse_evrp_instance(xml)
    solution, cost = run_alns(nodes, links, requests, fleet, drivers=drivers, iterations=args.iterations, weight_time=args.weight_time, weight_balance=args.weight_balance)
    dist, _ = build_distance_matrix(nodes, links)
    pretty_print_solution(solution, dist)
    save_solution(solution, nodes, filename='output/solution.json')


if __name__ == '__main__':
    main()
