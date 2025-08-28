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


# --- 1. Parse EVRP instances ---
def _parse_xml(xml_path):
    """Parse legacy XML-based EVRP instance files."""
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


def _parse_evrp(path):
    """Parse .evrp files (TSPLIB-like format)."""
    header = {}
    node_coords = {}
    demands = {}
    stations = set()
    depot_ids = []

    with open(path, 'r') as f:
        lines = [ln.rstrip() for ln in f]

    i = 0
    n = len(lines)
    # read header until NODE_COORD_SECTION
    while i < n:
        ln = lines[i]
        if ln.startswith('NODE_COORD_SECTION'):
            i += 1
            break
        if ':' in ln:
            k, v = ln.split(':', 1)
            header[k.strip()] = v.strip()
        i += 1

    # NODE_COORD_SECTION
    while i < n:
        ln = lines[i].strip()
        if ln == '' or ln.startswith('DEMAND_SECTION') or ln.endswith('_SECTION'):
            break
        parts = ln.split()
        if len(parts) >= 3:
            idx = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            node_coords[idx] = (x, y)
        i += 1

    # move to DEMAND_SECTION
    while i < n and not lines[i].startswith('DEMAND_SECTION'):
        i += 1
    if i < n and lines[i].startswith('DEMAND_SECTION'):
        i += 1
    # DEMAND_SECTION
    while i < n:
        ln = lines[i].strip()
        if ln == '' or ln.startswith('STATIONS_COORD_SECTION') or ln.endswith('_SECTION') or ln.startswith('DEPOT_SECTION'):
            break
        parts = ln.split()
        if len(parts) >= 2:
            idx = int(parts[0])
            d = int(parts[1])
            demands[idx] = d
        i += 1

    # STATIONS_COORD_SECTION (may contain just ids)
    while i < n and not lines[i].startswith('STATIONS_COORD_SECTION'):
        i += 1
    if i < n and lines[i].startswith('STATIONS_COORD_SECTION'):
        i += 1
        while i < n:
            ln = lines[i].strip()
            if ln == '' or ln.startswith('DEPOT_SECTION') or ln.endswith('_SECTION'):
                break
            parts = ln.split()
            if len(parts) >= 1:
                try:
                    stations.add(int(parts[0]))
                except ValueError:
                    pass
            i += 1

    # DEPOT_SECTION
    while i < n and not lines[i].startswith('DEPOT_SECTION'):
        i += 1
    if i < n and lines[i].startswith('DEPOT_SECTION'):
        i += 1
        while i < n:
            ln = lines[i].strip()
            if ln == '' or ln == '-1' or ln == 'EOF':
                if ln == '-1':
                    break
                i += 1
                continue
            try:
                depot_ids.append(int(ln.split()[0]))
            except Exception:
                pass
            i += 1

    dim = int(header.get('DIMENSION', len(node_coords)))
    vehicles = int(header.get('VEHICLES', 1))
    capacity = float(header.get('CAPACITY', 0))
    energy_capacity = float(header.get('ENERGY_CAPACITY', 0))
    energy_consumption = float(header.get('ENERGY_CONSUMPTION', 1.0))

    file_ids = sorted(node_coords.keys())
    id_to_idx = {fid: idx for idx, fid in enumerate(file_ids)}

    nodes = []
    for fid in file_ids:
        idx = id_to_idx[fid]
        x, y = node_coords[fid]
        ntype = 'customer'
        if fid in depot_ids:
            ntype = 'depot'
        elif fid in stations:
            ntype = 'charging_station'
        nodes.append({'id': idx, 'type': ntype, 'lat': float(x), 'lon': float(y)})

    links = {}
    for i, fid_i in enumerate(file_ids):
        xi, yi = node_coords[fid_i]
        for j, fid_j in enumerate(file_ids):
            if i == j:
                continue
            xj, yj = node_coords[fid_j]
            dist = math.hypot(xi - xj, yi - yj)
            travel_time = dist / 40.0
            links[(i, j)] = {'distance': float(dist), 'travel_time': float(travel_time)}

    requests = []
    for fid, d in demands.items():
        if fid in id_to_idx:
            idx = id_to_idx[fid]
            if d > 0 and nodes[idx]['type'] != 'depot':
                requests.append({'node_id': idx, 'load': int(d)})

    fleet = [{
        'type': 'ev_standard',
        'max_load_capacity': capacity,
        'max_energy_capacity': energy_capacity,
        'consumption_per_km': energy_consumption,
        'quantity': vehicles,
    }]

    drivers = []
    return nodes, links, requests, fleet, drivers


def parse_instance(path):
    """Parse an EVRP instance from either XML or .evrp format."""
    if path.lower().endswith('.evrp'):
        return _parse_evrp(path)
    return _parse_xml(path)


# --- 2. Build distance matrix ---
def build_mats(nodes, links, sigma_alpha=0.2):
    """Build distance, expected travel time (mu) and sigma matrices.

    sigma_alpha: fallback relative stddev: sigma = alpha * mu when explicit sigma missing.
    """
    n = len(nodes)
    big = 1e6
    dist = np.full((n, n), big)
    mu = np.full((n, n), big)  # expected travel times (in hours)
    sigma = np.zeros((n, n))
    for i in range(n):
        dist[i, i] = 0.0
        mu[i, i] = 0.0
        sigma[i, i] = 0.0
    for (i, j), data in links.items():
        if 0 <= i < n and 0 <= j < n:
            d = data['distance'] if isinstance(data, dict) else data
            dist[i, j] = d
            # travel_time may be present (in hours) or not
            tt = None
            if isinstance(data, dict):
                tt = data.get('travel_time')
            if tt is None:
                mu[i, j] = dist[i, j] / 40.0
            else:
                if tt > 24:  # likely seconds -> convert to hours
                    mu[i, j] = tt / 3600.0
                else:
                    mu[i, j] = tt
            # optional sigma provided in links dict
            if isinstance(data, dict) and data.get('sigma') is not None:
                sigma[i, j] = float(data.get('sigma'))
            else:
                sigma[i, j] = max(1e-6, sigma_alpha * mu[i, j])
    # symmetrize when appropriate
    for i in range(n):
        for j in range(n):
            if dist[i, j] < big and dist[j, i] >= big:
                dist[j, i] = dist[i, j]
            if mu[i, j] < big and mu[j, i] >= big:
                mu[j, i] = mu[i, j]
                sigma[j, i] = sigma[i, j]
    return dist, mu, sigma


# --- 3. Simple initial solution (greedy capacity-respecting) ---
def init_solution(nodes, requests, fleet, dist, drivers=None):
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
        """Check if at least one vehicle type can perform the route considering load and battery
        and allowing full recharge at charging stations (nodes marked 'charging_station')."""
        load = sum(customer_demands.get(c, 0) for c in route if c != depot_id)
        # compute distance between successive nodes for energy consumption
        for v in vehicles:
            if load > v['capacity']:
                continue
            # simulate battery along route (full start)
            battery = v.get('energy', 0.0)
            consumption = v.get('consumption', v.get('consumption_per_km', 0.2))
            feasible = True
            for i in range(len(route) - 1):
                a = route[i]
                b = route[i + 1]
                # energy consumed proportional to distance
                energy_needed = dist[a, b] * consumption
                battery -= energy_needed
                # if next is charging station, recharge to full
                # nodes list is available in outer scope; map id to type
                try:
                    ntype = nodes[a]['type'] if 0 <= a < len(nodes) else None
                except Exception:
                    ntype = None
                # recharge if arriving at station
                if 0 <= b < len(nodes) and nodes[b]['type'] == 'charging_station':
                    battery = v.get('energy', 0.0)
                if battery < -1e-6:
                    feasible = False
                    break
            if feasible:
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
def route_dist(route, dist):
    d = 0.0
    for i in range(len(route) - 1):
        a = route[i]
        b = route[i + 1]
        d += dist[a, b]
    return d


def assign_drivers(routes, drivers, dist, mu_matrix=None):
    """Assign drivers to routes trying to balance total route durations (LPT heuristic).

    Returns a dict mapping route_index -> driver_id and a dict driver_id -> total_time_hours
    """
    if not drivers:
        return {}, {}
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

def score(solution, dist, customer_demands, weight_time=1.0, weight_balance=0.0, mu_matrix=None, sigma_matrix=None, monte_carlo=False, mc_samples=50, penalty_unserved=1e5, vehicles=None, nodes=None):
    total = 0.0
    routes = solution.get('routes', [])
    unassigned = set(solution.get('unassigned', []))
    depot = solution.get('depot', 0)
    drivers = solution.get('drivers', None)
    driver_assignments = solution.get('driver_assignments', None)

    # Objective 1: expected total travel time (using mu if available), optionally via Monte Carlo
    def route_expected_time(route):
        # compute expected time deterministically (sum mu) or via MC
        if not monte_carlo:
            t = 0.0
            for i in range(len(route) - 1):
                a = route[i]
                b = route[i + 1]
                if mu_matrix is not None:
                    t += mu_matrix[a, b]
                else:
                    t += dist[a, b] / 40.0
            return t
        else:
            # Monte Carlo samples using mu and sigma matrices
            samples = []
            for _ in range(mc_samples):
                s = 0.0
                for i in range(len(route) - 1):
                    a = route[i]
                    b = route[i + 1]
                    mu_ = mu_matrix[a, b] if mu_matrix is not None else dist[a, b] / 40.0
                    sigma_ = sigma_matrix[a, b] if sigma_matrix is not None else max(1e-6, 0.2 * mu_)
                    val = random.gauss(mu_, sigma_)
                    # ensure non-negative
                    s += max(0.0, val)
                samples.append(s)
            return sum(samples) / len(samples)

    time_cost = 0.0
    for r in routes:
        time_cost += route_expected_time(r)

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
    vehicles = solution.get('vehicles', []) if vehicles is None else vehicles
    # helper: check if any vehicle can serve this route (capacity + energy)
    def route_feasible_by_fleet(route):
        load = sum(customer_demands.get(c, 0) for c in route if c != depot)
        # energy feasibility considering recharging at stations
        for v in vehicles:
            cap = v.get('capacity', v.get('max_load_capacity', float('inf')))
            if load > cap:
                continue
            energy_cap = v.get('energy', v.get('max_energy_capacity', float('inf')))
            consumption = v.get('consumption', v.get('consumption_per_km', 0.2))
            battery = energy_cap
            feasible = True
            for i in range(len(route) - 1):
                a = route[i]
                b = route[i + 1]
                energy_needed = dist[a, b] * consumption
                battery -= energy_needed
                if 0 <= b < len(nodes) and nodes[b]['type'] == 'charging_station':
                    battery = energy_cap
                if battery < -1e-6:
                    feasible = False
                    break
            if feasible:
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
            assignments, driver_loads = assign_drivers(routes, drivers, dist, mu_matrix)
        else:
            assignments = driver_assignments
            # compute loads
            _, driver_loads = assign_drivers(routes, drivers, dist, mu_matrix)

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
class Solution:
    """Concrete State implementing objective() for ALNS."""

    def __init__(self, solution, dist, demands):
        # solution is a dict containing 'routes', 'unassigned', 'vehicles', 'depot'
        self.solution = solution
        self.dist = dist
        self.demands = demands
        self.vehicles = solution.get('vehicles', [])
        self.depot = solution.get('depot', 0)
        # optional advanced fields (set by solve)
        self.mu_matrix = None
        self.weight_time = 1.0
        self.weight_balance = 0.0

    def objective(self) -> float:
        return score(
            self.solution,
            self.dist,
            self.demands,
            weight_time=self.weight_time,
            weight_balance=self.weight_balance,
            mu_matrix=getattr(self, 'mu_matrix', None),
            sigma_matrix=getattr(self, 'sigma_matrix', None),
            monte_carlo=getattr(self, 'monte_carlo', False),
            mc_samples=getattr(self, 'mc_samples', 50),
            vehicles=self.solution.get('vehicles', None),
            nodes=getattr(self, 'nodes_ref', None),
        )


def remove_random(state, rng, n_remove=2, **kwargs):
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
        assigns, loads = assign_drivers(new_sol['routes'], new_sol['drivers'], state.dist, state.mu_matrix if hasattr(state, 'mu_matrix') else None)
        new_sol['driver_assignments'] = assigns

    new_state = Solution(new_sol, state.dist, state.demands)
    # propagate mu and objective weights from current state
    try:
        new_state.mu_matrix = state.mu_matrix
        new_state.sigma_matrix = getattr(state, 'sigma_matrix', None)
        new_state.weight_time = state.weight_time
        new_state.weight_balance = state.weight_balance
        new_state.monte_carlo = getattr(state, 'monte_carlo', False)
        new_state.mc_samples = getattr(state, 'mc_samples', 50)
        new_state.nodes_ref = getattr(state, 'nodes_ref', None)
    except Exception:
        pass
    return new_state


def insert_greedy(state, rng, **kwargs):
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
                cost = route_dist(r_copy, dist) - route_dist(r, dist)
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
        assigns, loads = assign_drivers(new_sol['routes'], new_sol['drivers'], dist, state.mu_matrix if hasattr(state, 'mu_matrix') else None)
        new_sol['driver_assignments'] = assigns

    new_state = Solution(new_sol, dist, demands)
    try:
        new_state.mu_matrix = state.mu_matrix
        new_state.sigma_matrix = getattr(state, 'sigma_matrix', None)
        new_state.weight_time = state.weight_time
        new_state.weight_balance = state.weight_balance
        new_state.monte_carlo = getattr(state, 'monte_carlo', False)
        new_state.mc_samples = getattr(state, 'mc_samples', 50)
        new_state.nodes_ref = getattr(state, 'nodes_ref', None)
    except Exception:
        pass
    return new_state


def swap_routes(state, rng, **kwargs):
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
        return Solution(new_sol, state.dist, state.demands)
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
        assigns, loads = assign_drivers(new_sol['routes'], new_sol['drivers'], state.dist, state.mu_matrix if hasattr(state, 'mu_matrix') else None)
        new_sol['driver_assignments'] = assigns

    new_state = Solution(new_sol, state.dist, state.demands)
    try:
        new_state.mu_matrix = state.mu_matrix
        new_state.sigma_matrix = getattr(state, 'sigma_matrix', None)
        new_state.weight_time = state.weight_time
        new_state.weight_balance = state.weight_balance
        new_state.monte_carlo = getattr(state, 'monte_carlo', False)
        new_state.mc_samples = getattr(state, 'mc_samples', 50)
        new_state.nodes_ref = getattr(state, 'nodes_ref', None)
    except Exception:
        pass
    return new_state


# --- 6. Run ALNS ---
def solve(nodes, links, requests, fleet, drivers=None, iterations=200, weight_time=1.0, weight_balance=0.0, monte_carlo=False, mc_samples=50, sigma_alpha=0.2):
    dist, mu, sigma = build_mats(nodes, links, sigma_alpha=sigma_alpha)
    sol0 = init_solution(nodes, requests, fleet, dist, drivers=drivers)
    depot = next((n['id'] for n in nodes if n['type'] == 'depot'), 0)
    sol0['depot'] = depot

    demands = {r['node_id']: r['load'] for r in requests}

    initial_cost = score(sol0, dist, demands, mu_matrix=mu, sigma_matrix=sigma, monte_carlo=monte_carlo, mc_samples=mc_samples, vehicles=sol0.get('vehicles', []), nodes=nodes)
    print(f"Initial solution cost: {initial_cost:.2f}")

    initial_state = Solution(sol0, dist, demands)
    # attach mu/sigma matrices and objective weights to the state so operators can propagate them
    initial_state.mu_matrix = mu
    initial_state.sigma_matrix = sigma
    initial_state.weight_time = weight_time
    initial_state.weight_balance = weight_balance
    initial_state.monte_carlo = monte_carlo
    initial_state.mc_samples = mc_samples
    initial_state.nodes_ref = nodes

    alns = ALNS()
    # register operators (destroy then repair)
    alns.add_destroy_operator(remove_random, name='rand_rem_2')
    # register another destroy with larger removal via a small wrapper
    def rand_rem_4(state, rng, **kwargs):
        return remove_random(state, rng, n_remove=4)

    alns.add_destroy_operator(rand_rem_4, name='rand_rem_4')
    alns.add_repair_operator(insert_greedy, name='greedy_insert')
    alns.add_repair_operator(swap_routes, name='swap')

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


def print_solution(solution, dist, prefix=""):
    """Pretty-print solution routes with optional solver prefix."""
    print(f"\n{prefix}Solution routes:")
    for i, r in enumerate(solution['routes']):
        d = route_dist(r, dist)
        print(f"{prefix} Route {i+1}: {r}  distance={d:.2f}")


def save(solution, nodes, filename='output/solution.json', prefix=""):
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
    print(f"{prefix} Saved solution to {filename}")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description='Run EVRP solver')
    parser.add_argument('--instance', default='evrp_rabat_data.xml',
                        help='Input EVRP instance file (.xml or .evrp)')
    parser.add_argument('--iterations', type=int, default=400,
                        help='ALNS/GA iterations')
    parser.add_argument('--weight-time', type=float, default=1.0,
                        help='Weight for expected travel time objective')
    parser.add_argument('--weight-balance', type=float, default=0.6,
                        help='Weight for balance (variance) objective')
    parser.add_argument('--solver', default='alns',
                        help='Comma-separated list of solving backends to run '
                             "(alns, ortools, ga) or 'all'")
    args = parser.parse_args(argv)

    inst = args.instance
    nodes, links, requests, fleet, drivers = parse_instance(inst)
    # parse solver list
    chosen = [s.strip().lower() for s in args.solver.split(',') if s.strip()]
    if 'all' in chosen:
        chosen = ['alns', 'ortools', 'ga']

    dist, _, _ = build_mats(nodes, links)
    for sname in chosen:
        if sname == 'ortools':
            from ortools_solver import solve_ortools
            solution, cost = solve_ortools(nodes, links, requests, fleet)
        elif sname == 'ga':
            from ga_solver import solve_ga
            solution, cost = solve_ga(nodes, links, requests, fleet,
                                     generations=args.iterations)
        else:
            solution, cost = solve(nodes, links, requests, fleet, drivers=drivers,
                                   iterations=args.iterations,
                                   weight_time=args.weight_time,
                                   weight_balance=args.weight_balance)
        prefix = f"[{sname.upper()}]"
        print(prefix, f"Total cost: {cost:.2f}")
        print_solution(solution, dist, prefix=prefix)
        save(solution, nodes, filename=f'output/solution_{sname}.json', prefix=prefix)


if __name__ == '__main__':
    main()
