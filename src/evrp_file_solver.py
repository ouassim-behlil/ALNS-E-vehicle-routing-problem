import math
import sys
from pathlib import Path

from evrp_alns_solver import run_alns, save_solution


def parse_evrp_file(path):
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

    return header, node_coords, demands, stations, depot_ids


def build_problem_from_evrp(path):
    header, node_coords, demands, stations, depot_ids = parse_evrp_file(path)

    dim = int(header.get('DIMENSION', len(node_coords)))
    vehicles = int(header.get('VEHICLES', 1))
    capacity = float(header.get('CAPACITY', 0))
    energy_capacity = float(header.get('ENERGY_CAPACITY', 0))
    energy_consumption = float(header.get('ENERGY_CONSUMPTION', 1.0))

    # sort node ids to create stable indexing
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

    # build links using Euclidean distances, and travel_time assuming 40 units/hour
    links = {}
    for i, fid_i in enumerate(file_ids):
        xi, yi = node_coords[fid_i]
        for j, fid_j in enumerate(file_ids):
            if i == j:
                continue
            xj, yj = node_coords[fid_j]
            dist = math.hypot(xi - xj, yi - yj)
            # travel_time in hours (assume avg speed 40 distance units/hour)
            travel_time = dist / 40.0
            links[(i, j)] = {'distance': float(dist), 'travel_time': float(travel_time)}

    # requests: include only demands strictly greater than zero and exclude depot nodes
    requests = []
    for fid, d in demands.items():
        if fid in id_to_idx:
            idx = id_to_idx[fid]
            if d > 0 and nodes[idx]['type'] != 'depot':
                requests.append({'node_id': idx, 'load': int(d)})

    # fleet: single vehicle type using headers
    fleet = [{
        'type': 'ev_standard',
        'max_load_capacity': capacity,
        'max_energy_capacity': energy_capacity,
        'consumption_per_km': energy_consumption,
        'quantity': vehicles,
    }]

    return nodes, links, requests, fleet


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description='Solve .evrp instance using ALNS-based solver')
    parser.add_argument('instance', help='Path to .evrp instance file')
    parser.add_argument('--iterations', type=int, default=1000, help='ALNS iterations')
    parser.add_argument('--output', default='output/solution_from_evrp.json', help='Output JSON')
    args = parser.parse_args(argv)

    inst = Path(args.instance)
    if not inst.exists():
        print(f'Instance not found: {inst}')
        sys.exit(1)

    nodes, links, requests, fleet = build_problem_from_evrp(str(inst))

    print(f'Parsed instance: nodes={len(nodes)} requests={len(requests)} fleet_types={len(fleet)}')

    # run ALNS
    solution, cost = run_alns(nodes, links, requests, fleet, drivers=None, iterations=args.iterations)
    print(f'Finished. cost={cost:.2f}')

    # save solution
    save_solution(solution, nodes, filename=args.output)


if __name__ == '__main__':
    main()
