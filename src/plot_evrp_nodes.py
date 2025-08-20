import folium
import xml.etree.ElementTree as ET
import os
import traceback

try:
    import osmnx as ox
    import networkx as nx
except Exception:
    ox = None
    nx = None

from pathlib import Path

# Path to the generated XML file - try common names and fall back to searching
script_dir = Path(__file__).parent
candidates = [script_dir.parent / 'evrp_rabat_data.xml', script_dir.parent / 'evrp_rabat_data_test.xml']
data_file_path = None
for c in candidates:
    if c.exists():
        data_file_path = c
        break

if data_file_path is None:
    # try to find any evrp_*.xml in project root
    xmls = list(script_dir.parent.glob('evrp_*.xml'))
    if xmls:
        data_file_path = xmls[0]

if data_file_path is None:
    print('No EVRP XML data file found. Looked for:', [str(p) for p in candidates])
    raise FileNotFoundError('No evrp XML file found in project root. Run dataset generator first.')

# Parse XML to extract node information
data_file = str(data_file_path)
tree = ET.parse(data_file)
root = tree.getroot()

# Get nodes
def get_nodes():
    nodes = []
    for node_elem in root.find('nodes'):
        node = {
            'id': int(node_elem.find('id').text),
            'lat': float(node_elem.find('latitude').text),
            'lon': float(node_elem.find('longitude').text),
            'type': node_elem.find('node_type').text,
            'charging_technology': node_elem.find('charging_technology').text
        }
        # optional osmid (used for realistic routing)
        osmid_elem = node_elem.find('osmid')
        if osmid_elem is not None and osmid_elem.text is not None:
            try:
                node['osmid'] = int(osmid_elem.text)
            except Exception:
                node['osmid'] = None
        nodes.append(node)
    return nodes

nodes = get_nodes()

# Center map on Rabat
center_lat = sum(n['lat'] for n in nodes) / len(nodes)
center_lon = sum(n['lon'] for n in nodes) / len(nodes)

m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Custom icons for each node type
icon_dict = {
    'depot': 'home',
    'customer': 'shopping-cart',
    'charging_station': 'bolt'
}
color_dict = {
    'depot': 'blue',
    'customer': 'green',
    'charging_station': 'orange'
}

for node in nodes:
    folium.Marker(
        location=[node['lat'], node['lon']],
        popup=f"ID: {node['id']}<br>Type: {node['type']}<br>Charging: {node['charging_technology']}",
        icon=folium.Icon(color=color_dict.get(node['type'], 'red'), icon=icon_dict.get(node['type'], 'info-sign'), prefix='fa')
    ).add_to(m)

# Save and display map
map_file = os.path.join(os.path.dirname(__file__), 'evrp_nodes_map.html')
m.save(map_file)

print(f"Map saved to {map_file}. Opening in browser...")

import webbrowser
webbrowser.open('file://' + os.path.abspath(map_file))

# If a solution file exists, draw routes on the map
try:
    import json
    sol_file = os.path.join(os.path.dirname(__file__), '../output/solution.json')
    if os.path.exists(sol_file):
        with open(sol_file, 'r', encoding='utf-8') as f:
            sol = json.load(f)
        routes = sol.get('routes', [])
        coords = {int(k): tuple(v) for k, v in sol.get('coords', {}).items()}
        # build mapping id -> osmid from XML nodes
        id_to_osmid = {n['id']: n.get('osmid') for n in nodes}
        colors = ['red', 'blue', 'purple', 'darkgreen', 'cadetblue', 'black']
        # If osmnx is available and osmids exist, compute realistic routes
        use_osm = ox is not None and any(v is not None for v in id_to_osmid.values())
        G = None
        if use_osm:
            try:
                lats = [n['lat'] for n in nodes]
                lons = [n['lon'] for n in nodes]
                north = max(lats) + 0.02
                south = min(lats) - 0.02
                east = max(lons) + 0.02
                west = min(lons) - 0.02
                print('Building road network graph for plotting (this may take a moment)...')
                G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
                G = ox.utils_graph.get_undirected(G)
            except Exception:
                print('Failed to build OSMnx bbox graph, trying local GraphML fallback...')
                # try local graphml fallback
                graphml_path = os.path.join(os.path.dirname(__file__), '..', 'rabt_road_network.graphml')
                try:
                    graphml_path = os.path.abspath(graphml_path)
                    if os.path.exists(graphml_path):
                        print('Loading local graphml:', graphml_path)
                        try:
                            # prefer osmnx loader if available
                            G = ox.load_graphml(graphml_path)
                        except Exception:
                            G = nx.read_graphml(graphml_path)
                            # convert string node ids to int when possible
                            try:
                                G = nx.relabel_nodes(G, lambda x: int(x), copy=True)
                            except Exception:
                                pass
                        # ensure undirected and has length attribute
                        try:
                            G = ox.utils_graph.get_undirected(G)
                        except Exception:
                            pass
                    else:
                        print('Local GraphML not found at', graphml_path)
                        G = None
                except Exception:
                    print('Failed to load local GraphML, falling back to straight polylines')
                    G = None

        for i, r in enumerate(routes):
            try:
                poly = []
                if G is not None:
                    # compute route by concatenating shortest paths between consecutive nodes using osmids
                    for a, b in zip(r[:-1], r[1:]):
                        osa = id_to_osmid.get(a)
                        osb = id_to_osmid.get(b)
                        if osa is None or osb is None:
                            # fallback to straight segment
                            if a in coords and b in coords:
                                poly.append(coords[a])
                                poly.append(coords[b])
                            continue
                        try:
                            path = ox.shortest_path(G, osa, osb, weight='length')
                            # append node coordinates for the path
                            for node_id in path:
                                data = G.nodes[node_id]
                                poly.append((data.get('y'), data.get('x')))
                        except Exception:
                            # fallback to straight segment
                            if a in coords and b in coords:
                                poly.append(coords[a])
                                poly.append(coords[b])
                else:
                    # simple straight polyline
                    for nid in r:
                        if nid in coords:
                            poly.append(coords[nid])

                # remove consecutive duplicates
                if poly:
                    cleaned = [poly[0]]
                    for p in poly[1:]:
                        if p != cleaned[-1]:
                            cleaned.append(p)
                else:
                    cleaned = []

                if len(cleaned) >= 2:
                    folium.PolyLine(locations=cleaned, color=colors[i % len(colors)], weight=4, opacity=0.8).add_to(m)
            except Exception as e:
                print('Error plotting route', i, e)
                traceback.print_exc()
        # save again with routes
        routed_map = os.path.join(os.path.dirname(__file__), 'evrp_nodes_map_with_routes.html')
        m.save(routed_map)
        print(f"Map with routes saved to {routed_map}. Opening in browser...")
        webbrowser.open('file://' + os.path.abspath(routed_map))
except Exception as e:
    print(f"No solution plotted: {e}")
