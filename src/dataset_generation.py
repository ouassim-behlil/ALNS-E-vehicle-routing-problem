
import osmnx as ox
import networkx as nx
import numpy as np
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
import math

class EVRPDataGenerator:
    def __init__(
        self,
        place_name="Rabat, Morocco",
        num_nodes=10,
        num_requests=5,
        total_vehicles: int | None = None,
        vehicle_spec: list | None = None,
        random_seed: int | None = None,
    ):
        """
        Initialize the EVRP data generator

        Args:
            place_name: Location to get real road network data
            num_nodes: Number of nodes to generate (including depot and charging stations)
            num_requests: Number of customer requests
            total_vehicles: total number of vehicles to create (if vehicle_spec is None)
            vehicle_spec: optional explicit list of vehicle type dicts. If provided, this will be
                used directly instead of the default internal template.
            random_seed: optional seed for reproducibility
        """
        self.place_name = place_name
        self.num_nodes = num_nodes
        self.num_requests = num_requests
        self.total_vehicles = total_vehicles
        self.vehicle_spec = vehicle_spec
        self.random_seed = random_seed
        if random_seed is not None:
            import random as _rnd

            _rnd.seed(random_seed)
            np.random.seed(random_seed)

        # Node types
        self.node_types = ["depot", "customer", "charging_station"]

        # Charging technologies
        self.charging_technologies = ["Type1", "Type2", "CCS", "CHAdeMO", "Tesla"]

        # Configure OSMnx
        ox.settings.use_cache = True
        ox.settings.log_console = True

        # Store the graph for distance calculations
        self.G = None
        self.node_mapping = {}  # Maps our node IDs to OSM node IDs
        
    def get_real_network(self):
        """Download and process real road network data"""
        print(f"Downloading road network for {self.place_name}...")
        
        # Download street network with more detailed parameters
        self.G = ox.graph_from_place(
            self.place_name, 
            network_type='drive',
            simplify=True,
            retain_all=False,
            truncate_by_edge=True
        )
        
        # Convert to undirected for simplicity and add edge speeds
        self.G = ox.convert.to_undirected(self.G)
        
        # Add edge speeds, travel times, and lengths
        self.G = ox.add_edge_speeds(self.G)
        self.G = ox.add_edge_travel_times(self.G)
        
        # Get node coordinates
        nodes_data = []
        for node_id, data in self.G.nodes(data=True):
            nodes_data.append({
                'osmid': node_id,
                'lat': data['y'],
                'lon': data['x']
            })
        
        print(f"Downloaded network with {len(nodes_data)} nodes")
        return self.G, nodes_data
    
    def calculate_real_distance_and_time(self, from_osmid, to_osmid):
        """Calculate real distance and travel time using OSMnx shortest path"""
        try:
            # Get shortest path
            path = ox.shortest_path(self.G, from_osmid, to_osmid, weight='length')
            
            if path is None:
                # If no path found, return Haversine distance as fallback
                from_data = self.G.nodes[from_osmid]
                to_data = self.G.nodes[to_osmid]
                distance = self.calculate_haversine_distance(
                    from_data['y'], from_data['x'],
                    to_data['y'], to_data['x']
                )
                travel_time = distance / 50.0  # Assume 50 km/h
                return distance, travel_time
            
            # Calculate total distance and time
            total_distance = 0
            total_time = 0
            
            for i in range(len(path) - 1):
                edge_data = self.G.edges[path[i], path[i + 1], 0]
                total_distance += edge_data.get('length', 0)  # meters
                total_time += edge_data.get('travel_time', 0)  # seconds
            
            # Convert to km and hours
            distance_km = total_distance / 1000.0
            time_hours = total_time / 3600.0
            
            return distance_km, time_hours
            
        except Exception as e:
            print(f"Error calculating path between {from_osmid} and {to_osmid}: {e}")
            # Fallback to Haversine distance
            from_data = self.G.nodes[from_osmid]
            to_data = self.G.nodes[to_osmid]
            distance = self.calculate_haversine_distance(
                from_data['y'], from_data['x'],
                to_data['y'], to_data['x']
            )
            travel_time = distance / 50.0
            return distance, travel_time
    
    def calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points (fallback method)"""
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def generate_nodes(self, nodes_data):
        """Generate nodes with random selection from real coordinates"""
        print("Generating nodes...")
        
        # Randomly select nodes from real data
        # pick diverse nodes across the downloaded list to avoid clustering
        k = min(self.num_nodes, len(nodes_data))
        # stratified sampling: keep first as depot candidate, then sample rest
        selected_nodes = []
        if k > 0:
            # choose a node near the geographic centroid as depot when possible
            lats = [nd['lat'] for nd in nodes_data]
            lons = [nd['lon'] for nd in nodes_data]
            centroid = (np.mean(lats), np.mean(lons))
            # find closest node to centroid
            def dist_to_cent(n):
                return (n['lat'] - centroid[0]) ** 2 + (n['lon'] - centroid[1]) ** 2

            nodes_sorted = sorted(nodes_data, key=dist_to_cent)
            selected_nodes.append(nodes_sorted[0])
            remaining = [n for n in nodes_data if n is not nodes_sorted[0]]
            if k - 1 > 0:
                # sample remaining to maximize spatial spread
                sampled = np.random.choice(len(remaining), size=k - 1, replace=False)
                for idx in sampled:
                    selected_nodes.append(remaining[int(idx)])

        nodes = []
        # ensure some charging stations (at least 15% of nodes)
        min_charging = max(1, int(0.15 * self.num_nodes))
        for i, node_data in enumerate(selected_nodes):
            # Store mapping from our node ID to OSM node ID
            self.node_mapping[i] = node_data.get('osmid')

            # Assign node types: depot, customers, charging stations
            if i == 0:
                node_type = "depot"
            else:
                # ensure first portion are customers up to requested number
                if len([n for n in nodes if n.get('node_type') == 'customer']) < self.num_requests:
                    node_type = 'customer'
                else:
                    # add charging stations until minimum reached, otherwise randomly
                    if len([n for n in nodes if n.get('node_type') == 'charging_station']) < min_charging:
                        node_type = 'charging_station'
                    else:
                        node_type = random.choices(['customer', 'charging_station'], weights=[0.7, 0.3])[0]

            # Assign charging technology
            if node_type == "depot":
                charging_tech = random.choice(["Type2", "CCS"])  # Depot has good charging
            elif node_type == "charging_station":
                charging_tech = random.choice(self.charging_technologies)
            else:
                charging_tech = "None"

            nodes.append({
                'id': i,
                'latitude': node_data['lat'],
                'longitude': node_data['lon'],
                'node_type': node_type,
                'charging_technology': charging_tech,
                'osmid': node_data.get('osmid')
            })
        
        return nodes
    
    def generate_links(self, nodes):
        """Generate links between nodes with real distances and travel times using OSMnx"""
        print("Generating links with real distances from OSMnx...")

        links = []
        total_pairs = len(nodes) * (len(nodes) - 1)
        print(f"Calculating distances for {total_pairs} node pairs...")

        pairs_processed = 0
        for i, node_from in enumerate(nodes):
            for j, node_to in enumerate(nodes):
                if i != j:
                    # Get real distance and travel time using OSMnx
                    distance, travel_time = self.calculate_real_distance_and_time(
                        node_from['osmid'], node_to['osmid']
                    )

                    # All links are symmetric in this case
                    symmetric = True

                    links.append({
                        'from': i,
                        'to': j,
                        'distance': round(distance, 2),
                        'travel_time': round(travel_time, 3),
                        'symmetric': symmetric
                    })

                    pairs_processed += 1
                    if pairs_processed % 50 == 0:
                        print(f"Processed {pairs_processed}/{total_pairs} pairs...")

        return links
    
    def generate_requests(self, nodes):
        """Generate customer requests with realistic demand patterns"""
        print("Generating requests...")
        requests = []
        customer_nodes = [node for node in nodes if node['node_type'] == 'customer']

        # Prioritize customers closer to the depot for frequent deliveries
        depot = next((n for n in nodes if n['node_type'] == 'depot'), None)
        def hav(lat1, lon1, lat2, lon2):
            R = 6371
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        if depot is not None:
            customer_nodes = sorted(customer_nodes, key=lambda c: hav(depot['latitude'], depot['longitude'], c['latitude'], c['longitude']))

        # create requests for all customer nodes (realistic requirement)
        for i, customer in enumerate(customer_nodes):
            # demand depends on customer type and proximity (urban smaller)
            base = random.randint(1, 12)
            # skew towards smaller sizes for closer customers
            if depot is not None and hav(depot['latitude'], depot['longitude'], customer['latitude'], customer['longitude']) < 5:
                requested_load = max(1, int(base * random.uniform(0.5, 1.0)))
            else:
                requested_load = base

            # Add realistic time windows (morning/afternoon peaks)
            if random.random() < 0.6:
                start_time = random.randint(8, 12)
            else:
                start_time = random.randint(12, 16)
            end_time = start_time + random.randint(1, 4)

            # service time in minutes
            service_time = random.randint(10, 30)

            # priority (1 highest, 3 lowest)
            priority = random.choices([1, 2, 3], weights=[0.2, 0.6, 0.2])[0]

            requests.append({
                'node_id': customer['id'],
                'requested_load': requested_load,
                'time_window_start': start_time,
                'time_window_end': min(end_time, 18),
                'service_time_minutes': service_time,
                'priority': priority
            })

        return requests
    
    def generate_fleet(self):
        """Generate fleet specifications optimized for Rabat's urban environment"""
        print("Generating fleet...")
        # If explicit vehicle_spec was provided, use it
        if self.vehicle_spec is not None:
            # validate and return a deep copy
            return [dict(v) for v in self.vehicle_spec]

        # Default vehicle types template
        vehicle_types = [
            {
                'type': 'compact_ev',
                'max_load_capacity': 12,
                'max_energy_capacity': 35,  # kWh
                'charging_technology': 'Type2',
                'consumption_per_km': 0.16,  # kWh/km
                'quantity': 0,
            },
            {
                'type': 'urban_delivery_ev',
                'max_load_capacity': 20,
                'max_energy_capacity': 50,  # kWh
                'charging_technology': 'CCS',
                'consumption_per_km': 0.20,  # kWh/km
                'quantity': 0,
            },
            {
                'type': 'medium_cargo_ev',
                'max_load_capacity': 35,
                'max_energy_capacity': 70,  # kWh
                'charging_technology': 'CCS',
                'consumption_per_km': 0.25,  # kWh/km
                'quantity': 0,
            },
        ]

        # If total_vehicles specified, distribute across types by weights
        total = self.total_vehicles if self.total_vehicles is not None else 6
        # weights favor urban delivery and compact for city
        weights = np.array([0.4, 0.4, 0.2])
        counts = (weights * total).astype(int)
        # ensure sum equals total
        diff = total - counts.sum()
        for i in range(diff):
            counts[i % len(counts)] += 1

        for t, c in zip(vehicle_types, counts):
            t['quantity'] = int(c)

        return vehicle_types

    def generate_drivers(self, num_drivers: int | None = None):
        """Generate a simple list of drivers with basic attributes.

        Each driver gets an id and optionally a skill level or shift length in hours.
        """
        print("Generating drivers...")
        drivers = []
        total = num_drivers if num_drivers is not None else max(1, (self.total_vehicles or 6))
        for i in range(total):
            drivers.append({
                'id': i,
                'name': f"driver_{i}",
                'shift_hours': 8,
                'skill_level': random.choice([1, 2, 3])
            })
        return drivers
    
    def create_xml(self, nodes, links, requests, fleet, filename="evrp_rabat_data.xml"):
        """Create XML file with all EVRP data"""
        print(f"Creating XML file: {filename}")
        
        # Create root element
        root = ET.Element("evrp_data")
        
        # Add metadata
        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "location").text = self.place_name
        ET.SubElement(metadata, "generation_method").text = "OSMnx with real distances"
        ET.SubElement(metadata, "num_nodes").text = str(len(nodes))
        ET.SubElement(metadata, "num_links").text = str(len(links))
        ET.SubElement(metadata, "num_requests").text = str(len(requests))
        
        # Add nodes
        nodes_elem = ET.SubElement(root, "nodes")
        for node in nodes:
            node_elem = ET.SubElement(nodes_elem, "node")
            ET.SubElement(node_elem, "id").text = str(node['id'])
            ET.SubElement(node_elem, "latitude").text = str(node['latitude'])
            ET.SubElement(node_elem, "longitude").text = str(node['longitude'])
            ET.SubElement(node_elem, "node_type").text = node['node_type']
            ET.SubElement(node_elem, "charging_technology").text = node['charging_technology']
            ET.SubElement(node_elem, "osmid").text = str(node['osmid'])
        
        # Add links
        links_elem = ET.SubElement(root, "links")
        for link in links:
            link_elem = ET.SubElement(links_elem, "link")
            ET.SubElement(link_elem, "from").text = str(link['from'])
            ET.SubElement(link_elem, "to").text = str(link['to'])
            ET.SubElement(link_elem, "distance").text = str(link['distance'])
            ET.SubElement(link_elem, "travel_time").text = str(link['travel_time'])
            ET.SubElement(link_elem, "symmetric").text = str(link['symmetric'])
        
        # Add requests
        requests_elem = ET.SubElement(root, "requests")
        for request in requests:
            request_elem = ET.SubElement(requests_elem, "request")
            ET.SubElement(request_elem, "node_id").text = str(request['node_id'])
            ET.SubElement(request_elem, "requested_load").text = str(request['requested_load'])
            ET.SubElement(request_elem, "time_window_start").text = str(request['time_window_start'])
            ET.SubElement(request_elem, "time_window_end").text = str(request['time_window_end'])
            # optional fields
            if 'service_time_minutes' in request:
                ET.SubElement(request_elem, "service_time_minutes").text = str(request['service_time_minutes'])
            if 'priority' in request:
                ET.SubElement(request_elem, "priority").text = str(request['priority'])
        
        # Add fleet
        fleet_elem = ET.SubElement(root, "fleet")
        for vehicle in fleet:
            vehicle_elem = ET.SubElement(fleet_elem, "vehicle")
            ET.SubElement(vehicle_elem, "type").text = vehicle['type']
            ET.SubElement(vehicle_elem, "max_load_capacity").text = str(vehicle['max_load_capacity'])
            ET.SubElement(vehicle_elem, "max_energy_capacity").text = str(vehicle['max_energy_capacity'])
            ET.SubElement(vehicle_elem, "charging_technology").text = vehicle['charging_technology']
            ET.SubElement(vehicle_elem, "consumption_per_km").text = str(vehicle['consumption_per_km'])
            ET.SubElement(vehicle_elem, "quantity").text = str(vehicle['quantity'])
        
        # Add drivers (simple list)
        drivers = self.generate_drivers()
        drivers_elem = ET.SubElement(root, "drivers")
        for d in drivers:
            d_elem = ET.SubElement(drivers_elem, "driver")
            ET.SubElement(d_elem, "id").text = str(d['id'])
            ET.SubElement(d_elem, "name").text = d['name']
            ET.SubElement(d_elem, "shift_hours").text = str(d['shift_hours'])
            ET.SubElement(d_elem, "skill_level").text = str(d['skill_level'])
        
        # Pretty print XML
        rough_string = ET.tostring(root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        print(f"XML file created successfully: {filename}")
        return filename
    
    def generate_summary_report(self, nodes, links, requests, fleet):
        """Generate a summary report of the generated data"""
        print("\n" + "="*60)
        print("EVRP DATA GENERATION SUMMARY - RABAT, MOROCCO")
        print("="*60)
        
        # Node statistics
        node_types_count = {}
        for node in nodes:
            node_type = node['node_type']
            node_types_count[node_type] = node_types_count.get(node_type, 0) + 1
        
        print(f"Total Nodes: {len(nodes)}")
        for node_type, count in node_types_count.items():
            print(f"  - {node_type.replace('_', ' ').title()}: {count}")
        
        # Distance statistics
        distances = [link['distance'] for link in links]
        print(f"\nDistance Statistics:")
        print(f"  - Average distance: {np.mean(distances):.2f} km")
        print(f"  - Min distance: {np.min(distances):.2f} km")
        print(f"  - Max distance: {np.max(distances):.2f} km")
        
        # Request statistics
        loads = [req['requested_load'] for req in requests]
        print(f"\nRequest Statistics:")
        print(f"  - Total requests: {len(requests)}")
        print(f"  - Total load demand: {sum(loads)} units")
        print(f"  - Average load per request: {np.mean(loads):.1f} units")
        
        # Fleet statistics
        print(f"\nFleet Statistics:")
        total_vehicles = sum(vehicle['quantity'] for vehicle in fleet)
        total_capacity = sum(vehicle['max_load_capacity'] * vehicle['quantity'] for vehicle in fleet)
        print(f"  - Total vehicles: {total_vehicles}")
        print(f"  - Total fleet capacity: {total_capacity} units")
        print(f"  - Fleet types: {len(fleet)}")
        
        print("="*60)

def main(argv=None):
    """Main function to generate EVRP data for Rabat.

    Accepts command-line arguments to configure generation, including
    the total number of vehicles (--total-vehicles).
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate EVRP dataset (Rabat example)")
    parser.add_argument('--place', default="Rabat, Morocco", help='Place name used by OSMnx')
    parser.add_argument('--num-nodes', type=int, default=13, help='Total number of nodes (including depot and charging stations)')
    parser.add_argument('--num-requests', type=int, default=10, help='Number of customer requests')
    parser.add_argument('--total-vehicles', type=int, default=2, help='Total number of vehicles to generate (overrides default distribution)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='evrp_rabat_data.xml', help='Output XML filename')

    args = parser.parse_args(argv)

    # Initialize generator with CLI options
    generator = EVRPDataGenerator(
        place_name=args.place,
        num_nodes=args.num_nodes,
        num_requests=args.num_requests,
        total_vehicles=args.total_vehicles,
        random_seed=args.random_seed,
    )

    try:
        # Get real network data
        print(f"Starting EVRP data generation for {args.place}...")
        G, nodes_data = generator.get_real_network()

        # Generate EVRP components
        nodes = generator.generate_nodes(nodes_data)
        links = generator.generate_links(nodes)
        requests = generator.generate_requests(nodes)
        fleet = generator.generate_fleet()

        # Create XML file
        filename = generator.create_xml(nodes, links, requests, fleet, filename=args.output)

        # Generate summary report
        generator.generate_summary_report(nodes, links, requests, fleet)

        print(f"\nData saved to: {filename}")
        print("Generation complete! Real distances calculated using OSMnx.")

    except Exception as e:
        print(f"Error generating EVRP data: {e}")
        print("Please check your internet connection and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check required packages
    try:
        import osmnx
        import networkx
        import numpy
    except ImportError:
        print("Please install required packages:")
        print("pip install osmnx networkx numpy")
        exit(1)
    main()