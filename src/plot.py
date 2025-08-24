import argparse
import os
import matplotlib
# use non-interactive backend when no DISPLAY
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# A clean style with larger fonts for report-ready figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    }
)


def plot_sol(sol_path, out_path=None, show=False, figsize=(10, 8)):
    """Plot a solution JSON file and save a high-quality image.

    Parameters
    ----------
    sol_path : str
        Path to the solution JSON file.
    out_path : str, optional
        File path for the generated image. If ``None`` an image with
        ``_plot.png`` suffix is created next to ``sol_path``.
    show : bool, optional
        Display the figure interactively.
    figsize : tuple, optional
        Size of the figure in inches.

    Returns
    -------
    str
        The path to the saved image.
    """

    with open(sol_path, "r") as f:
        sol = json.load(f)

    routes = sol.get('routes', [])
    coords = sol.get('coords', {})

    # convert coords keys to ints and arrays
    coord_map = {int(k): np.array(v, dtype=float) for k, v in coords.items()}

    # determine depot: assume node 0 exists
    depot_id = 0
    if depot_id not in coord_map and routes and routes[0]:
        depot_id = routes[0][0]

    # prepare plot
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap('tab10')

    # plot each route
    for i, route in enumerate(routes):
        if not route:
            continue
        pts = [coord_map[int(n)] for n in route if int(n) in coord_map]
        if not pts:
            continue
        pts = np.vstack(pts)
        color = cmap(i % 10)
        ax.plot(pts[:, 0], pts[:, 1], '-', color=color, linewidth=2, alpha=0.8, label=f'Route {i+1}')
        ax.scatter(pts[1:-1, 0] if len(pts) > 2 else [], pts[1:-1, 1] if len(pts) > 2 else [], c=[color], s=30)
        # mark route start and end
        ax.scatter(pts[0, 0], pts[0, 1], c='k', marker='s', s=60, zorder=5)
        ax.scatter(pts[-1, 0], pts[-1, 1], c='k', marker='o', s=60, zorder=5)

    # plot depot
    if depot_id in coord_map:
        d = coord_map[depot_id]
        ax.scatter(d[0], d[1], c='red', marker='*', s=150, zorder=10, label='Depot')

    ax.set_title(f'Solution plot: {os.path.basename(sol_path)}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc="best")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    if out_path is None:
        out_path = os.path.splitext(sol_path)[0] + '_plot.png'
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved plot to {out_path}")
    if show:
        plt.show()
    plt.close(fig)

    return out_path


def parse_file(path):
    """Lightweight parser for .evrp files used in this repo.

    Returns: file_ids list, coord_map dict {id: (x,y)}, set(station_ids), depot_ids list
    """
    file_ids = []
    coord_map = {}
    stations = set()
    depot_ids = []
    with open(path, 'r') as f:
        lines = [ln.rstrip() for ln in f]
    i = 0
    n = len(lines)
    # find NODE_COORD_SECTION
    while i < n and not lines[i].startswith('NODE_COORD_SECTION'):
        i += 1
    if i < n and lines[i].startswith('NODE_COORD_SECTION'):
        i += 1
        while i < n:
            ln = lines[i].strip()
            if ln == '' or ln.startswith('DEMAND_SECTION') or ln.endswith('_SECTION'):
                break
            parts = ln.split()
            if len(parts) >= 3:
                fid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                file_ids.append(fid)
                coord_map[fid] = (x, y)
            i += 1
    # stations
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
                except Exception:
                    pass
            i += 1
    # depot section
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
    return file_ids, coord_map, stations, depot_ids


def plot_inst(evrp_path, out_path=None, show=False, figsize=(10, 8)):
    """Plot the nodes of an EVRP instance and save as an image.

    Parameters
    ----------
    evrp_path : str
        Path to the ``.evrp`` instance file.
    out_path : str, optional
        File path for the generated image. If ``None`` an image with
        ``_nodes.png`` suffix is created next to ``evrp_path``.
    show : bool, optional
        Display the figure interactively.
    figsize : tuple, optional
        Size of the figure in inches.

    Returns
    -------
    str
        The path to the saved image.
    """

    file_ids, coord_map, stations, depot_ids = parse_file(evrp_path)
    if not file_ids:
        raise ValueError(f"No nodes found in {evrp_path}")
    pts = np.array([coord_map[fid] for fid in file_ids])

    fig, ax = plt.subplots(figsize=figsize)
    # plot all nodes
    ax.scatter(pts[:, 0], pts[:, 1], c="gray", s=30, label="Node")
    # annotate nodes with their original file ids
    for fid in file_ids:
        x, y = coord_map[fid]
        ax.annotate(str(fid), (x, y), textcoords="offset points", xytext=(3, 3), fontsize=8)
    # plot stations
    for sid in stations:
        if sid in coord_map:
            x, y = coord_map[sid]
            station_label = None
            if "Charging station" not in ax.get_legend_handles_labels()[1]:
                station_label = "Charging station"
            ax.scatter(x, y, c="blue", marker="P", s=120, label=station_label)
    # depot
    if depot_ids:
        did = depot_ids[0]
        if did in coord_map:
            x, y = coord_map[did]
            ax.scatter(x, y, c="red", marker="*", s=200, label="Depot")

    ax.set_title(f"Instance nodes: {os.path.basename(evrp_path)}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="best")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    if out_path is None:
        out_path = os.path.splitext(evrp_path)[0] + "_nodes.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved instance plot to {out_path}")
    if show:
        plt.show()
    plt.close(fig)

    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot EVRP instance and/or solution')
    parser.add_argument('--instance', help='Path to .evrp instance file', default=None)
    parser.add_argument('--solution', help='Path to solution JSON', default=None)
    parser.add_argument('--out', help='Output PNG path for solution (or instance if only instance provided)', default=None)
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    args = parser.parse_args()
    # If instance provided, first plot nodes
    if args.instance:
        inst_out = None
        if args.out and args.solution is None:
            inst_out = args.out
        plot_inst(args.instance, out_path=inst_out, show=args.show)
    # Then plot solution if provided
    if args.solution:
        plot_sol(args.solution, out_path=args.out, show=args.show)
    if not args.instance and not args.solution:
        parser.print_help()
