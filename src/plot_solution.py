import json
import argparse
import os
import matplotlib
# use non-interactive backend when no DISPLAY
if os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_solution(sol_path, out_path=None, show=False, figsize=(10, 8)):
    with open(sol_path, 'r') as f:
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
    ax.legend(loc='best', fontsize='small')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    if out_path is None:
        out_path = os.path.splitext(sol_path)[0] + '_plot.png'
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved plot to {out_path}')
    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot EVRP solution JSON')
    parser.add_argument('solution', help='Path to solution JSON')
    parser.add_argument('--out', help='Output PNG path', default=None)
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    args = parser.parse_args()
    plot_solution(args.solution, out_path=args.out, show=args.show)
