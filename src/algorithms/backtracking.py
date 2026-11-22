import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

_ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

try:
    from src.algorithms.cvrp_parser import CVRPParser, CVRPInstance
except ImportError:
    try:
        from .cvrp_parser import CVRPParser, CVRPInstance
    except ImportError:
        from cvrp_parser import CVRPParser, CVRPInstance


def solve_cvrp(
    instance: CVRPInstance,
    time_limit: Optional[float] = None,
    max_customers: Optional[int] = None
) -> Dict[str, Any]:
    node_ids = sorted(instance.coords.keys() if instance.coords else range(1, instance.dimension + 1))
    depot_id = instance.depot_id
    depot_idx = node_ids.index(depot_id)

    all_customers = [i for i, nid in enumerate(node_ids) if nid != depot_id]
    if max_customers is not None and len(all_customers) > max_customers:
        customers = all_customers[:max_customers]
    else:
        customers = all_customers

    start_time = time.time()
    if not customers:
        return {
            "algorithm": "backtracking",
            "instance_name": instance.name,
            "total_distance": 0,
            "routes": [],
            "num_vehicles": 0,
            "execution_time": 0.0
        }

    num_nodes = len(node_ids)
    demands = [instance.demands[nid] for nid in node_ids]
    capacity = instance.capacity
    dist = instance.distance_matrix

    min_out = [min(dist[v][w] for w in range(num_nodes) if w != v) for v in range(num_nodes)]
    min_depot_out = min_out[depot_idx]
    is_symmetric = all(dist[i][j] == dist[j][i] for i in range(num_nodes) for j in range(i + 1, num_nodes))
    sorted_neighbors = [
        sorted(customers, key=lambda v, u=u: dist[u][v])
        for u in range(num_nodes)
    ]

    def greedy_init():
        unvisited = set(customers)
        routes = []
        total_cost = 0
        while unvisited:
            curr_route = [depot_id]
            curr_u = depot_idx
            curr_load = 0
            while True:
                cands = [v for v in unvisited if curr_load + demands[v] <= capacity]
                if not cands:
                    break
                v = min(cands, key=lambda x: dist[curr_u][x])
                unvisited.remove(v)
                curr_route.append(node_ids[v])
                total_cost += dist[curr_u][v]
                curr_load += demands[v]
                curr_u = v
            total_cost += dist[curr_u][depot_idx]
            curr_route.append(depot_id)
            routes.append(curr_route)
        return total_cost, routes

    best_cost, best_routes = greedy_init()
    visited = [False] * num_nodes
    total_customers = len(customers)
    init_rem_min_out = sum(min_out[v] for v in customers)
    init_rem_demand = sum(demands[v] for v in customers)

    curr_route = [depot_id]
    routes: List[List[int]] = []
    timed_out = False

    def backtrack(
        u: int,
        load: int,
        count: int,
        cost: int,
        must_visit: int,
        rem_min_out: int,
        rem_demand: int
    ):
        nonlocal best_cost, best_routes, timed_out
        if timed_out:
            return
        if time_limit is not None and time.time() - start_time > time_limit:
            timed_out = True
            return

        if count == total_customers:
            if is_symmetric and len(curr_route) >= 3 and curr_route[1] > curr_route[-1]:
                return
            total = cost + dist[u][depot_idx]
            if total < best_cost:
                best_cost = total
                best_routes = [r[:] for r in routes] + [curr_route + [depot_id]]
            return

        rem_cap = (capacity - load) if u != depot_idx else 0
        excess_demand = rem_demand - rem_cap
        k_new = (excess_demand + capacity - 1) // capacity if excess_demand > 0 else 0

        rem_lb = rem_min_out + (min_out[u] if u != depot_idx else 0) + k_new * min_depot_out
        if cost + rem_lb >= best_cost:
            return

        if u == depot_idx:
            for v in customers:
                if not visited[v]:
                    must_visit = v
                    break

        for v in sorted_neighbors[u]:
            if not visited[v] and load + demands[v] <= capacity:
                d_uv = dist[u][v]
                if cost + d_uv >= best_cost:
                    continue
                visited[v] = True
                curr_route.append(node_ids[v])
                backtrack(
                    v,
                    load + demands[v],
                    count + 1,
                    cost + d_uv,
                    must_visit,
                    rem_min_out - min_out[v],
                    rem_demand - demands[v]
                )
                curr_route.pop()
                visited[v] = False

        if u != depot_idx and visited[must_visit]:
            if not (is_symmetric and len(curr_route) >= 3 and curr_route[1] > curr_route[-1]):
                d_ret = dist[u][depot_idx]
                if cost + d_ret + rem_min_out + ((rem_demand + capacity - 1) // capacity) * min_depot_out < best_cost:
                    closed = curr_route[:] + [depot_id]
                    routes.append(closed)
                    saved_route = curr_route[:]
                    curr_route.clear()
                    curr_route.append(depot_id)
                    backtrack(depot_idx, 0, count, cost + d_ret, -1, rem_min_out, rem_demand)
                    curr_route.clear()
                    curr_route.extend(saved_route)
                    routes.pop()

    backtrack(depot_idx, 0, 0, 0, -1, init_rem_min_out, init_rem_demand)
    elapsed_time = round(time.time() - start_time, 4)

    return {
        "algorithm": "backtracking",
        "instance_name": instance.name,
        "total_distance": best_cost,
        "routes": best_routes,
        "num_vehicles": len(best_routes),
        "execution_time": elapsed_time
    }


def solve_from_file(
    file_path: str,
    time_limit: Optional[float] = None,
    max_customers: Optional[int] = None
) -> Dict[str, Any]:
    instance = CVRPParser.parse_file(file_path)
    return solve_cvrp(instance, time_limit, max_customers)


if __name__ == "__main__":
    result = solve_from_file("data/CVRPLIB/E-n13-k4.vrp")
    print("Cost:", result["total_distance"], "Vehicles:", result["num_vehicles"], "Time:", result["execution_time"])
