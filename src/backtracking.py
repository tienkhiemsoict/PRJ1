import time
from typing import List, Dict, Any, Optional
try:
    from src.cvrp_parser import CVRPParser, CVRPInstance
except ImportError:
    from cvrp_parser import CVRPParser, CVRPInstance


def solve_cvrp(instance: CVRPInstance, time_limit: float = 10.0, max_customers: Optional[int] = 10) -> Dict[str, Any]:
    node_ids = sorted(instance.coords.keys())
    depot_id = instance.depot_id
    depot_idx = node_ids.index(depot_id)

    all_customers = [i for i, nid in enumerate(node_ids) if nid != depot_id]
    if max_customers is not None and len(all_customers) > max_customers:
        customers = all_customers[:max_customers]
    else:
        customers = all_customers

    demands = [instance.demands[nid] for nid in node_ids]
    capacity = instance.capacity
    dist = instance.distance_matrix

    min_in = {v: min(dist[u][v] for u in range(len(node_ids)) if u != v) for v in customers}

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
    visited = {v: False for v in customers}
    start_time = time.time()

    def backtrack(u, load, count, cost, curr_route, routes):
        nonlocal best_cost, best_routes
        if time.time() - start_time > time_limit:
            return

        rem_lb = sum(min_in[v] for v in customers if not visited[v])
        if cost + rem_lb + dist[u][depot_idx] >= best_cost:
            return

        if count == len(customers):
            total = cost + dist[u][depot_idx]
            if total < best_cost:
                best_cost = total
                best_routes = [r[:] for r in routes] + [curr_route + [depot_id]]
            return

        cands = [v for v in customers if not visited[v] and load + demands[v] <= capacity]
        cands.sort(key=lambda v: dist[u][v])
        for v in cands:
            visited[v] = True
            backtrack(v, load + demands[v], count + 1, cost + dist[u][v], curr_route + [node_ids[v]], routes)
            visited[v] = False

        if u != depot_idx:
            backtrack(depot_idx, 0, count, cost + dist[u][depot_idx], [depot_id], routes + [curr_route + [depot_id]])

    backtrack(depot_idx, 0, 0, 0, [depot_id], [])
    elapsed_time = round(time.time() - start_time, 4)

    return {
        "algorithm": "backtracking",
        "instance_name": instance.name,
        "total_distance": best_cost,
        "routes": best_routes,
        "num_vehicles": len(best_routes),
        "execution_time": elapsed_time
    }


def solve_from_file(file_path: str, time_limit: float = 10.0, max_customers: Optional[int] = 10) -> Dict[str, Any]:
    instance = CVRPParser.parse_file(file_path)
    return solve_cvrp(instance, time_limit, max_customers)


if __name__ == "__main__":
    result = solve_from_file("CVRPLIB/P-n16-k8.vrp", time_limit=5.0, max_customers=10)
    print("Cost:", result["total_distance"])
    print("Time:", result["execution_time"])
    print("Routes:", result["routes"])

