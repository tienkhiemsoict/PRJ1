import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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


def solve_cvrp(instance: CVRPInstance, max_customers: Optional[int] = None) -> Dict[str, Any]:
    node_ids = sorted(instance.coords.keys() if instance.coords else range(1, instance.dimension + 1))
    depot_id = instance.depot_id
    depot_idx = node_ids.index(depot_id)

    all_customers = [i for i, nid in enumerate(node_ids) if nid != depot_id]
    if max_customers is not None and len(all_customers) > max_customers:
        cust_indices = all_customers[:max_customers]
    else:
        cust_indices = all_customers

    k = len(cust_indices)
    start_time = time.time()
    if k == 0:
        return {
            "algorithm": "dynamic_programming",
            "instance_name": instance.name,
            "total_distance": 0,
            "routes": [],
            "num_vehicles": 0,
            "execution_time": 0.0
        }

    demands = [instance.demands[nid] for nid in node_ids]
    capacity = instance.capacity
    dist = instance.distance_matrix

    memo: Dict[Tuple[int, int, int], Tuple[float, List[int]]] = {}
    full_mask = (1 << k) - 1

    def dp(mask: int, last_u: int, rem_cap: int) -> Tuple[float, List[int]]:
        if mask == full_mask:
            return dist[last_u][depot_idx], [depot_id]

        state = (mask, last_u, rem_cap)
        if state in memo:
            return memo[state]

        best_cost = float("inf")
        best_path: List[int] = []

        for i in range(k):
            if not (mask & (1 << i)):
                cust_idx = cust_indices[i]
                d = demands[cust_idx]
                if d <= rem_cap:
                    sub_cost, sub_path = dp(mask | (1 << i), cust_idx, rem_cap - d)
                    total_c = dist[last_u][cust_idx] + sub_cost
                    if total_c < best_cost:
                        best_cost = total_c
                        best_path = [node_ids[cust_idx]] + sub_path

        if last_u != depot_idx:
            sub_cost, sub_path = dp(mask, depot_idx, capacity)
            total_c = dist[last_u][depot_idx] + sub_cost
            if total_c < best_cost:
                best_cost = total_c
                best_path = [depot_id] + sub_path

        memo[state] = (best_cost, best_path)
        return memo[state]

    best_cost, raw_path = dp(0, depot_idx, capacity)
    elapsed_time = round(time.time() - start_time, 4)

    full_path = [depot_id] + raw_path
    routes: List[List[int]] = []
    curr_route: List[int] = []
    for node in full_path:
        curr_route.append(node)
        if node == depot_id and len(curr_route) > 1:
            routes.append(curr_route)
            curr_route = [depot_id]

    if len(curr_route) > 1:
        curr_route.append(depot_id)
        routes.append(curr_route)

    return {
        "algorithm": "dynamic_programming",
        "instance_name": instance.name,
        "total_distance": best_cost,
        "routes": routes,
        "num_vehicles": len(routes),
        "execution_time": elapsed_time
    }


def solve_from_file(file_path: str, max_customers: Optional[int] = None) -> Dict[str, Any]:
    instance = CVRPParser.parse_file(file_path)
    return solve_cvrp(instance, max_customers)


if __name__ == "__main__":
    result = solve_from_file("data/CVRPLIB/E-n13-k4.vrp")
    print("Cost:", result["total_distance"], "Vehicles:", result["num_vehicles"], "Time:", result["execution_time"])
