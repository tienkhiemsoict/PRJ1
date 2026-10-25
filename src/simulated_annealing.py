import math
import random
import time
from typing import List, Dict, Any, Optional
try:
    from src.cvrp_parser import CVRPParser, CVRPInstance
except ImportError:
    from cvrp_parser import CVRPParser, CVRPInstance


def solve_cvrp(
    instance: CVRPInstance,
    t_init: float = 1000.0,
    t_min: float = 0.01,
    cooling_rate: float = 0.995,
    iters_per_temp: int = 100,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    node_ids = sorted(instance.coords.keys())
    depot_id = instance.depot_id
    customers = [nid for nid in node_ids if nid != depot_id]

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    demands = instance.demands
    capacity = instance.capacity
    dist = instance.distance_matrix

    def split_routes(tour: List[int]):
        routes: List[List[int]] = []
        curr_route = [depot_id]
        curr_load = 0
        total_cost = 0

        for c in tour:
            d = demands[c]
            if curr_load + d <= capacity:
                curr_route.append(c)
                curr_load += d
            else:
                curr_route.append(depot_id)
                routes.append(curr_route)
                curr_route = [depot_id, c]
                curr_load = d

        if len(curr_route) > 1:
            curr_route.append(depot_id)
            routes.append(curr_route)

        for r in routes:
            for i in range(len(r) - 1):
                total_cost += dist[id_to_idx[r[i]]][id_to_idx[r[i + 1]]]

        return routes, total_cost

    start_time = time.time()

    curr_tour = customers[:]
    random.shuffle(curr_tour)
    curr_routes, curr_cost = split_routes(curr_tour)

    best_tour = curr_tour[:]
    best_cost = curr_cost
    best_routes = curr_routes

    t = t_init
    n = len(curr_tour)

    while t > t_min and n > 1:
        for _ in range(iters_per_temp):
            new_tour = curr_tour[:]
            move_type = random.randint(1, 3)
            i, j = random.sample(range(n), 2)
            if i > j:
                i, j = j, i

            if move_type == 1:
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            elif move_type == 2:
                new_tour[i:j + 1] = reversed(new_tour[i:j + 1])
            else:
                val = new_tour.pop(j)
                new_tour.insert(i, val)

            _, new_cost = split_routes(new_tour)
            delta = new_cost - curr_cost

            if delta < 0 or random.random() < math.exp(-delta / t):
                curr_tour = new_tour
                curr_cost = new_cost
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_tour = curr_tour[:]
                    best_routes, _ = split_routes(best_tour)

        t *= cooling_rate

    elapsed_time = round(time.time() - start_time, 4)

    return {
        "algorithm": "simulated_annealing",
        "instance_name": instance.name,
        "total_distance": best_cost,
        "routes": best_routes,
        "num_vehicles": len(best_routes),
        "execution_time": elapsed_time
    }


def solve_from_file(
    file_path: str,
    t_init: float = 1000.0,
    t_min: float = 0.01,
    cooling_rate: float = 0.995,
    iters_per_temp: int = 100,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    instance = CVRPParser.parse_file(file_path)
    return solve_cvrp(instance, t_init, t_min, cooling_rate, iters_per_temp, seed)


if __name__ == "__main__":
    result = solve_from_file("CVRPLIB/A-n32-k5.vrp", t_init=1000.0, cooling_rate=0.99, iters_per_temp=80, seed=42)
    print("Cost:", result["total_distance"])
    print("Time:", result["execution_time"])
    print("Vehicles:", result["num_vehicles"])
    print("Routes:", result["routes"])

