import math
import random
import time
from typing import List, Dict, Any, Optional
try:
    from src.cvrp_parser import CVRPParser, CVRPInstance
except ImportError:
    from cvrp_parser import CVRPParser, CVRPInstance


def two_opt_route(route: List[int], dist: List[List[int]], id_to_idx: Dict[int, int]) -> List[int]:
    if len(route) <= 3:
        return route
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                d_old = dist[id_to_idx[best[i - 1]]][id_to_idx[best[i]]] + dist[id_to_idx[best[j]]][id_to_idx[best[j + 1]]]
                d_new = dist[id_to_idx[best[i - 1]]][id_to_idx[best[j]]] + dist[id_to_idx[best[i]]][id_to_idx[best[j + 1]]]
                if d_new < d_old:
                    best[i:j + 1] = reversed(best[i:j + 1])
                    improved = True
                    break
            if improved:
                break
    return best


def solve_cvrp(
    instance: CVRPInstance,
    t_init: float = 1000.0,
    t_min: float = 0.05,
    cooling_rate: float = 0.985,
    iters_per_temp: int = 40,
    seed: Optional[int] = None,
    return_history: bool = False
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    node_ids = sorted(instance.coords.keys() if instance.coords else range(1, instance.dimension + 1))
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

    def get_nn_tour():
        unvisited = set(customers)
        curr = depot_id
        tour = []
        while unvisited:
            nxt = min(unvisited, key=lambda c: dist[id_to_idx[curr]][id_to_idx[c]])
            tour.append(nxt)
            unvisited.remove(nxt)
            curr = nxt
        return tour

    start_time = time.time()
    n = len(customers)
    if n == 0:
        return {
            "algorithm": "simulated_annealing",
            "instance_name": instance.name,
            "total_distance": 0,
            "routes": [],
            "num_vehicles": 0,
            "execution_time": 0.0
        }

    nn_tour = get_nn_tour()
    nn_routes, nn_cost = split_routes(nn_tour)

    rand_tour = customers[:]
    random.shuffle(rand_tour)
    rand_routes, rand_cost = split_routes(rand_tour)

    if nn_cost <= rand_cost:
        curr_tour = nn_tour[:]
        curr_cost = nn_cost
    else:
        curr_tour = rand_tour[:]
        curr_cost = rand_cost

    best_tour = curr_tour[:]
    best_cost = curr_cost
    history = [best_cost]

    t = t_init
    while t > t_min and n > 1:
        for _ in range(iters_per_temp):
            new_tour = curr_tour[:]
            move_type = random.randint(1, 3)
            i, j = sorted(random.sample(range(n), 2))

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

        t *= cooling_rate
        if return_history:
            history.append(best_cost)

    raw_routes, _ = split_routes(best_tour)
    best_routes = [two_opt_route(r, dist, id_to_idx) for r in raw_routes]
    best_cost = sum(dist[id_to_idx[r[i]]][id_to_idx[r[i + 1]]] for r in best_routes for i in range(len(r) - 1))
    if return_history:
        history.append(best_cost)

    elapsed_time = round(time.time() - start_time, 4)

    res = {
        "algorithm": "simulated_annealing",
        "instance_name": instance.name,
        "total_distance": best_cost,
        "routes": best_routes,
        "num_vehicles": len(best_routes),
        "execution_time": elapsed_time
    }
    if return_history:
        res["history"] = history
    return res


def solve_from_file(
    file_path: str,
    t_init: float = 1000.0,
    t_min: float = 0.05,
    cooling_rate: float = 0.985,
    iters_per_temp: int = 40,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    instance = CVRPParser.parse_file(file_path)
    return solve_cvrp(instance, t_init, t_min, cooling_rate, iters_per_temp, seed)


if __name__ == "__main__":
    result = solve_from_file("CVRPLIB/A-n32-k5.vrp", seed=42)
    print("Cost:", result["total_distance"])
    print("Time:", result["execution_time"])
    print("Vehicles:", result["num_vehicles"])
    print("Routes:", result["routes"])
