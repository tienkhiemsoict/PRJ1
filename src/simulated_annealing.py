import math
import random
import time
from typing import List, Dict, Any, Optional, Callable
try:
    from src.cvrp_parser import CVRPParser, CVRPInstance
except ImportError:
    from cvrp_parser import CVRPParser, CVRPInstance


def solve_cvrp(
    instance: CVRPInstance,
    t_init: float = 200.0,
    t_min: float = 0.05,
    cooling_rate: float = 0.995,
    iters_per_temp: int = 350,
    seed: Optional[int] = None,
    return_history: bool = False,
    progress_callback: Optional[Callable[[int, float, Optional[List[List[int]]]], None]] = None
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    node_ids = sorted(instance.coords.keys() if instance.coords else range(1, instance.dimension + 1))
    depot_id = instance.depot_id
    customers = [nid for nid in node_ids if nid != depot_id]

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    demands = instance.demands
    capacity = instance.capacity
    num_trucks = instance.num_trucks
    dist = instance.distance_matrix

    def _calc_route_dist(route: List[int]) -> int:
        return sum(dist[id_to_idx[route[i]]][id_to_idx[route[i + 1]]] for i in range(len(route) - 1))

    def _pack_to_num_trucks(routes: List[List[int]]) -> List[List[int]]:
        if num_trucks <= 0 or len(routes) <= num_trucks:
            return routes
        loads = [sum(demands[c] for c in r[1:-1]) for r in routes]
        while len(routes) > num_trucks:
            # Try emptying the lightest route by relocating its customers to other routes
            src_idx = min(range(len(routes)), key=lambda idx: loads[idx])
            src_custs = routes[src_idx][1:-1]
            temp_routes = [r[:] for i, r in enumerate(routes) if i != src_idx]
            temp_loads = [l for i, l in enumerate(loads) if i != src_idx]
            moved_all = True
            for c in src_custs:
                d = demands[c]
                best_pos = None
                best_delta = float("inf")
                for r_idx, r in enumerate(temp_routes):
                    if temp_loads[r_idx] + d <= capacity:
                        c_i = id_to_idx[c]
                        for pos in range(1, len(r)):
                            u_i = id_to_idx[r[pos - 1]]
                            v_i = id_to_idx[r[pos]]
                            delta = dist[u_i][c_i] + dist[c_i][v_i] - dist[u_i][v_i]
                            if delta < best_delta:
                                best_delta = delta
                                best_pos = (r_idx, pos)
                if best_pos is None:
                    moved_all = False
                    break
                r_idx, pos = best_pos
                temp_routes[r_idx].insert(pos, c)
                temp_loads[r_idx] += d
            if not moved_all:
                break
            routes = temp_routes
            loads = temp_loads
        return routes

    def split_routes(tour: List[int]):
        routes: List[List[int]] = []
        curr_route = [depot_id]
        curr_load = 0

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

        routes = _pack_to_num_trucks(routes)
        total_dist = sum(_calc_route_dist(r) for r in routes)
        penalty = max(0, len(routes) - num_trucks) * 1_000_000 if num_trucks > 0 else 0

        return routes, total_dist + penalty

    start_time = time.time()
    n = len(customers)
    if n == 0:
        res = {
            "algorithm": "simulated_annealing",
            "instance_name": instance.name,
            "total_distance": 0,
            "routes": [],
            "num_vehicles": 0,
            "execution_time": 0.0
        }
        if return_history:
            res["history"] = [0]
        return res

    curr_tour = customers[:]
    random.shuffle(curr_tour)
    curr_routes, curr_cost = split_routes(curr_tour)

    best_tour = curr_tour[:]
    best_cost = curr_cost
    best_routes = curr_routes
    history = [best_cost]

    t = t_init
    step = 0
    last_emit_time = 0.0
    last_emitted_cost = float("inf")
    while t > t_min and n > 1:
        step += 1
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

            new_routes, new_cost = split_routes(new_tour)
            delta = new_cost - curr_cost

            if delta < 0 or random.random() < math.exp(-delta / t):
                curr_tour = new_tour
                curr_cost = new_cost
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_tour = curr_tour[:]
                    best_routes = new_routes

        t *= cooling_rate
        if return_history:
            history.append(best_cost)

        if progress_callback is not None:
            now = time.time()
            if step == 1 or (now - last_emit_time) >= 0.18 or t <= t_min:
                last_emit_time = now
                emit_dist = sum(_calc_route_dist(r) for r in best_routes)
                if best_cost < last_emitted_cost:
                    last_emitted_cost = best_cost
                    progress_callback(step, emit_dist, best_routes)
                else:
                    progress_callback(step, emit_dist, None)

    elapsed_time = round(time.time() - start_time, 4)
    final_distance = sum(_calc_route_dist(r) for r in best_routes)

    res = {
        "algorithm": "simulated_annealing",
        "instance_name": instance.name,
        "total_distance": final_distance,
        "routes": best_routes,
        "num_vehicles": len(best_routes),
        "execution_time": elapsed_time
    }
    if return_history:
        res["history"] = history
    return res


def solve_from_file(
    file_path: str,
    t_init: float = 200.0,
    t_min: float = 0.05,
    cooling_rate: float = 0.995,
    iters_per_temp: int = 350,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    instance = CVRPParser.parse_file(file_path)
    return solve_cvrp(instance, t_init, t_min, cooling_rate, iters_per_temp, seed)


if __name__ == "__main__":
    result = solve_from_file("CVRPLIB/A-n32-k5.vrp", seed=42)
    print("Cost:", result["total_distance"], "Time:", result["execution_time"], "Vehicles:", result["num_vehicles"])
