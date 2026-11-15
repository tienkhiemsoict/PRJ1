import random
import time
from typing import List, Dict, Any, Optional
try:
    from src.cvrp_parser import CVRPParser, CVRPInstance
except ImportError:
    from cvrp_parser import CVRPParser, CVRPInstance


def solve_cvrp(
    instance: CVRPInstance,
    pop_size: int = 80,
    generations: int = 3500,
    crossover_rate: float = 0.75,
    mutation_rate: float = 0.5,
    tournament_size: int = 5,
    elite_size: int = 3,
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
    num_trucks = instance.num_trucks
    dist = instance.distance_matrix

    def _calc_route_dist(route: List[int]) -> int:
        return sum(dist[id_to_idx[route[i]]][id_to_idx[route[i + 1]]] for i in range(len(route) - 1))

    def _pack_to_num_trucks(routes: List[List[int]]) -> List[List[int]]:
        if num_trucks <= 0 or len(routes) <= num_trucks:
            return routes
        loads = [sum(demands[c] for c in r[1:-1]) for r in routes]
        while len(routes) > num_trucks:
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

    def ox_crossover(p1: List[int], p2: List[int]) -> List[int]:
        m = len(p1)
        if m < 2:
            return p1[:]
        i, j = sorted(random.sample(range(m), 2))
        child = [0] * m
        child[i:j + 1] = p1[i:j + 1]
        p1_set = set(child[i:j + 1])
        ptr = (j + 1) % m
        for gene in p2[j + 1:] + p2[:j + 1]:
            if gene not in p1_set:
                child[ptr] = gene
                ptr = (ptr + 1) % m
        return child

    def mutate(tour: List[int]):
        if len(tour) < 2:
            return
        i, j = sorted(random.sample(range(len(tour)), 2))
        tour[i:j + 1] = reversed(tour[i:j + 1])

    start_time = time.time()
    n = len(customers)
    if n == 0:
        res = {
            "algorithm": "genetic_algorithm",
            "instance_name": instance.name,
            "total_distance": 0,
            "routes": [],
            "num_vehicles": 0,
            "execution_time": 0.0
        }
        if return_history:
            res["history"] = [0]
        return res

    pop = [random.sample(customers, n) for _ in range(pop_size)]
    best_ind = pop[0][:]
    best_cost = float("inf")
    history: List[float] = []

    for _ in range(generations):
        costs = [split_routes(ind)[1] for ind in pop]
        for ind, c in zip(pop, costs):
            if c < best_cost:
                best_cost = c
                best_ind = ind[:]

        if return_history:
            history.append(best_cost)

        ranked = sorted(zip(pop, costs), key=lambda x: x[1])
        new_pop = [ranked[k][0][:] for k in range(min(elite_size, pop_size))]

        def select():
            cands = random.sample(ranked, min(tournament_size, len(ranked)))
            return min(cands, key=lambda x: x[1])[0]

        while len(new_pop) < pop_size:
            p1 = select()
            p2 = select()
            if random.random() < crossover_rate and n >= 2:
                c1 = ox_crossover(p1, p2)
            else:
                c1 = p1[:]
            if random.random() < mutation_rate:
                mutate(c1)
            new_pop.append(c1)

        pop = new_pop

    best_routes, _ = split_routes(best_ind)
    best_cost = sum(_calc_route_dist(r) for r in best_routes)
    elapsed_time = round(time.time() - start_time, 4)

    res = {
        "algorithm": "genetic_algorithm",
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
    pop_size: int = 80,
    generations: int = 3500,
    crossover_rate: float = 0.75,
    mutation_rate: float = 0.5,
    tournament_size: int = 5,
    elite_size: int = 3,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    instance = CVRPParser.parse_file(file_path)
    return solve_cvrp(
        instance,
        pop_size,
        generations,
        crossover_rate,
        mutation_rate,
        tournament_size,
        elite_size,
        seed
    )


if __name__ == "__main__":
    result = solve_from_file("CVRPLIB/A-n32-k5.vrp", seed=42)
    print("Cost:", result["total_distance"], "Time:", result["execution_time"], "Vehicles:", result["num_vehicles"])
