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
    pop_size: int = 50,
    generations: int = 100,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.25,
    tournament_size: int = 3,
    elite_size: int = 2,
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

    def ox_crossover(p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        if n < 2:
            return p1[:]
        i, j = sorted(random.sample(range(n), 2))
        child = [None] * n
        child[i:j + 1] = p1[i:j + 1]
        p1_set = set(child[i:j + 1])
        ptr = (j + 1) % n
        for gene in p2[j + 1:] + p2[:j + 1]:
            if gene not in p1_set:
                child[ptr] = gene
                ptr = (ptr + 1) % n
        return child

    def mutate(tour: List[int]):
        if len(tour) < 2:
            return
        i, j = sorted(random.sample(range(len(tour)), 2))
        tour[i:j + 1] = reversed(tour[i:j + 1])

    start_time = time.time()
    n = len(customers)
    if n == 0:
        return {
            "algorithm": "genetic_algorithm",
            "instance_name": instance.name,
            "total_distance": 0,
            "routes": [],
            "num_vehicles": 0,
            "execution_time": 0.0
        }

    nn_tour = get_nn_tour()
    pop = [random.sample(customers, n) for _ in range(max(1, pop_size - 1))]
    pop.append(nn_tour[:])

    best_ind = pop[0][:]
    best_cost = float("inf")
    history = []

    for _ in range(generations):
        costs = [split_routes(ind)[1] for ind in pop]
        for ind, c in zip(pop, costs):
            if c < best_cost:
                best_cost = c
                best_ind = ind[:]

        if return_history:
            history.append(best_cost)

        ranked = sorted(zip(pop, costs), key=lambda x: x[1])
        new_pop = [ranked[k][0][:] for k in range(min(elite_size, len(ranked)))]

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

    raw_routes, _ = split_routes(best_ind)
    best_routes = [two_opt_route(r, dist, id_to_idx) for r in raw_routes]
    best_cost = sum(dist[id_to_idx[r[i]]][id_to_idx[r[i + 1]]] for r in best_routes for i in range(len(r) - 1))
    if return_history:
        history.append(best_cost)

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
    pop_size: int = 50,
    generations: int = 100,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.25,
    tournament_size: int = 3,
    elite_size: int = 2,
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
    print("Cost:", result["total_distance"])
    print("Time:", result["execution_time"])
    print("Vehicles:", result["num_vehicles"])
    print("Routes:", result["routes"])
