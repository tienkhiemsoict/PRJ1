import asyncio
import json
import math
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from src.cvrp_parser import CVRPParser, CVRPInstance
    from src.backtracking import solve_cvrp as solve_bt
    from src.dynamic_programming import solve_cvrp as solve_dp
    from src.simulated_annealing import solve_cvrp as solve_sa
    from src.genetic_algorithm import solve_cvrp as solve_ga
    from src.experiment_pipeline import run_experiment, get_optimal_value
except ImportError:
    from cvrp_parser import CVRPParser, CVRPInstance
    from backtracking import solve_cvrp as solve_bt
    from dynamic_programming import solve_cvrp as solve_dp
    from simulated_annealing import solve_cvrp as solve_sa
    from genetic_algorithm import solve_cvrp as solve_ga
    from experiment_pipeline import run_experiment, get_optimal_value


THRESHOLD_N = 15
DATA_DIR = Path("CVRPLIB")
DEFAULT_CENTER = (21.0045, 105.8431)  # Khu vực Bách Khoa Hà Nội

app = FastAPI(
    title="CVRP Optimization Backend",
    description=f"Backend web service giải bài toán CVRP (Ngưỡng N={THRESHOLD_N}: N<={THRESHOLD_N} chạy chính xác, N>{THRESHOLD_N} chạy metaheuristic)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerPoint(BaseModel):
    id: Optional[int] = None
    lat: float
    lng: float
    demand: int = 10


class DepotPoint(BaseModel):
    lat: float
    lng: float


class BaseCVRPRequest(BaseModel):
    file_path: Optional[str] = None
    file_content: Optional[str] = None
    depot: Optional[DepotPoint] = None
    customers: Optional[List[CustomerPoint]] = None
    capacity: int = 100
    num_trucks: int = 0


class BacktrackingRequest(BaseCVRPRequest):
    time_limit: Optional[float] = None
    max_customers: Optional[int] = None


class DPRequest(BaseCVRPRequest):
    max_customers: Optional[int] = None


class SARequest(BaseCVRPRequest):
    t_init: float = 1000.0
    t_min: float = 0.05
    cooling_rate: float = 0.985
    iters_per_temp: int = 40
    seed: Optional[int] = None
    return_history: bool = False


class GARequest(BaseCVRPRequest):
    pop_size: int = 50
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    tournament_size: int = 3
    elite_size: int = 2
    seed: Optional[int] = None
    return_history: bool = False


class UnifiedSolveRequest(BaseCVRPRequest):
    algorithm: str = "auto"
    params: Dict[str, Any] = {}


class ExperimentRequest(BaseModel):
    instances: Optional[List[str]] = None
    threshold_n: int = THRESHOLD_N
    meta_runs: int = 10
    time_limit_bt: Optional[float] = None
    output_dir: str = "experiments"


ALGO_REGISTRY = {
    "backtracking": (solve_bt, True, ("time_limit", "max_customers")),
    "bnb": (solve_bt, True, ("time_limit", "max_customers")),
    "branch_and_bound": (solve_bt, True, ("time_limit", "max_customers")),
    "dp": (solve_dp, True, ("max_customers",)),
    "dynamic_programming": (solve_dp, True, ("max_customers",)),
    "sa": (solve_sa, False, ("t_init", "t_min", "cooling_rate", "iters_per_temp", "seed", "return_history")),
    "simulated_annealing": (solve_sa, False, ("t_init", "t_min", "cooling_rate", "iters_per_temp", "seed", "return_history")),
    "ga": (solve_ga, False, ("pop_size", "generations", "crossover_rate", "mutation_rate", "tournament_size", "elite_size", "seed", "return_history")),
    "genetic_algorithm": (solve_ga, False, ("pop_size", "generations", "crossover_rate", "mutation_rate", "tournament_size", "elite_size", "seed", "return_history")),
}

BASE_FIELDS = {"file_path", "file_content", "depot", "customers", "capacity", "num_trucks"}


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return int(round(2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))))


def _instance_to_map_coords(inst: CVRPInstance) -> Dict[int, Tuple[float, float]]:
    """Map Euclidean CVRP coords to geographic (lat, lng) if not already in lat/lng range."""
    if not inst.coords:
        return {}
    xs = [c[0] for c in inst.coords.values()]
    ys = [c[1] for c in inst.coords.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if 8.0 <= min_x <= 24.0 and 102.0 <= min_y <= 110.0:
        return {nid: (float(lat), float(lng)) for nid, (lat, lng) in inst.coords.items()}

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    deg_span = 0.04
    c_lat, c_lng = DEFAULT_CENTER
    mapped: Dict[int, Tuple[float, float]] = {}
    for nid, (x, y) in inst.coords.items():
        lat = c_lat + ((y - min_y) / span_y - 0.5) * deg_span
        lng = c_lng + ((x - min_x) / span_x - 0.5) * deg_span
        mapped[nid] = (round(lat, 6), round(lng, 6))
    return mapped


def _build_instance_from_points(
    depot: DepotPoint,
    customers: List[CustomerPoint],
    capacity: int,
    num_trucks: int
) -> CVRPInstance:
    inst = CVRPInstance(
        name="Custom-Map",
        dimension=1 + len(customers),
        capacity=max(1, capacity),
        num_trucks=max(0, num_trucks),
        depot_id=1
    )
    inst.coords[1] = (depot.lat, depot.lng)
    inst.demands[1] = 0
    for idx, c in enumerate(customers, start=2):
        inst.coords[idx] = (c.lat, c.lng)
        inst.demands[idx] = max(1, c.demand)

    node_ids = sorted(inst.coords.keys())
    n = len(node_ids)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lng1 = inst.coords[node_ids[i]]
            lat2, lng2 = inst.coords[node_ids[j]]
            d = _haversine_m(lat1, lng1, lat2, lng2)
            matrix[i][j] = matrix[j][i] = d
    inst.distance_matrix = matrix
    return inst


def _resolve_vrp_path(file_path: str) -> Path:
    candidates = [
        Path(file_path),
        DATA_DIR / file_path,
        DATA_DIR / f"{file_path}.vrp",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise HTTPException(status_code=404, detail=f"File not found: {file_path}")


def resolve_instance(req: BaseCVRPRequest) -> CVRPInstance:
    if req.depot is not None and req.customers is not None:
        return _build_instance_from_points(req.depot, req.customers, req.capacity, req.num_trucks)
    if req.file_content and req.file_content.strip():
        return CVRPParser.parse_string(req.file_content)
    if req.file_path:
        return CVRPParser.parse_file(str(_resolve_vrp_path(req.file_path)))
    raise HTTPException(status_code=400, detail="Must provide file_path, file_content, or depot and customers.")


def _enrich_result(inst: CVRPInstance, res: Dict[str, Any]) -> Dict[str, Any]:
    map_coords = _instance_to_map_coords(inst)
    node_ids = sorted(inst.coords.keys() if inst.coords else range(1, inst.dimension + 1))
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    dist = inst.distance_matrix
    cap = max(1, inst.capacity)

    route_details = []
    fill_rates = []
    for idx, r in enumerate(res.get("routes", []), start=1):
        load = sum(inst.demands.get(nid, 0) for nid in r if nid != inst.depot_id)
        r_dist = 0
        for i in range(len(r) - 1):
            u, v = id_to_idx.get(r[i]), id_to_idx.get(r[i + 1])
            if u is not None and v is not None:
                r_dist += dist[u][v]
        fr = round((load / cap) * 100.0, 1)
        fill_rates.append(fr)
        route_details.append({
            "vehicle": idx,
            "route": r,
            "load": load,
            "capacity": inst.capacity,
            "fill_rate": fr,
            "distance": r_dist,
            "path_coords": [[map_coords[nid][0], map_coords[nid][1]] for nid in r if nid in map_coords]
        })

    res["fill_rate"] = round(sum(fill_rates) / len(fill_rates), 1) if fill_rates else 0.0
    res["capacity"] = inst.capacity
    res["dimension"] = inst.dimension
    res["route_details"] = route_details
    res["map_nodes"] = [
        {
            "id": nid,
            "lat": map_coords[nid][0],
            "lng": map_coords[nid][1],
            "demand": inst.demands.get(nid, 0),
            "is_depot": nid == inst.depot_id
        }
        for nid in node_ids if nid in map_coords
    ]
    return res


def _validate_instance_feasibility(inst: CVRPInstance) -> None:
    cust_demands = [d for nid, d in inst.demands.items() if nid != inst.depot_id]
    if not cust_demands:
        return
    max_d = max(cust_demands)
    if max_d > inst.capacity:
        raise HTTPException(
            status_code=400,
            detail=f"Nhu cầu khách hàng ({max_d}) vượt quá tải trọng mỗi xe ({inst.capacity})."
        )
    total_d = sum(cust_demands)
    if inst.num_trucks > 0 and total_d > inst.num_trucks * inst.capacity:
        raise HTTPException(
            status_code=400,
            detail=f"Tổng nhu cầu ({total_d}) vượt quá tổng sức chứa của {inst.num_trucks} xe ({inst.num_trucks * inst.capacity}). Hãy tăng số xe hoặc tải trọng."
        )


def _dispatch_solve(algorithm: str, inst: CVRPInstance, params: Dict[str, Any]) -> Dict[str, Any]:
    _validate_instance_feasibility(inst)

    algo = algorithm.lower().replace("-", "_")
    if algo == "auto":
        algo = "dynamic_programming" if inst.dimension <= THRESHOLD_N else "simulated_annealing"

    entry = ALGO_REGISTRY.get(algo)
    if entry is None:
        raise HTTPException(status_code=400, detail=f"Thuật toán không hỗ trợ: '{algorithm}'.")

    solver, is_exact, allowed_keys = entry
    if is_exact and inst.dimension > THRESHOLD_N:
        raise HTTPException(
            status_code=400,
            detail=f"N={inst.dimension} > {THRESHOLD_N}: Không được chạy thuật toán chính xác. Vui lòng chọn thuật toán Metaheuristic (simulated-annealing hoặc genetic-algorithm)."
        )
    if not is_exact and inst.dimension <= THRESHOLD_N:
        raise HTTPException(
            status_code=400,
            detail=f"N={inst.dimension} <= {THRESHOLD_N}: Thuộc nhóm bài toán nhỏ. Vui lòng chọn thuật toán chính xác (backtracking hoặc dp)."
        )

    kwargs = {k: params[k] for k in allowed_keys if k in params and params[k] is not None}
    res = solver(inst, **kwargs)
    if inst.num_trucks > 0 and res.get("num_vehicles", 0) > inst.num_trucks:
        raise HTTPException(
            status_code=400,
            detail=f"Không thể xếp hợp lệ vào giới hạn {inst.num_trucks} xe (phương án cần {res['num_vehicles']} xe). Hãy tăng số xe hoặc tải trọng."
        )
    return _enrich_result(inst, res)


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "CVRP Optimization Backend",
        "policy": f"Ngưỡng N={THRESHOLD_N}: N<={THRESHOLD_N} chỉ chạy thuật toán chính xác, N>{THRESHOLD_N} chỉ chạy metaheuristic",
        "supported_algorithms": [
            "backtracking",
            "dynamic_programming",
            "simulated_annealing",
            "genetic_algorithm",
            "auto"
        ]
    }


@app.get("/api/instances")
def list_instances():
    if not DATA_DIR.exists():
        return {"instances": []}

    results = []
    for f in sorted(DATA_DIR.glob("*.vrp")):
        try:
            inst = CVRPParser.parse_file(str(f))
            opt = get_optimal_value(f)
            is_exact = inst.dimension <= THRESHOLD_N
            results.append({
                "name": inst.name or f.stem,
                "file": f.name,
                "path": str(f).replace("\\", "/"),
                "dimension": inst.dimension,
                "capacity": inst.capacity,
                "optimal_value": opt,
                "category": "exact" if is_exact else "metaheuristic",
                "recommended_algorithm": "exact (backtracking/dp)" if is_exact else "metaheuristic (sa/ga)"
            })
        except Exception:
            results.append({
                "name": f.stem,
                "file": f.name,
                "path": str(f).replace("\\", "/"),
                "error": "Failed to parse"
            })
    return {"count": len(results), "instances": results}


@app.get("/api/instances/{name}")
def get_instance_details(name: str):
    cand = _resolve_vrp_path(name)
    inst = CVRPParser.parse_file(str(cand))
    data = inst.to_dict()
    data["optimal_value"] = get_optimal_value(cand)
    data["category"] = "exact" if inst.dimension <= THRESHOLD_N else "metaheuristic"
    map_coords = _instance_to_map_coords(inst)
    data["map_nodes"] = [
        {
            "id": nid,
            "lat": map_coords[nid][0],
            "lng": map_coords[nid][1],
            "demand": inst.demands.get(nid, 0),
            "is_depot": nid == inst.depot_id
        }
        for nid in sorted(map_coords.keys())
    ]
    return data


@app.post("/api/upload")
async def upload_cvrp_file(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty file or request body.")
    inst = CVRPParser.parse_string(body.decode("utf-8"))
    data = inst.to_dict()
    data["category"] = "exact" if inst.dimension <= THRESHOLD_N else "metaheuristic"
    map_coords = _instance_to_map_coords(inst)
    data["map_nodes"] = [
        {
            "id": nid,
            "lat": map_coords[nid][0],
            "lng": map_coords[nid][1],
            "demand": inst.demands.get(nid, 0),
            "is_depot": nid == inst.depot_id
        }
        for nid in sorted(map_coords.keys())
    ]
    return data


@app.post("/api/solve/backtracking")
def solve_backtracking_endpoint(req: BacktrackingRequest):
    inst = resolve_instance(req)
    return _dispatch_solve("backtracking", inst, req.model_dump(exclude=BASE_FIELDS))


@app.post("/api/solve/dp")
def solve_dp_endpoint(req: DPRequest):
    inst = resolve_instance(req)
    return _dispatch_solve("dp", inst, req.model_dump(exclude=BASE_FIELDS))


@app.post("/api/solve/simulated-annealing")
def solve_sa_endpoint(req: SARequest):
    inst = resolve_instance(req)
    return _dispatch_solve("simulated_annealing", inst, req.model_dump(exclude=BASE_FIELDS))


@app.post("/api/solve/genetic-algorithm")
def solve_ga_endpoint(req: GARequest):
    inst = resolve_instance(req)
    return _dispatch_solve("genetic_algorithm", inst, req.model_dump(exclude=BASE_FIELDS))


@app.post("/api/solve")
def solve_unified_endpoint(req: UnifiedSolveRequest):
    inst = resolve_instance(req)
    params = dict(req.params)
    if inst.dimension > THRESHOLD_N and "return_history" not in params:
        params["return_history"] = True
    return _dispatch_solve(req.algorithm, inst, params)


@app.post("/api/compare")
def compare_algorithms(req: UnifiedSolveRequest):
    inst = resolve_instance(req)
    results = []
    if inst.dimension <= THRESHOLD_N:
        algos = [
            ("Backtracking", "backtracking", {"time_limit": 10.0}),
            ("Dynamic Programming", "dynamic_programming", {})
        ]
    else:
        algos = [
            ("Genetic Algorithm", "genetic_algorithm", {"return_history": True}),
            ("Simulated Annealing", "simulated_annealing", {"return_history": True})
        ]

    for label, key, extra in algos:
        merged_params = {**req.params, **extra}
        res = _dispatch_solve(key, inst, merged_params)
        results.append({
            "algo": label,
            "dist": res["total_distance"],
            "vehicles": res["num_vehicles"],
            "fill_rate": res["fill_rate"],
            "time_ms": round(res["execution_time"] * 1000.0, 2),
            "history": res.get("history", [])
        })
    return results


@app.websocket("/ws/solve")
async def websocket_solve(websocket: WebSocket):
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        req = UnifiedSolveRequest(**payload)
        inst = resolve_instance(req)
        params = dict(req.params)
        if inst.dimension > THRESHOLD_N:
            params["return_history"] = True

        res = _dispatch_solve(req.algorithm, inst, params)
        history = res.get("history", [])
        if history:
            step_stride = max(1, len(history) // 25)
            for idx in range(0, len(history), step_stride):
                await websocket.send_json({
                    "type": "progress",
                    "generation": idx + 1,
                    "best_fitness": history[idx]
                })
                await asyncio.sleep(0.02)

        await websocket.send_json({
            "type": "complete",
            "result": res
        })
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/api/experiment/run")
def run_experiment_endpoint(req: ExperimentRequest):
    if not req.instances:
        instances = [str(f) for f in sorted(DATA_DIR.glob("*.vrp"))]
    else:
        instances = [str(_resolve_vrp_path(item)).replace("\\", "/") for item in req.instances]

    return run_experiment(
        instance_files=instances,
        threshold_n=req.threshold_n,
        meta_runs=req.meta_runs,
        time_limit_bt=req.time_limit_bt,
        output_dir=req.output_dir
    )


@app.get("/api/experiment/results")
def get_experiment_results(output_dir: str = "experiments"):
    p = Path(output_dir) / "benchmark_results.json"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Chưa có kết quả thực nghiệm. Hãy chạy thực nghiệm trước.")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
