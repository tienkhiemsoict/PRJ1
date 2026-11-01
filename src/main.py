import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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


app = FastAPI(
    title="CVRP Optimization Backend",
    description="Backend web service giải bài toán CVRP (Ngưỡng N=20: N<=20 chạy chính xác, N>20 chạy metaheuristic)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BaseCVRPRequest(BaseModel):
    file_path: Optional[str] = None
    file_content: Optional[str] = None


class BacktrackingRequest(BaseCVRPRequest):
    time_limit: float = 5.0
    max_customers: Optional[int] = None


class DPRequest(BaseCVRPRequest):
    max_customers: Optional[int] = None


class SARequest(BaseCVRPRequest):
    t_init: float = 1000.0
    t_min: float = 0.05
    cooling_rate: float = 0.985
    iters_per_temp: int = 40
    seed: Optional[int] = 42


class GARequest(BaseCVRPRequest):
    pop_size: int = 50
    generations: int = 100
    crossover_rate: float = 0.85
    mutation_rate: float = 0.25
    tournament_size: int = 3
    elite_size: int = 2
    seed: Optional[int] = 42


class UnifiedSolveRequest(BaseModel):
    algorithm: str = "auto"
    file_path: Optional[str] = None
    file_content: Optional[str] = None
    params: Dict[str, Any] = {}


class ExperimentRequest(BaseModel):
    instances: Optional[List[str]] = None
    threshold_n: int = 20
    meta_runs: int = 10
    time_limit_bt: float = 5.0
    output_dir: str = "experiments"


def resolve_instance(file_path: Optional[str], file_content: Optional[str]) -> CVRPInstance:
    if file_content and file_content.strip():
        return CVRPParser.parse_string(file_content)
    if file_path:
        path = Path(file_path)
        if not path.is_file():
            alt_path = Path("CVRPLIB") / file_path
            if alt_path.is_file():
                path = alt_path
            else:
                alt_vrp = Path("CVRPLIB") / f"{file_path}.vrp"
                if alt_vrp.is_file():
                    path = alt_vrp
                else:
                    raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        return CVRPParser.parse_file(str(path))
    raise HTTPException(status_code=400, detail="Must provide either file_path or file_content.")


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "CVRP Optimization Backend",
        "policy": "Ngưỡng N=20: N<=20 chỉ chạy thuật toán chính xác, N>20 chỉ chạy metaheuristic",
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
    data_dir = Path("CVRPLIB")
    if not data_dir.exists():
        return {"instances": []}

    results = []
    for f in sorted(data_dir.glob("*.vrp")):
        try:
            inst = CVRPParser.parse_file(str(f))
            opt = get_optimal_value(f)
            rec_algo = "exact (backtracking/dp)" if inst.dimension <= 20 else "metaheuristic (sa/ga)"
            results.append({
                "name": inst.name or f.stem,
                "file": f.name,
                "path": str(f).replace("\\", "/"),
                "dimension": inst.dimension,
                "capacity": inst.capacity,
                "optimal_value": opt,
                "category": "exact" if inst.dimension <= 20 else "metaheuristic",
                "recommended_algorithm": rec_algo
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
    data_dir = Path("CVRPLIB")
    cand = data_dir / f"{name}.vrp" if not name.endswith(".vrp") else data_dir / name
    if not cand.is_file():
        raise HTTPException(status_code=404, detail=f"Instance {name} not found.")

    inst = CVRPParser.parse_file(str(cand))
    opt = get_optimal_value(cand)
    data = inst.to_dict()
    data["optimal_value"] = opt
    data["category"] = "exact" if inst.dimension <= 20 else "metaheuristic"
    return data


@app.post("/api/upload")
async def upload_cvrp_file(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty file or request body.")
    text = body.decode("utf-8")
    inst = CVRPParser.parse_string(text)
    data = inst.to_dict()
    data["category"] = "exact" if inst.dimension <= 20 else "metaheuristic"
    return data


@app.post("/api/solve/backtracking")
def solve_backtracking_endpoint(req: BacktrackingRequest):
    inst = resolve_instance(req.file_path, req.file_content)
    if inst.dimension > 20:
        raise HTTPException(
            status_code=400,
            detail=f"Bài toán có N={inst.dimension} > 20. Theo quy định chỉ áp dụng thuật toán Metaheuristic (simulated-annealing hoặc genetic-algorithm)."
        )
    return solve_bt(inst, time_limit=req.time_limit, max_customers=req.max_customers)


@app.post("/api/solve/dp")
def solve_dp_endpoint(req: DPRequest):
    inst = resolve_instance(req.file_path, req.file_content)
    if inst.dimension > 20:
        raise HTTPException(
            status_code=400,
            detail=f"Bài toán có N={inst.dimension} > 20. Theo quy định chỉ áp dụng thuật toán Metaheuristic (simulated-annealing hoặc genetic-algorithm)."
        )
    return solve_dp(inst, max_customers=req.max_customers)


@app.post("/api/solve/simulated-annealing")
def solve_sa_endpoint(req: SARequest):
    inst = resolve_instance(req.file_path, req.file_content)
    if inst.dimension <= 20:
        raise HTTPException(
            status_code=400,
            detail=f"Bài toán có N={inst.dimension} <= 20. Theo quy định chỉ áp dụng thuật toán chính xác (backtracking hoặc dp)."
        )
    return solve_sa(
        inst,
        t_init=req.t_init,
        t_min=req.t_min,
        cooling_rate=req.cooling_rate,
        iters_per_temp=req.iters_per_temp,
        seed=req.seed
    )


@app.post("/api/solve/genetic-algorithm")
def solve_ga_endpoint(req: GARequest):
    inst = resolve_instance(req.file_path, req.file_content)
    if inst.dimension <= 20:
        raise HTTPException(
            status_code=400,
            detail=f"Bài toán có N={inst.dimension} <= 20. Theo quy định chỉ áp dụng thuật toán chính xác (backtracking hoặc dp)."
        )
    return solve_ga(
        inst,
        pop_size=req.pop_size,
        generations=req.generations,
        crossover_rate=req.crossover_rate,
        mutation_rate=req.mutation_rate,
        tournament_size=req.tournament_size,
        elite_size=req.elite_size,
        seed=req.seed
    )


@app.post("/api/solve")
def solve_unified_endpoint(req: UnifiedSolveRequest):
    inst = resolve_instance(req.file_path, req.file_content)
    algo = req.algorithm.lower().replace("-", "_")
    p = req.params

    if algo == "auto":
        algo = "dynamic_programming" if inst.dimension <= 20 else "simulated_annealing"

    if algo in ["backtracking", "bnb", "branch_and_bound"]:
        if inst.dimension > 20:
            raise HTTPException(
                status_code=400,
                detail=f"N={inst.dimension} > 20: Không được chạy thuật toán chính xác. Vui lòng chọn thuật toán Metaheuristic."
            )
        return solve_bt(
            inst,
            time_limit=p.get("time_limit", 5.0),
            max_customers=p.get("max_customers", None)
        )

    elif algo in ["dp", "dynamic_programming"]:
        if inst.dimension > 20:
            raise HTTPException(
                status_code=400,
                detail=f"N={inst.dimension} > 20: Không được chạy thuật toán chính xác. Vui lòng chọn thuật toán Metaheuristic."
            )
        return solve_dp(
            inst,
            max_customers=p.get("max_customers", None)
        )

    elif algo in ["sa", "simulated_annealing"]:
        if inst.dimension <= 20:
            raise HTTPException(
                status_code=400,
                detail=f"N={inst.dimension} <= 20: Thuộc nhóm bài toán nhỏ. Vui lòng chọn thuật toán chính xác (backtracking hoặc dp)."
            )
        return solve_sa(
            inst,
            t_init=p.get("t_init", 1000.0),
            t_min=p.get("t_min", 0.05),
            cooling_rate=p.get("cooling_rate", 0.985),
            iters_per_temp=p.get("iters_per_temp", 40),
            seed=p.get("seed", 42)
        )

    elif algo in ["ga", "genetic_algorithm"]:
        if inst.dimension <= 20:
            raise HTTPException(
                status_code=400,
                detail=f"N={inst.dimension} <= 20: Thuộc nhóm bài toán nhỏ. Vui lòng chọn thuật toán chính xác (backtracking hoặc dp)."
            )
        return solve_ga(
            inst,
            pop_size=p.get("pop_size", 50),
            generations=p.get("generations", 100),
            crossover_rate=p.get("crossover_rate", 0.85),
            mutation_rate=p.get("mutation_rate", 0.25),
            tournament_size=p.get("tournament_size", 3),
            elite_size=p.get("elite_size", 2),
            seed=p.get("seed", 42)
        )

    else:
        raise HTTPException(status_code=400, detail=f"Thuật toán không hỗ trợ: '{req.algorithm}'.")


@app.post("/api/experiment/run")
def run_experiment_endpoint(req: ExperimentRequest):
    data_dir = Path("CVRPLIB")
    if not req.instances:
        instances = [str(f) for f in sorted(data_dir.glob("*.vrp"))]
    else:
        instances = []
        for item in req.instances:
            p = Path(item)
            if not p.is_file():
                alt = data_dir / item if item.endswith(".vrp") else data_dir / f"{item}.vrp"
                if alt.is_file():
                    p = alt
            instances.append(str(p).replace("\\", "/"))

    res = run_experiment(
        instance_files=instances,
        threshold_n=req.threshold_n,
        meta_runs=req.meta_runs,
        time_limit_bt=req.time_limit_bt,
        output_dir=req.output_dir
    )
    return res


@app.get("/api/experiment/results")
def get_experiment_results(output_dir: str = "experiments"):
    p = Path(output_dir) / "benchmark_results.json"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Chưa có kết quả thực nghiệm. Hãy chạy thực nghiệm trước.")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
