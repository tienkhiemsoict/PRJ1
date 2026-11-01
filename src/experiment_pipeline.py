import os
import sys
import re
import csv
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from src.cvrp_parser import CVRPParser, CVRPInstance
    from src.backtracking import solve_cvrp as solve_bt
    from src.dynamic_programming import solve_cvrp as solve_dp
    from src.simulated_annealing import solve_cvrp as solve_sa
    from src.genetic_algorithm import solve_cvrp as solve_ga
except ImportError:
    from cvrp_parser import CVRPParser, CVRPInstance
    from backtracking import solve_cvrp as solve_bt
    from dynamic_programming import solve_cvrp as solve_dp
    from simulated_annealing import solve_cvrp as solve_sa
    from genetic_algorithm import solve_cvrp as solve_ga


def get_optimal_value(filepath: Path) -> Optional[int]:
    sol_p = filepath.with_suffix(".sol")
    if sol_p.exists():
        text = sol_p.read_text(encoding="utf-8")
        m = re.search(r"Cost\s+(\d+)", text, re.IGNORECASE)
        if m:
            return int(m.group(1))

    if filepath.exists():
        text = filepath.read_text(encoding="utf-8")
        m = re.search(r"Optimal value:\s*(\d+)", text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def run_experiment(
    instance_files: List[str],
    threshold_n: int = 20,
    meta_runs: int = 10,
    time_limit_bt: float = 5.0,
    sa_params: Optional[Dict[str, Any]] = None,
    ga_params: Optional[Dict[str, Any]] = None,
    output_dir: str = "experiments",
    verbose: bool = True
) -> Dict[str, Any]:
    if sa_params is None:
        sa_params = {"t_init": 1000.0, "t_min": 0.05, "cooling_rate": 0.985, "iters_per_temp": 40}

    if ga_params is None:
        ga_params = {"pop_size": 50, "generations": 100, "crossover_rate": 0.85, "mutation_rate": 0.25}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    exact_results: List[Dict[str, Any]] = []
    meta_results: List[Dict[str, Any]] = []

    total_files = len(instance_files)

    for f_idx, file_str in enumerate(instance_files, 1):
        fpath = Path(file_str)
        if not fpath.exists():
            continue

        instance = CVRPParser.parse_file(str(fpath))
        testcase = instance.name or fpath.stem
        n = instance.dimension
        cap = instance.capacity
        optimal = get_optimal_value(fpath)

        # N <= threshold_n (N <= 20): CHỈ CHẠY 2 THUẬT TOÁN CHÍNH XÁC
        if n <= threshold_n:
            if verbose:
                print(f"[{f_idx}/{total_files}] (Exact N<={threshold_n}) {testcase}: N={n}, Cap={cap}, Optimal={optimal}")

            # 1. Backtracking
            res_bt = solve_bt(instance, time_limit=time_limit_bt, max_customers=None)
            exact_results.append({
                "testcase": testcase,
                "n": n,
                "cap": cap,
                "vehicle": res_bt["num_vehicles"],
                "optimal": optimal,
                "algor": "backtracking",
                "time": round(res_bt["execution_time"], 4)
            })

            # 2. Dynamic Programming
            res_dp = solve_dp(instance, max_customers=None)
            exact_results.append({
                "testcase": testcase,
                "n": n,
                "cap": cap,
                "vehicle": res_dp["num_vehicles"],
                "optimal": optimal,
                "algor": "dynamic_programming",
                "time": round(res_dp["execution_time"], 4)
            })

        # N > threshold_n (N > 20): CHỈ CHẠY 2 THUẬT TOÁN METAHEURISTIC
        else:
            if verbose:
                print(f"[{f_idx}/{total_files}] (Meta N>{threshold_n}) {testcase}: N={n}, Cap={cap}, Optimal={optimal} ({meta_runs} runs)")

            # 1. Simulated Annealing (10 runs)
            sa_times: List[float] = []
            sa_gaps: List[float] = []
            sa_vehicles: List[int] = []

            for r in range(meta_runs):
                res_sa = solve_sa(
                    instance,
                    t_init=sa_params.get("t_init", 1000.0),
                    t_min=sa_params.get("t_min", 0.05),
                    cooling_rate=sa_params.get("cooling_rate", 0.985),
                    iters_per_temp=sa_params.get("iters_per_temp", 40),
                    seed=42 + r
                )
                sa_times.append(res_sa["execution_time"])
                sa_vehicles.append(res_sa["num_vehicles"])
                if optimal is not None and optimal > 0:
                    gap = ((res_sa["total_distance"] - optimal) / optimal) * 100.0
                    sa_gaps.append(gap)

            avg_time_sa = round(sum(sa_times) / len(sa_times), 4)
            avg_gap_sa = round(sum(sa_gaps) / len(sa_gaps), 2) if sa_gaps else None
            vehicle_sa = sa_vehicles[0] if sa_vehicles else 0

            meta_results.append({
                "testcase": testcase,
                "n": n,
                "cap": cap,
                "vehicle": vehicle_sa,
                "optimal": optimal,
                "algor": "simulated_annealing",
                "runs": meta_runs,
                "avg_time": avg_time_sa,
                "avg_gap": f"{avg_gap_sa:+.2f}%" if avg_gap_sa is not None else "N/A"
            })

            # 2. Genetic Algorithm (10 runs)
            ga_times: List[float] = []
            ga_gaps: List[float] = []
            ga_vehicles: List[int] = []

            for r in range(meta_runs):
                res_ga = solve_ga(
                    instance,
                    pop_size=ga_params.get("pop_size", 50),
                    generations=ga_params.get("generations", 100),
                    crossover_rate=ga_params.get("crossover_rate", 0.85),
                    mutation_rate=ga_params.get("mutation_rate", 0.25),
                    seed=42 + r
                )
                ga_times.append(res_ga["execution_time"])
                ga_vehicles.append(res_ga["num_vehicles"])
                if optimal is not None and optimal > 0:
                    gap = ((res_ga["total_distance"] - optimal) / optimal) * 100.0
                    ga_gaps.append(gap)

            avg_time_ga = round(sum(ga_times) / len(ga_times), 4)
            avg_gap_ga = round(sum(ga_gaps) / len(ga_gaps), 2) if ga_gaps else None
            vehicle_ga = ga_vehicles[0] if ga_vehicles else 0

            meta_results.append({
                "testcase": testcase,
                "n": n,
                "cap": cap,
                "vehicle": vehicle_ga,
                "optimal": optimal,
                "algor": "genetic_algorithm",
                "runs": meta_runs,
                "avg_time": avg_time_ga,
                "avg_gap": f"{avg_gap_ga:+.2f}%" if avg_gap_ga is not None else "N/A"
            })

    save_benchmark_outputs(exact_results, meta_results, out_path)
    generate_all_charts(exact_results, meta_results, out_path)

    return {
        "status": "success",
        "threshold_n": threshold_n,
        "meta_runs": meta_runs,
        "exact_count": len(exact_results),
        "meta_count": len(meta_results),
        "exact_results": exact_results,
        "meta_results": meta_results,
        "output_files": {
            "exact_csv": str(out_path / "exact_results.csv"),
            "meta_csv": str(out_path / "metaheuristic_results.csv"),
            "json": str(out_path / "benchmark_results.json"),
            "markdown": str(out_path / "benchmark_summary.md"),
            "chart_exact": str(out_path / "chart_exact_comparison.png"),
            "chart_convergence": str(out_path / "chart_meta_convergence.png"),
            "chart_meta": str(out_path / "chart_meta_analysis.png")
        }
    }


def save_benchmark_outputs(exact_results: List[Dict[str, Any]], meta_results: List[Dict[str, Any]], out_dir: Path):
    # 1. JSON
    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"exact_results": exact_results, "metaheuristic_results": meta_results}, f, indent=2, ensure_ascii=False)

    # 2. Exact CSV
    exact_csv = out_dir / "exact_results.csv"
    with open(exact_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["testcase", "n", "cap", "vehicle", "optimal", "algor", "time"])
        writer.writeheader()
        for row in exact_results:
            writer.writerow(row)

    # 3. Metaheuristic CSV
    meta_csv = out_dir / "metaheuristic_results.csv"
    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["testcase", "n", "cap", "vehicle", "optimal", "algor", "runs", "avg_time", "avg_gap"])
        writer.writeheader()
        for row in meta_results:
            writer.writerow(row)

    # 4. Markdown Report
    md_path = out_dir / "benchmark_summary.md"
    lines = [
        "# CVRP Benchmark Summary Report",
        "",
        f"> **Thời gian tạo**: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Quy tắc phân loại**: Ngưỡng $N = 20$  ",
        f"> - $N \\le 20$: Chỉ chạy 2 thuật toán chính xác (`backtracking`, `dynamic_programming`).  ",
        f"> - $N > 20$: Chỉ chạy 2 thuật toán metaheuristic (`simulated_annealing`, `genetic_algorithm`), số lần chạy $runs = 10$.  ",
        "",
        "---",
        "",
        "## 1. Kết quả thuật toán chính xác (N <= 20)",
        "",
        "| testcase | n | cap | vehicle | optimal | algor | time |",
        "| :--- | :---: | :---: | :---: | :---: | :--- | :---: |"
    ]

    for r in exact_results:
        opt_str = str(r["optimal"]) if r["optimal"] is not None else "N/A"
        lines.append(f"| `{r['testcase']}` | {r['n']} | {r['cap']} | {r['vehicle']} | {opt_str} | **{r['algor']}** | {r['time']:.4f} s |")

    lines.extend([
        "",
        "### Biểu đồ so sánh thời gian thuật toán chính xác",
        "![So sánh thuật toán chính xác](chart_exact_comparison.png)",
        "",
        "---",
        "",
        "## 2. Báo cáo tổng hợp thuật toán Metaheuristic (N > 20, 10 runs trung bình)",
        ""
    ])

    def parse_gap(val):
        try:
            return float(str(val).replace("%", "").replace("+", "").strip())
        except Exception:
            return 0.0

    sa_rows = [r for r in meta_results if r["algor"] == "simulated_annealing"]
    ga_rows = [r for r in meta_results if r["algor"] == "genetic_algorithm"]

    if sa_rows and ga_rows:
        sa_avg_gap = sum(parse_gap(r["avg_gap"]) for r in sa_rows) / len(sa_rows)
        ga_avg_gap = sum(parse_gap(r["avg_gap"]) for r in ga_rows) / len(ga_rows)
        sa_avg_time = sum(float(r["avg_time"]) for r in sa_rows) / len(sa_rows)
        ga_avg_time = sum(float(r["avg_time"]) for r in ga_rows) / len(ga_rows)

        lines.extend([
            "### Bảng tổng kết trung bình (83 bài toán)",
            "",
            "| Thuật toán | Số bài toán | Số runs | Thời gian TB (`avg_time`) | Sai số TB (`avg_gap`) |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| **Simulated Annealing (SA)** | {len(sa_rows)} | 10 | {sa_avg_time:.4f} s | **+{sa_avg_gap:.2f}%** |",
            f"| **Genetic Algorithm (GA)** | {len(ga_rows)} | 10 | {ga_avg_time:.4f} s | **+{ga_avg_gap:.2f}%** |",
            "",
            "### Một số bài toán tiêu biểu đại diện",
            "",
            "| testcase | n | cap | vehicle | optimal | algor | runs | avg_time | avg_gap |",
            "| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: |"
        ])

        sample_cases = ["A-n32-k5", "A-n45-k6", "A-n60-k9", "B-n35-k5", "B-n50-k7", "P-n23-k8", "P-n50-k10", "P-n76-k5"]
        for sc in sample_cases:
            for r in meta_results:
                if r["testcase"] == sc:
                    opt_str = str(r["optimal"]) if r["optimal"] is not None else "N/A"
                    lines.append(
                        f"| `{r['testcase']}` | {r['n']} | {r['cap']} | {r['vehicle']} | {opt_str} | **{r['algor']}** | {r['runs']} | {r['avg_time']:.4f} s | **{r['avg_gap']}** |"
                    )

        lines.extend([
            "",
            "*(Dữ liệu chi tiết toàn bộ 83 bài toán được lưu trữ tại `metaheuristic_results.csv`)*",
            "",
            "---",
            "",
            "## 3. Biểu đồ trực quan",
            "",
            "### Đồ thị hội tụ (Convergence Curves)",
            "![Đồ thị hội tụ](chart_meta_convergence.png)",
            "",
            "### So sánh hiệu năng tổng thể Metaheuristic (SA vs GA)",
            "![So sánh Metaheuristic](chart_meta_comparison.png)"
        ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_all_charts(exact_results: List[Dict[str, Any]], meta_results: List[Dict[str, Any]], out_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams["font.sans-serif"] = "DejaVu Sans"
        plt.rcParams["axes.edgecolor"] = "#cccccc"
        plt.rcParams["axes.linewidth"] = 0.8

        # Chart 1: Exact Comparison
        testcases = sorted(list(set(r["testcase"] for r in exact_results)))
        if testcases:
            bt_times, dp_times = [], []
            for tc in testcases:
                bt_row = next((r for r in exact_results if r["testcase"] == tc and r["algor"] == "backtracking"), None)
                dp_row = next((r for r in exact_results if r["testcase"] == tc and r["algor"] == "dynamic_programming"), None)
                bt_times.append(float(bt_row["time"]) if bt_row else 0)
                dp_times.append(float(dp_row["time"]) if dp_row else 0)

            fig, ax = plt.subplots(figsize=(8, 4.8))
            x = np.arange(len(testcases))
            w = 0.35

            rects1 = ax.bar(x - w / 2, bt_times, w, label="Backtracking", color="#d95f02", alpha=0.85)
            rects2 = ax.bar(x + w / 2, dp_times, w, label="Dynamic Programming", color="#2b5c8f", alpha=0.85)
            ax.set_title("So sánh thời gian thực thi thuật toán chính xác (N <= 20)", fontsize=12, pad=10, fontweight="bold")
            ax.set_xlabel("Bài toán kiểm thử", fontsize=11)
            ax.set_ylabel("Thời gian chạy (giây)", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(testcases, fontsize=10)
            ax.legend(loc="upper right", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5, axis="y")

            for rect in rects1:
                h = rect.get_height()
                ax.annotate(f"{h:.2f}s", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=9)
            for rect in rects2:
                h = rect.get_height()
                ax.annotate(f"{h:.2f}s", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=9)

            plt.tight_layout()
            plt.savefig(out_dir / "chart_exact_comparison.png", dpi=150)
            plt.close()

        # Chart 2: Convergence Curves
        cases = [
            ("A-n32-k5", "A-n32-k5 (N=32)"),
            ("P-n50-k10", "P-n50-k10 (N=50)"),
            ("E-n101-k14", "E-n101-k14 (N=101)")
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for idx, (c_name, title) in enumerate(cases):
            ax = axes[idx]
            vrp_path = Path("CVRPLIB") / f"{c_name}.vrp"
            if vrp_path.exists():
                inst = CVRPParser.parse_file(str(vrp_path))
                opt = get_optimal_value(vrp_path)
                res_sa = solve_sa(inst, iters_per_temp=30, cooling_rate=0.985, return_history=True, seed=42)
                res_ga = solve_ga(inst, pop_size=40, generations=100, return_history=True, seed=42)
                x_sa = np.linspace(0, 100, len(res_sa.get("history", [])))
                x_ga = np.linspace(0, 100, len(res_ga.get("history", [])))
                ax.plot(x_sa, res_sa.get("history", []), label="Simulated Annealing", color="#2b5c8f", linewidth=2.0)
                ax.plot(x_ga, res_ga.get("history", []), label="Genetic Algorithm", color="#d95f02", linewidth=2.0)
                if opt:
                    ax.axhline(opt, color="#2ca02c", linestyle="--", linewidth=1.6, label=f"Optimal ({opt})")
                ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
                ax.set_xlabel("Tiến trình tối ưu (%)", fontsize=10)
                ax.set_ylabel("Chi phí quãng đường", fontsize=10)
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(fontsize=9, loc="upper right")

        fig.suptitle("Đồ thị hội tụ (Convergence Curves) của 2 thuật toán Metaheuristic", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(out_dir / "chart_meta_convergence.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Chart 3: Metaheuristic Direct Comparison (GAP & Runtime)
        sa_rows = [r for r in meta_results if r["algor"] == "simulated_annealing"]
        ga_rows = [r for r in meta_results if r["algor"] == "genetic_algorithm"]

        def parse_gap(val):
            try:
                return float(str(val).replace("%", "").replace("+", "").strip())
            except Exception:
                return 0.0

        if sa_rows and ga_rows:
            sa_gaps = [parse_gap(r["avg_gap"]) for r in sa_rows]
            ga_gaps = [parse_gap(r["avg_gap"]) for r in ga_rows]
            sa_times = [float(r["avg_time"]) for r in sa_rows]
            ga_times = [float(r["avg_time"]) for r in ga_rows]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
            algs = ["Simulated Annealing", "Genetic Algorithm"]
            colors = ["#2b5c8f", "#d95f02"]

            # Subplot 1: Average GAP
            mean_sa_gap = np.mean(sa_gaps)
            mean_ga_gap = np.mean(ga_gaps)
            b1 = ax1.bar(algs, [mean_sa_gap, mean_ga_gap], color=colors, width=0.45, alpha=0.85)
            ax1.set_title("Sai số trung bình GAP (%) trên 83 bài toán", fontsize=12, fontweight="bold", pad=10)
            ax1.set_ylabel("Sai số GAP trung bình (%)", fontsize=10)
            ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
            for bar in b1:
                yval = bar.get_height()
                ax1.annotate(f"{yval:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, yval),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

            # Subplot 2: Average Time
            mean_sa_time = np.mean(sa_times)
            mean_ga_time = np.mean(ga_times)
            b2 = ax2.bar(algs, [mean_sa_time, mean_ga_time], color=colors, width=0.45, alpha=0.85)
            ax2.set_title("Thời gian chạy trung bình (giây / lượt)", fontsize=12, fontweight="bold", pad=10)
            ax2.set_ylabel("Thời gian trung bình (s)", fontsize=10)
            ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
            for bar in b2:
                yval = bar.get_height()
                ax2.annotate(f"{yval:.4f}s", xy=(bar.get_x() + bar.get_width() / 2, yval),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

            fig.suptitle("So sánh hiệu năng tổng thể 2 thuật toán Metaheuristic (83 bài toán, 10 runs)", fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            plt.savefig(out_dir / "chart_meta_comparison.png", dpi=150, bbox_inches="tight")
            plt.close()

    except Exception as e:
        print(f"Chart generation warning: {e}")


def main():
    parser = argparse.ArgumentParser(description="CVRP Automated Benchmark Pipeline (Threshold N=20)")
    parser.add_argument("--all", action="store_true", help="Run benchmark on all instances in CVRPLIB/")
    parser.add_argument("--instances", nargs="+", default=None, help="List of instance filenames or paths")
    parser.add_argument("--threshold-n", type=int, default=20, help="Threshold N (default: 20)")
    parser.add_argument("--meta-runs", type=int, default=10, help="Number of runs for metaheuristic algorithms (default: 10)")
    parser.add_argument("--time-limit-bt", type=float, default=5.0, help="Time limit for backtracking on small instances")
    parser.add_argument("--output-dir", default="experiments", help="Output directory for reports and data")

    args = parser.parse_args()

    data_dir = Path("CVRPLIB")
    if args.all or args.instances is None:
        instance_files = [str(f) for f in sorted(data_dir.glob("*.vrp"))]
    else:
        instance_files = []
        for item in args.instances:
            p = Path(item)
            if not p.is_file():
                alt = data_dir / item if item.endswith(".vrp") else data_dir / f"{item}.vrp"
                if alt.is_file():
                    p = alt
            instance_files.append(str(p))

    print("=== Starting CVRP Benchmark Pipeline (Threshold N=20) ===")
    print(f"Total instances: {len(instance_files)}")
    print(f"Partition rule: N <= {args.threshold_n} -> Exact Only | N > {args.threshold_n} -> Metaheuristic Only")
    print(f"Metaheuristic runs: {args.meta_runs}\n")

    t_start = time.time()
    result = run_experiment(
        instance_files=instance_files,
        threshold_n=args.threshold_n,
        meta_runs=args.meta_runs,
        time_limit_bt=args.time_limit_bt,
        output_dir=args.output_dir
    )
    total_time = round(time.time() - t_start, 2)

    print("\n=== Benchmark Completed Successfully ===")
    print(f"Total Runtime: {total_time}s")
    print(f"Exact records: {result['exact_count']}")
    print(f"Metaheuristic records: {result['meta_count']}")
    print(f"Markdown report: {result['output_files']['markdown']}")
    print(f"Exact CSV: {result['output_files']['exact_csv']}")
    print(f"Metaheuristic CSV: {result['output_files']['meta_csv']}")
    print(f"JSON: {result['output_files']['json']}")


if __name__ == "__main__":
    main()
