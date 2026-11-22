import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DB_PATH = Path(__file__).resolve().parent.parent / "cvrp.db"

DEFAULT_VEHICLES = [
    ("Xe máy", 30, 4),
    ("Xe tải nhỏ", 60, 3),
    ("Xe tải vừa", 100, 5),
    ("Xe tải lớn", 200, 4),
]

# Các bộ dữ liệu mẫu tọa độ thực tế tại Hà Nội (nằm trên các tuyến phố chính) phục vụ Demo
DEFAULT_CUSTOMER_DATASETS = [
    {
        "dataset_name": "Demo-Hanoi-N10",
        "capacity": 60,
        "num_trucks": 3,
        "depot": {"lat": 21.0045, "lng": 105.8431},  # ĐH Bách Khoa - Đại Cồ Việt
        "customers": [
            {"id": 2, "lat": 21.0074, "lng": 105.8473, "demand": 15},  # Bạch Mai
            {"id": 3, "lat": 21.0122, "lng": 105.8495, "demand": 18},  # Phố Huế
            {"id": 4, "lat": 21.0168, "lng": 105.8442, "demand": 12},  # Lê Duẩn
            {"id": 5, "lat": 21.0115, "lng": 105.8348, "demand": 20},  # Xã Đàn
            {"id": 6, "lat": 21.0056, "lng": 105.8295, "demand": 14},  # Chùa Bộc
            {"id": 7, "lat": 20.9985, "lng": 105.8325, "demand": 16},  # Trường Chinh
            {"id": 8, "lat": 20.9952, "lng": 105.8412, "demand": 15},  # Giải Phóng
            {"id": 9, "lat": 20.9978, "lng": 105.8515, "demand": 18},  # Minh Khai
            {"id": 10, "lat": 21.0028, "lng": 105.8560, "demand": 12}, # Kim Ngưu
        ],
    },
    {
        "dataset_name": "Demo-Hanoi-N22",
        "capacity": 100,
        "num_trucks": 5,
        "depot": {"lat": 21.0045, "lng": 105.8431},  # ĐH Bách Khoa - Đại Cồ Việt
        "customers": [
            {"id": 2, "lat": 21.0074, "lng": 105.8473, "demand": 18},
            {"id": 3, "lat": 21.0122, "lng": 105.8495, "demand": 22},
            {"id": 4, "lat": 21.0168, "lng": 105.8442, "demand": 15},
            {"id": 5, "lat": 21.0225, "lng": 105.8518, "demand": 20},
            {"id": 6, "lat": 21.0268, "lng": 105.8465, "demand": 16},
            {"id": 7, "lat": 21.0242, "lng": 105.8365, "demand": 24},
            {"id": 8, "lat": 21.0185, "lng": 105.8312, "demand": 19},
            {"id": 9, "lat": 21.0115, "lng": 105.8348, "demand": 17},
            {"id": 10, "lat": 21.0056, "lng": 105.8295, "demand": 21},
            {"id": 11, "lat": 21.0092, "lng": 105.8215, "demand": 14},
            {"id": 12, "lat": 21.0018, "lng": 105.8205, "demand": 23},
            {"id": 13, "lat": 20.9965, "lng": 105.8265, "demand": 18},
            {"id": 14, "lat": 20.9985, "lng": 105.8325, "demand": 15},
            {"id": 15, "lat": 20.9912, "lng": 105.8385, "demand": 20},
            {"id": 16, "lat": 20.9885, "lng": 105.8445, "demand": 16},
            {"id": 17, "lat": 20.9952, "lng": 105.8412, "demand": 19},
            {"id": 18, "lat": 20.9935, "lng": 105.8525, "demand": 22},
            {"id": 19, "lat": 20.9978, "lng": 105.8515, "demand": 14},
            {"id": 20, "lat": 21.0028, "lng": 105.8560, "demand": 18},
            {"id": 21, "lat": 21.0088, "lng": 105.8585, "demand": 20},
            {"id": 22, "lat": 21.0155, "lng": 105.8565, "demand": 17},
        ],
    },
]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                capacity INTEGER NOT NULL,
                default_count INTEGER NOT NULL DEFAULT 3
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT NOT NULL,
                node_id INTEGER NOT NULL,
                lat REAL NOT NULL,
                lng REAL NOT NULL,
                demand INTEGER NOT NULL,
                is_depot INTEGER NOT NULL DEFAULT 0,
                capacity INTEGER NOT NULL DEFAULT 100,
                num_trucks INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        # Ensure columns exist if an older cvrp.db was already created
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(customers)").fetchall()}
        if "capacity" not in cols:
            conn.execute("ALTER TABLE customers ADD COLUMN capacity INTEGER NOT NULL DEFAULT 100")
        if "num_trucks" not in cols:
            conn.execute("ALTER TABLE customers ADD COLUMN num_trucks INTEGER NOT NULL DEFAULT 0")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_name TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                capacity INTEGER NOT NULL,
                algorithm TEXT NOT NULL,
                total_distance REAL NOT NULL,
                num_vehicles INTEGER NOT NULL,
                fill_rate REAL NOT NULL,
                execution_time REAL NOT NULL,
                routes_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        for name, cap, cnt in DEFAULT_VEHICLES:
            conn.execute(
                "INSERT OR IGNORE INTO vehicles (name, capacity, default_count) VALUES (?, ?, ?)",
                (name, cap, cnt),
            )

        # Seed demo datasets if not already present
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for ds in DEFAULT_CUSTOMER_DATASETS:
            exists = conn.execute(
                "SELECT 1 FROM customers WHERE dataset_name = ? LIMIT 1",
                (ds["dataset_name"],),
            ).fetchone()
            if not exists:
                conn.execute(
                    """
                    INSERT INTO customers (
                        dataset_name, node_id, lat, lng, demand, is_depot, capacity, num_trucks, created_at
                    ) VALUES (?, 1, ?, ?, 0, 1, ?, ?, ?)
                    """,
                    (
                        ds["dataset_name"],
                        ds["depot"]["lat"],
                        ds["depot"]["lng"],
                        ds["capacity"],
                        ds["num_trucks"],
                        now,
                    ),
                )
                for c in ds["customers"]:
                    conn.execute(
                        """
                        INSERT INTO customers (
                            dataset_name, node_id, lat, lng, demand, is_depot, capacity, num_trucks, created_at
                        ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
                        """,
                        (
                            ds["dataset_name"],
                            c["id"],
                            c["lat"],
                            c["lng"],
                            c["demand"],
                            ds["capacity"],
                            ds["num_trucks"],
                            now,
                        ),
                    )
        conn.commit()


def list_vehicles() -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, capacity, default_count FROM vehicles ORDER BY capacity ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_vehicle(name: str, capacity: int, default_count: int = 3) -> Dict[str, Any]:
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO vehicles (name, capacity, default_count)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                capacity = excluded.capacity,
                default_count = excluded.default_count
            """,
            (name.strip(), max(1, capacity), max(1, default_count)),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, name, capacity, default_count FROM vehicles WHERE name = ?",
            (name.strip(),),
        ).fetchone()
    return dict(row)


def save_customer_dataset(
    dataset_name: str,
    depot: Dict[str, float],
    customers: List[Dict[str, Any]],
    capacity: int = 100,
    num_trucks: int = 0,
) -> Dict[str, Any]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_name = dataset_name.strip() or f"Map-{now}"
    with _get_conn() as conn:
        conn.execute("DELETE FROM customers WHERE dataset_name = ?", (clean_name,))
        conn.execute(
            """
            INSERT INTO customers (
                dataset_name, node_id, lat, lng, demand, is_depot, capacity, num_trucks, created_at
            ) VALUES (?, 1, ?, ?, 0, 1, ?, ?, ?)
            """,
            (clean_name, float(depot["lat"]), float(depot["lng"]), max(1, capacity), max(0, num_trucks), now),
        )
        for idx, c in enumerate(customers, start=2):
            conn.execute(
                """
                INSERT INTO customers (
                    dataset_name, node_id, lat, lng, demand, is_depot, capacity, num_trucks, created_at
                ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    clean_name,
                    int(c.get("id") or idx),
                    float(c["lat"]),
                    float(c["lng"]),
                    int(c.get("demand", 10)),
                    max(1, capacity),
                    max(0, num_trucks),
                    now,
                ),
            )
        conn.commit()
    return {"dataset_name": clean_name, "count": 1 + len(customers), "created_at": now}


def list_customer_datasets() -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT dataset_name AS name,
                   COUNT(*) AS dimension,
                   MAX(capacity) AS capacity,
                   MAX(num_trucks) AS num_trucks,
                   MAX(created_at) AS created_at
            FROM customers
            GROUP BY dataset_name
            ORDER BY MIN(id) ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_customer_dataset(dataset_name: str) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT node_id AS id, lat, lng, demand, is_depot, capacity, num_trucks
            FROM customers
            WHERE dataset_name = ?
            ORDER BY is_depot DESC, node_id ASC
            """,
            (dataset_name,),
        ).fetchall()
    if not rows:
        return None
    nodes = [
        {
            "id": r["id"],
            "lat": r["lat"],
            "lng": r["lng"],
            "demand": r["demand"],
            "is_depot": bool(r["is_depot"]),
        }
        for r in rows
    ]
    return {
        "name": dataset_name,
        "dimension": len(nodes),
        "capacity": rows[0]["capacity"],
        "num_trucks": rows[0]["num_trucks"],
        "map_nodes": nodes,
    }


def log_experiment(
    instance_name: str,
    dimension: int,
    capacity: int,
    algorithm: str,
    total_distance: float,
    num_vehicles: int,
    fill_rate: float,
    execution_time: float,
    routes: List[List[int]],
) -> int:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO experiments (
                instance_name, dimension, capacity, algorithm,
                total_distance, num_vehicles, fill_rate, execution_time,
                routes_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                instance_name or "Custom-Map",
                int(dimension),
                int(capacity),
                algorithm,
                float(total_distance),
                int(num_vehicles),
                float(fill_rate),
                float(execution_time),
                json.dumps(routes),
                now,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_experiments(limit: int = 50) -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, instance_name, dimension, capacity, algorithm,
                   total_distance, num_vehicles, fill_rate, execution_time,
                   routes_json, created_at
            FROM experiments
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(1, limit),),
        ).fetchall()
    results = []
    for r in rows:
        item = dict(r)
        item["routes"] = json.loads(item.pop("routes_json"))
        results.append(item)
    return results


def clear_experiments() -> int:
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM experiments")
        conn.commit()
        return cur.rowcount
