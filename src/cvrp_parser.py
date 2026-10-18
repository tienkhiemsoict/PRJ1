import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union
from pathlib import Path


@dataclass
class CVRPInstance:
    name: str = ""
    dimension: int = 0         # Tổng số nút (gồm kho và khách hàng)
    capacity: int = 0          # Sức chứa Q của mỗi xe
    depot_id: int = 1          # ID kho hàng (mặc định là 1)
    coords: Dict[int, Tuple[float, float]] = field(default_factory=dict) # {node_id: (x, y)}
    demands: Dict[int, int] = field(default_factory=dict)                 # {node_id: demand}
    distance_matrix: List[List[int]] = field(default_factory=list)        # Ma trận khoảng cách N x N

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "capacity": self.capacity,
            "depot_id": self.depot_id,
            "coords": self.coords,
            "demands": self.demands,
            "distance_matrix": self.distance_matrix
        }


class CVRPParser:

    @staticmethod
    def _calc_distance_matrix(coords: Dict[int, Tuple[float, float]], edge_type: str) -> List[List[int]]:
        node_ids = sorted(coords.keys())
        n = len(node_ids)
        matrix = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                id_i, id_j = node_ids[i], node_ids[j]
                x1, y1 = coords[id_i]
                x2, y2 = coords[id_j]

                dx, dy = x1 - x2, y1 - y2

                if edge_type == "ATT":
                    rij = math.sqrt((dx * dx + dy * dy) / 10.0)
                    tij = int(rij + 0.5)
                    matrix[i][j] = tij + 1 if tij < rij else tij
                else:  # Mặc định EUC_2D
                    matrix[i][j] = int(math.sqrt(dx * dx + dy * dy) + 0.5)

        return matrix

    @classmethod
    def parse_string(cls, content: str) -> CVRPInstance:
        """Parse nội dung dữ liệu từ dạng chuỗi (Phù hợp khi nhận file Upload từ Web)"""
        instance = CVRPInstance()
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        current_section = None
        edge_weight_type = "EUC_2D"

        for line in lines:
            if line == "EOF":
                break

            # 1. Đọc các thông tin Header
            if ":" in line and not current_section:
                parts = line.split(":", 1)
                key = parts[0].strip().upper()
                val = parts[1].strip()

                if key == "NAME":
                    instance.name = val
                elif key == "DIMENSION":
                    instance.dimension = int(val)
                elif key == "CAPACITY":
                    instance.capacity = int(val)
                elif key == "EDGE_WEIGHT_TYPE":
                    edge_weight_type = val
                continue

            # 2. Nhận diện các Section dữ liệu
            if line in ["NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"]:
                current_section = line
                continue

            # 3. Đọc dữ liệu chi tiết từng Section
            parts = line.split()
            if not parts:
                continue

            if current_section == "NODE_COORD_SECTION":
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                instance.coords[node_id] = (x, y)

            elif current_section == "DEMAND_SECTION":
                node_id = int(parts[0])
                demand = int(parts[1])
                instance.demands[node_id] = demand

            elif current_section == "DEPOT_SECTION":
                depot_id = int(parts[0])
                if depot_id != -1:
                    instance.depot_id = depot_id

        # 4. Tính toán ma trận khoảng cách
        instance.distance_matrix = cls._calc_distance_matrix(instance.coords, edge_weight_type)

        return instance

    @classmethod
    def parse_file(cls, filepath: Union[str, Path]) -> CVRPInstance:
        """Parse dữ liệu trực tiếp từ đường dẫn file trên máy"""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.parse_string(content)