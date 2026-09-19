# -*- coding: utf-8 -*-
"""
实际案例：列生成 + 分支定价求解代码
运行环境：Python 3.9+；依赖：pip install numpy scipy

模型说明：
1. 限制主问题 RMP 使用 scipy.optimize.linprog 求解 LP 松弛；
2. 定价子问题使用标签算法搜索负折减成本路径；
3. 分支定价优先处理班列开行频率 x_r，再处理 z_w、y_wp；
4. 所有算例数据和模型参数均从 --data-dir 指定的外部 CSV 读取；
5. 约束(3)采用“班列最小开行规模约束”：
   Qmin_r * x_r <= sum_w sum_p theta_pr * y_wp
   其中 theta_pr=1 表示路径 p 使用班列 r，否则为 0。

"""

from __future__ import annotations

import math
import argparse
import csv
from pathlib import Path as FilePath
import time
import heapq
import itertools
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import lil_matrix


# =========================
# 一、全局参数
# =========================
EPS = 1e-4          # 列生成负折减成本判断容差
INT_EPS = 1e-6      # 整数性判断容差
INF = 1e100         # 足够大的数


# =========================
# 二、数据结构
# =========================
@dataclass(frozen=True)
class Edge:
    """网络弧段。kind 为 road 或 rail；rail 弧记录所属班列 service。"""
    u: int
    v: int
    kind: str
    service: Optional[str]
    dist: float
    sections: Tuple[Tuple[int, int], ...] = tuple()

    @property
    def state(self):
        # 运输状态用于判断是否换装：公路、公路不换装；同一班列连续运行不换装
        return ("road", None) if self.kind == "road" else ("rail", self.service)


@dataclass
class Path:
    """路径列。包含路径成本、是否使用班列区段、是否在节点换装等信息。"""
    od: str
    edges: Tuple[Edge, ...]
    trans_nodes: Tuple[int, ...]
    nodes: Tuple[int, ...]
    d_road: float
    d_rail: float
    delta: Dict[Tuple[str, int, int], int]     # delta[(r,u,v)] = 1 表示路径用班列 r 的区段 u-v
    lam: Dict[int, int]                        # lam[s] = 1 表示路径在节点 s 换装
    services_used: Tuple[str, ...]             # 路径使用过的班列集合
    cost: float                                # 单位路径广义成本
    signature: Tuple[Tuple[int, int, str, str], ...]  # 路径唯一标识，避免重复列


@dataclass
class InstanceData:
    """算例数据。"""
    nodes: List[int]
    hubs: List[int]
    od_data: Dict[str, Dict[str, float]]
    road_dist: Dict[Tuple[int, int], float]
    rail_dist: Dict[Tuple[int, int], float]
    services: Dict[str, List[int]]
    city_names: Dict[int, str]

    # 成本和排放参数
    b_road: float
    b_rail: float
    e_road: float
    e_rail: float
    gamma: float
    tau: float
    alpha: float

    # 班列与网络能力参数
    train_capacity: float
    min_loading: float
    max_frequency: int
    rail_section_capacity: int
    hub_capacity: float
    train_fixed_cost: float

    # 由 build_edges 自动生成
    edges: List[Edge] = field(default_factory=list)
    outgoing: Dict[int, List[Edge]] = field(default_factory=dict)
    service_sections: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    physical_rail_sections: List[Tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class BranchCut:
    """分支约束。"""
    var_type: str      # x、z 或 y
    key: Any           # 变量索引
    sense: str         # <= 或 >=
    rhs: float


@dataclass
class BPNode:
    """分支定价树节点。"""
    node_id: int
    depth: int
    cuts: List[BranchCut]
    paths: Dict[str, List[Path]]
    lb: float = -INF


@dataclass
class DualPack:
    """RMP 对偶变量包。"""
    pi: Dict[str, float]                         # 需求平衡约束
    cap: Dict[Tuple[str, int, int], float]        # 班列区段容量约束
    minscale: Dict[str, float]                    # 班列最小开行规模约束
    hub: Dict[int, float]                         # 节点换装能力约束
    branch_y: Dict[Tuple[str, Tuple[Tuple[int, int, str, str], ...]], float] = field(default_factory=dict)


@dataclass
class RMPResult:
    """限制主问题求解结果。"""
    status: str
    obj: float = INF
    solution: Optional[Dict[str, Any]] = None
    duals: Optional[DualPack] = None


@dataclass
class Label:
    """定价子问题中的标签。"""
    cost: float
    node: int
    state: Optional[Tuple[str, Optional[str]]]
    visited: frozenset
    edges: Tuple[Edge, ...]
    trans_nodes: Tuple[int, ...]
    used_services: frozenset
    road_arc_count: int
    has_rail: bool


# =========================
# 三、基础函数
# =========================
def canonical_arc(i: int, j: int) -> Tuple[int, int]:
    """距离表按无向弧存储，统一为较小编号在前。"""
    return (i, j) if i < j else (j, i)


def is_integer_value(x: float) -> bool:
    """判断变量是否可视为整数。"""
    return abs(x - round(x)) <= INT_EPS


def frac_part(x: float) -> float:
    """变量的小数部分。"""
    return x - math.floor(x)


def trans_required(prev_state, next_state) -> bool:
    """判断是否发生换装。"""
    if prev_state is None:
        return False
    if prev_state == next_state:
        return False
    if prev_state[0] == "road" and next_state[0] == "road":
        return False
    return True


def trans_cost(data: InstanceData) -> float:
    """单位换装广义成本。"""
    return data.gamma + data.alpha * data.tau


def road_cost(data: InstanceData, dist: float) -> float:
    """单位 TEU 在给定公路距离上的广义成本。"""
    return dist * (data.b_road + data.alpha * data.e_road)


def rail_cost(data: InstanceData, dist: float) -> float:
    """单位 TEU 在给定铁路距离上的广义成本。"""
    return dist * (data.b_rail + data.alpha * data.e_rail)


def direct_road_cost(data: InstanceData, od: str) -> float:
    """某 OD 的全公路直运单位广义成本。"""
    return road_cost(data, data.od_data[od]["road_direct"])


# =========================
# 四、构建案例数据
# =========================
def _csv_rows(directory, name, required):
    filename = FilePath(directory) / name
    with filename.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{name}: required columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{name}: no data rows")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{name}: malformed CSV row")
    return rows


def _number(value, label, integer=False, positive=False):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a number") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{label}: invalid nonnegative/positive number")
    if integer:
        if not result.is_integer():
            raise ValueError(f"{label}: expected an integer")
        return int(result)
    return result


def _node_id(value):
    return _number(value, "node ID", integer=True, positive=True)


def build_case(data_dir) -> InstanceData:
    """Load case data exclusively from an explicitly supplied CSV directory."""
    nodes, hubs, city_names = [], [], {}
    for row in _csv_rows(data_dir, "nodes.csv", {"node", "is_hub"}):
        node = _node_id(row["node"])
        if node in city_names:
            raise ValueError(f"nodes.csv: duplicate node {node}")
        hub = row["is_hub"].strip()
        if hub not in {"0", "1"}:
            raise ValueError("nodes.csv: is_hub must be 0 or 1")
        nodes.append(node)
        city_names[node] = row.get("name", "").strip() or str(node)
        if hub == "1":
            hubs.append(node)

    od_data = {}
    for row in _csv_rows(data_dir, "od.csv", {"od", "origin", "destination", "demand_teu", "direct_road_km"}):
        od = row["od"].strip()
        if not od or od in od_data:
            raise ValueError("od.csv: empty or duplicate OD identifier")
        origin, dest = _node_id(row["origin"]), _node_id(row["destination"])
        if origin not in city_names or dest not in city_names or origin == dest:
            raise ValueError(f"od.csv: invalid endpoints for {od}")
        od_data[od] = {
            "origin": origin, "dest": dest,
            "demand": _number(row["demand_teu"], "demand", integer=True),
            "road_direct": _number(row["direct_road_km"], "direct road distance", positive=True),
        }

    road_dist, rail_dist = {}, {}
    seen = set()
    for row in _csv_rows(data_dir, "distances.csv", {"origin", "destination", "road_km", "rail_km"}):
        u, v = _node_id(row["origin"]), _node_id(row["destination"])
        if u not in city_names or v not in city_names or u == v:
            raise ValueError("distances.csv: invalid endpoints")
        arc = canonical_arc(u, v)
        if arc in seen:
            raise ValueError(f"distances.csv: duplicate undirected pair {arc}")
        seen.add(arc)
        if not row["road_km"].strip() and not row["rail_km"].strip():
            raise ValueError(f"distances.csv: both mode distances missing for {arc}")
        for name, target in [("road_km", road_dist), ("rail_km", rail_dist)]:
            if row[name].strip():
                target[arc] = _number(row[name], name, positive=True)

    services = {}
    for row in _csv_rows(data_dir, "services.csv", {"service", "node_sequence"}):
        service = row["service"].strip()
        if len(service) < 2 or not service[1:].isdigit() or service in services:
            raise ValueError("services.csv: unique service IDs must have a one-character prefix and integer suffix")
        seq = [_node_id(v.strip()) for v in row["node_sequence"].split(",")]
        if len(seq) < 2 or len(set(seq)) != len(seq) or not set(seq).issubset(city_names):
            raise ValueError(f"services.csv: invalid node sequence for {service}")
        for u, v in zip(seq, seq[1:]):
            if canonical_arc(u, v) not in rail_dist:
                raise ValueError(f"services.csv: missing rail distance for {service}, {u}-{v}")
        services[service] = seq

    required = {"b_road", "b_rail", "e_road", "e_rail", "gamma", "tau", "alpha",
                "train_capacity", "min_loading", "max_frequency", "rail_section_capacity",
                "hub_capacity", "train_fixed_cost"}
    params = {}
    for row in _csv_rows(data_dir, "parameters.csv", {"parameter", "value"}):
        key = row["parameter"].strip()
        if key not in required or key in params:
            raise ValueError(f"parameters.csv: unknown or duplicate parameter {key}")
        params[key] = _number(row["value"], key,
                              integer=key in {"max_frequency", "rail_section_capacity"},
                              positive=key == "train_capacity")
    if set(params) != required:
        raise ValueError(f"parameters.csv: missing {sorted(required - set(params))}")
    if params["min_loading"] > params["train_capacity"]:
        raise ValueError("parameters.csv: minimum loading exceeds train capacity")
    data = InstanceData(nodes=nodes, hubs=hubs, city_names=city_names,
                        od_data=od_data, road_dist=road_dist, rail_dist=rail_dist,
                        services=services, **params)
    build_edges(data)
    return data




# =========================
# 五、生成公路弧和铁路服务弧
# =========================
def build_edges(data: InstanceData) -> None:
    """
    构建网络弧段。
    公路弧按双向处理；铁路班列按服务方向处理；
    同一班列允许从线路中任一节点上车，并在其下游任一节点下车。
    """
    outgoing = {i: [] for i in data.nodes}
    edges: List[Edge] = []
    keys = set()

    # 公路弧：双向
    for (i, j), d in data.road_dist.items():
        for u, v in [(i, j), (j, i)]:
            key = (u, v, "road", "")
            if key not in keys:
                e = Edge(u=u, v=v, kind="road", service=None, dist=d)
                edges.append(e)
                outgoing[u].append(e)
                keys.add(key)

    # 铁路服务弧：同一班列可跨多个物理区段形成一条服务弧
    service_sections: Dict[str, List[Tuple[int, int]]] = {}
    physical_sections = set()

    for r, seq in data.services.items():
        service_sections[r] = []
        section_dists: List[float] = []

        # 逐段记录班列经过的物理区段
        for u, v in zip(seq[:-1], seq[1:]):
            arc = canonical_arc(u, v)
            if arc not in data.rail_dist:
                raise ValueError(f"班列 {r} 的铁路弧段 {(u, v)} 缺少铁路距离。")
            d = data.rail_dist[arc]
            service_sections[r].append((u, v))
            section_dists.append(d)
            physical_sections.add((u, v))

        # 生成同一班列的跨区段服务弧
        for a in range(len(seq) - 1):
            dist_sum = 0.0
            sections = []
            for b in range(a + 1, len(seq)):
                dist_sum += section_dists[b - 1]
                sections.append((seq[b - 1], seq[b]))

                u, v = seq[a], seq[b]
                key = (u, v, "rail", r)
                if key not in keys:
                    e = Edge(
                        u=u,
                        v=v,
                        kind="rail",
                        service=r,
                        dist=dist_sum,
                        sections=tuple(sections),
                    )
                    edges.append(e)
                    outgoing[u].append(e)
                    keys.add(key)

    data.edges = edges
    data.outgoing = outgoing
    data.service_sections = service_sections
    data.physical_rail_sections = sorted(physical_sections)


# =========================
# 六、路径生成与路径成本
# =========================
def make_path(data: InstanceData, od: str, edges: Tuple[Edge, ...], trans_nodes: Tuple[int, ...]) -> Path:
    """根据弧段序列生成路径列。"""
    nodes = [edges[0].u] if edges else []
    for e in edges:
        nodes.append(e.v)

    d_road = sum(e.dist for e in edges if e.kind == "road")
    d_rail = sum(e.dist for e in edges if e.kind == "rail")

    delta: Dict[Tuple[str, int, int], int] = {}
    services_used: List[str] = []

    for e in edges:
        if e.kind == "rail":
            for u, v in e.sections:
                delta[(e.service, u, v)] = 1
            if e.service not in services_used:
                services_used.append(e.service)

    lam = {s: 1 for s in trans_nodes}

    cost = road_cost(data, d_road) + rail_cost(data, d_rail) + len(lam) * trans_cost(data)
    signature = tuple((e.u, e.v, e.kind, e.service or "") for e in edges)

    return Path(
        od=od,
        edges=edges,
        trans_nodes=trans_nodes,
        nodes=tuple(nodes),
        d_road=d_road,
        d_rail=d_rail,
        delta=delta,
        lam=lam,
        services_used=tuple(services_used),
        cost=cost,
        signature=signature,
    )


# =========================
# 七、定价子问题：标签算法
# =========================
def label_dominates(a: Label, b: Label) -> bool:
    """标签支配规则，用于减少标签数量。"""
    return (
        a.node == b.node
        and a.state == b.state
        and a.has_rail == b.has_rail
        and a.used_services == b.used_services
        and a.road_arc_count <= b.road_arc_count
        and a.cost <= b.cost + EPS
    )


def pricing_label_algorithm(
    data: InstanceData,
    od: str,
    duals: DualPack,
    forbidden: Set[Tuple[Tuple[int, int, str, str], ...]],
    branch_y_dual: Optional[Dict[Tuple[str, Tuple[Tuple[int, int, str, str], ...]], float]] = None,
    max_labels_per_state: int = 120,
    max_pops: int = 300000,
    max_rail_services: int = 4,
    max_road_arcs: int = 3,
) -> Tuple[Optional[Path], float]:
    """
    定价子问题。
    在当前对偶变量下，为 OD 搜索负折减成本路径。
    """
    origin = int(data.od_data[od]["origin"])
    dest = int(data.od_data[od]["dest"])

    start = Label(
        cost=0.0,
        node=origin,
        state=None,
        visited=frozenset([origin]),
        edges=tuple(),
        trans_nodes=tuple(),
        used_services=frozenset(),
        road_arc_count=0,
        has_rail=False,
    )

    pq = []
    counter = itertools.count()
    heapq.heappush(pq, (0.0, next(counter), start))

    labels_by_state: Dict[Any, List[Label]] = {}
    best_path = None
    best_rc = INF
    branch_y_dual = branch_y_dual or {}
    pops = 0

    while pq:
        pops += 1
        if pops > max_pops:
            break

        _, _, lab = heapq.heappop(pq)

        # 到达终点后形成候选路径
        if lab.node == dest:
            if lab.has_rail:
                p = make_path(data, od, lab.edges, lab.trans_nodes)
                if p.signature not in forbidden:
                    # Branch rows on path-flow variables contribute to the
                    # reduced cost of a newly generated column.  Frequency
                    # and road-only-flow branches have no path coefficient.
                    rc = (
                        lab.cost
                        - duals.pi.get(od, 0.0)
                        - branch_y_dual.get((od, p.signature), 0.0)
                    )
                    if rc < best_rc:
                        best_rc = rc
                        best_path = p
            continue

        for e in data.outgoing.get(lab.node, []):
            if e.v in lab.visited:
                continue

            # 控制公路接续弧数量，避免无意义绕行
            new_road_count = lab.road_arc_count + (1 if e.kind == "road" else 0)
            if new_road_count > max_road_arcs:
                continue

            new_used_services = lab.used_services
            first_use_service = False

            if e.kind == "rail" and e.service not in new_used_services:
                first_use_service = True
                new_used_services = frozenset(set(new_used_services) | {e.service})

            if len(new_used_services) > max_rail_services:
                continue

            add_cost = 0.0
            new_trans_nodes = lab.trans_nodes

            # 换装成本及换装能力约束对偶修正
            if trans_required(lab.state, e.state):
                if lab.node not in data.hubs:
                    continue
                add_cost += trans_cost(data) - duals.hub.get(lab.node, 0.0)
                new_trans_nodes = lab.trans_nodes + (lab.node,)

            if e.kind == "road":
                add_cost += road_cost(data, e.dist)
            else:
                add_cost += rail_cost(data, e.dist)

                # 班列区段容量约束对偶修正
                for u, v in e.sections:
                    add_cost -= duals.cap.get((e.service, u, v), 0.0)

                # 班列最小开行规模约束对偶修正：每条路径对每个班列只计一次
                if first_use_service:
                    add_cost += duals.minscale.get(e.service, 0.0)

            new_label = Label(
                cost=lab.cost + add_cost,
                node=e.v,
                state=e.state,
                visited=frozenset(set(lab.visited) | {e.v}),
                edges=lab.edges + (e,),
                trans_nodes=new_trans_nodes,
                used_services=new_used_services,
                road_arc_count=new_road_count,
                has_rail=lab.has_rail or e.kind == "rail",
            )

            state_key = (
                new_label.node,
                new_label.state,
                new_label.has_rail,
                new_label.used_services,
                new_label.road_arc_count,
            )
            bucket = labels_by_state.setdefault(state_key, [])

            if any(label_dominates(old, new_label) for old in bucket):
                continue

            bucket[:] = [old for old in bucket if not label_dominates(new_label, old)]

            if len(bucket) >= max_labels_per_state:
                worst_idx = max(range(len(bucket)), key=lambda k: bucket[k].cost)
                if bucket[worst_idx].cost <= new_label.cost:
                    continue
                bucket.pop(worst_idx)

            bucket.append(new_label)
            heapq.heappush(pq, (new_label.cost, next(counter), new_label))

    return best_path, best_rc


# =========================
# 八、初始路径集合
# =========================
def dummy_duals_for_initial_paths(data: InstanceData) -> DualPack:
    """用全公路直运单位成本构造虚拟对偶变量，生成初始联运路径。"""
    return DualPack(
        pi={od: direct_road_cost(data, od) for od in data.od_data},
        cap={},
        minscale={},
        hub={s: 0.0 for s in data.hubs},
    )


def initial_path_pool(data: InstanceData, k_each_od: int = 8) -> Dict[str, List[Path]]:
    """
    为每个 OD 生成若干条初始联运路径。
    即便没有初始联运路径，RMP 中也有全公路 z_w 变量保证可行。
    """
    paths = {od: [] for od in data.od_data}
    duals = dummy_duals_for_initial_paths(data)

    for od in data.od_data:
        forbidden = set()
        for _ in range(k_each_od):
            p, _ = pricing_label_algorithm(data, od, duals, forbidden)
            if p is None:
                break
            paths[od].append(p)
            forbidden.add(p.signature)

    return paths


def enumerate_full_paths(
    data: InstanceData,
    od: str,
    max_road_arcs: int = 3,
    max_rail_services: int = 4,
) -> List[Path]:
    """Enumerate every simple feasible path under the pricing bounds.

    The enumeration uses exactly the same state-transition, no-cycle, road-arc,
    and distinct-service rules as ``pricing_label_algorithm``.  It is intended
    only for the full-path MILP benchmark; the branch-and-price method avoids
    materializing this pool.
    """
    origin = int(data.od_data[od]["origin"])
    dest = int(data.od_data[od]["dest"])
    paths: List[Path] = []

    def dfs(
        node: int,
        state: Optional[Tuple[str, Optional[str]]],
        visited: frozenset,
        edges: Tuple[Edge, ...],
        trans_nodes: Tuple[int, ...],
        used_services: frozenset,
        road_arc_count: int,
        has_rail: bool,
    ) -> None:
        if node == dest:
            if has_rail:
                paths.append(make_path(data, od, edges, trans_nodes))
            return
        for edge in data.outgoing.get(node, []):
            if edge.v in visited:
                continue
            next_road_count = road_arc_count + (1 if edge.kind == "road" else 0)
            if next_road_count > max_road_arcs:
                continue
            next_services = used_services
            if edge.kind == "rail":
                next_services = frozenset(set(used_services) | {edge.service})
                if len(next_services) > max_rail_services:
                    continue
            next_trans_nodes = trans_nodes
            if trans_required(state, edge.state):
                if node not in data.hubs:
                    continue
                next_trans_nodes = trans_nodes + (node,)
            dfs(
                edge.v,
                edge.state,
                frozenset(set(visited) | {edge.v}),
                edges + (edge,),
                next_trans_nodes,
                next_services,
                next_road_count,
                has_rail or edge.kind == "rail",
            )

    dfs(origin, None, frozenset([origin]), tuple(), tuple(), frozenset(), 0, False)
    return paths


def enumerate_full_path_pool(data: InstanceData) -> Dict[str, List[Path]]:
    """Enumerate the complete candidate path pool for every OD pair."""
    return {od: enumerate_full_paths(data, od) for od in data.od_data}


def solve_full_path_milp(
    data: InstanceData,
    paths: Dict[str, List[Path]],
    time_limit: float = 60.0,
    mip_rel_gap: float = 1e-4,
) -> Dict[str, Any]:
    """Solve the compact full-path MILP with the paper's original constraints.

    This benchmark has the same variables and constraints as ``build_lp`` but
    materializes every feasible path.  Solver status and bounds are returned
    even when the time limit is reached, so a time-limited run is not reported
    as an exact optimum.
    """
    var_names: List[Tuple[Any, ...]] = []
    idx: Dict[Tuple[Any, ...], int] = {}
    for r in data.services:
        idx[("x", r)] = len(var_names)
        var_names.append(("x", r))
    for od in data.od_data:
        idx[("z", od)] = len(var_names)
        var_names.append(("z", od))
    for od, plist in paths.items():
        for p in plist:
            idx[("y", od, p.signature)] = len(var_names)
            var_names.append(("y", od, p.signature))

    n = len(var_names)
    objective = np.zeros(n)
    lower = np.zeros(n)
    upper = np.full(n, np.inf)
    integrality = np.zeros(n, dtype=int)
    for r in data.services:
        col = idx[("x", r)]
        objective[col] = data.train_fixed_cost
        upper[col] = data.max_frequency
        integrality[col] = 1
    for od in data.od_data:
        col = idx[("z", od)]
        objective[col] = direct_road_cost(data, od)
        upper[col] = data.od_data[od]["demand"]
    for od, plist in paths.items():
        for p in plist:
            objective[idx[("y", od, p.signature)]] = p.cost

    row_specs: List[Tuple[List[Tuple[int, float]], float, float]] = []
    for od in data.od_data:
        terms = [(idx[("z", od)], 1.0)]
        terms.extend((idx[("y", od, p.signature)], 1.0) for p in paths[od])
        row_specs.append((terms, data.od_data[od]["demand"], data.od_data[od]["demand"]))
    for r, sec_list in data.service_sections.items():
        for u, v in sec_list:
            terms = [(idx[("x", r)], -data.train_capacity)]
            for od, plist in paths.items():
                terms.extend(
                    (idx[("y", od, p.signature)], 1.0)
                    for p in plist
                    if p.delta.get((r, u, v), 0)
                )
            row_specs.append((terms, -np.inf, 0.0))
    for r in data.services:
        terms = [(idx[("x", r)], data.min_loading)]
        for od, plist in paths.items():
            terms.extend(
                (idx[("y", od, p.signature)], -1.0)
                for p in plist
                if r in p.services_used
            )
        row_specs.append((terms, -np.inf, 0.0))
    for u, v in data.physical_rail_sections:
        terms = [
            (idx[("x", r)], 1.0)
            for r, sec_list in data.service_sections.items()
            if (u, v) in sec_list
        ]
        row_specs.append((terms, -np.inf, data.rail_section_capacity))
    for s in data.hubs:
        terms = []
        for od, plist in paths.items():
            terms.extend(
                (idx[("y", od, p.signature)], 1.0)
                for p in plist
                if p.lam.get(s, 0)
            )
        row_specs.append((terms, -np.inf, data.hub_capacity))

    matrix = lil_matrix((len(row_specs), n), dtype=float)
    row_lower = np.full(len(row_specs), -np.inf)
    row_upper = np.full(len(row_specs), np.inf)
    for i, (terms, lo, hi) in enumerate(row_specs):
        for col, value in terms:
            matrix[i, col] += value
        row_lower[i] = lo
        row_upper[i] = hi

    started = time.time()
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix.tocsr(), row_lower, row_upper),
        options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "presolve": True},
    )
    runtime = time.time() - started
    return {
        "status": int(result.status),
        "success": bool(result.success),
        "message": str(result.message),
        "objective": None if result.fun is None else float(result.fun),
        "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        "mip_node_count": None if getattr(result, "mip_node_count", None) is None else int(result.mip_node_count),
        "mip_dual_bound": None if getattr(result, "mip_dual_bound", None) is None else float(result.mip_dual_bound),
        "runtime_seconds": runtime,
        "solver": "scipy.optimize.milp (HiGHS)",
        "time_limit": time_limit,
        "mip_rel_gap": mip_rel_gap,
        "model_size": {
            "variables": n,
            "binary_or_integer_variables": int(integrality.sum()),
            "continuous_variables": int(n - integrality.sum()),
            "constraints": len(row_specs),
            "nonzero_coefficients": int(matrix.nnz),
        },
        "path_counts": {od: len(plist) for od, plist in paths.items()},
        "total_paths": sum(len(plist) for plist in paths.values()),
    }


# =========================
# 九、分支定价求解器
# =========================
class BranchAndPriceSolver:
    """列生成 + 分支定价主求解器。"""

    def __init__(
        self,
        data: InstanceData,
        cg_eps: float = 1e-4,
        gap_tol: float = 1e-4,
        time_limit: float = 600.0,
        max_nodes: int = 10000,
        max_cg_iter: int = 200,
    ):
        self.data = data
        self.cg_eps = cg_eps
        self.gap_tol = gap_tol
        self.time_limit = time_limit
        self.max_nodes = max_nodes
        self.max_cg_iter = max_cg_iter
        self.node_counter = itertools.count()
        self.nodes_explored = 0
        self.columns_generated = 0
        self.final_lower_bound = -INF
        self.start_time = None

    def all_road_upper_bound(self) -> float:
        """全公路直运方案目标值，用作初始上界。"""
        return sum(
            self.data.od_data[od]["demand"] * direct_road_cost(self.data, od)
            for od in self.data.od_data
        )

    def all_road_solution(self, paths: Dict[str, List[Path]]) -> Dict[str, Any]:
        """构造全公路直运初始解。"""
        return {
            "obj": self.all_road_upper_bound(),
            "x": {r: 0.0 for r in self.data.services},
            "z": {od: self.data.od_data[od]["demand"] for od in self.data.od_data},
            "y": {},
            "paths": {od: list(paths[od]) for od in paths},
            "status": "initial_all_road",
        }

    def build_lp(self, paths: Dict[str, List[Path]], cuts: List[BranchCut]):
        """构造 RMP 的 LP 矩阵。"""
        data = self.data
        var_names = []
        idx = {}

        # x_r：班列开行频次
        for r in data.services:
            idx[("x", r)] = len(var_names)
            var_names.append(("x", r))

        # z_w：全公路直运货流
        for od in data.od_data:
            idx[("z", od)] = len(var_names)
            var_names.append(("z", od))

        # y_wp：路径货流
        path_map = {}
        for od, plist in paths.items():
            for p in plist:
                key = ("y", od, p.signature)
                if key not in idx:
                    idx[key] = len(var_names)
                    var_names.append(key)
                    path_map[(od, p.signature)] = p

        n = len(var_names)
        c = np.zeros(n)

        # 目标函数：固定开行成本 + 全公路成本 + 路径运输/换装/碳成本
        for r in data.services:
            c[idx[("x", r)]] = data.train_fixed_cost
        for od in data.od_data:
            c[idx[("z", od)]] = direct_road_cost(data, od)
        for (od, sig), p in path_map.items():
            c[idx[("y", od, sig)]] = p.cost

        Aeq, beq, eq_rows = [], [], []

        # 需求平衡约束：sum_p y_wp + z_w = q_w
        for od in data.od_data:
            row = np.zeros(n)
            row[idx[("z", od)]] = 1.0
            for p in paths[od]:
                row[idx[("y", od, p.signature)]] = 1.0
            Aeq.append(row)
            beq.append(data.od_data[od]["demand"])
            eq_rows.append(("balance", od))

        Aub, bub, ub_rows = [], [], []

        # 班列区段容量约束：sum y_wp * delta <= Q_r * x_r
        for r, sec_list in data.service_sections.items():
            for u, v in sec_list:
                row = np.zeros(n)
                row[idx[("x", r)]] = -data.train_capacity
                for od, plist in paths.items():
                    for p in plist:
                        if p.delta.get((r, u, v), 0):
                            row[idx[("y", od, p.signature)]] += 1.0
                Aub.append(row)
                bub.append(0.0)
                ub_rows.append(("cap", (r, u, v)))

        # 班列最小开行规模约束：Qmin*x_r - sum theta_pr*y_wp <= 0
        for r in data.services:
            row = np.zeros(n)
            row[idx[("x", r)]] = data.min_loading
            for od, plist in paths.items():
                for p in plist:
                    if r in p.services_used:
                        row[idx[("y", od, p.signature)]] -= 1.0
            Aub.append(row)
            bub.append(0.0)
            ub_rows.append(("minscale", r))

        # 铁路物理区段通过能力：sum beta*x_r <= U
        for u, v in data.physical_rail_sections:
            row = np.zeros(n)
            for r, sec_list in data.service_sections.items():
                if (u, v) in sec_list:
                    row[idx[("x", r)]] += 1.0
            Aub.append(row)
            bub.append(data.rail_section_capacity)
            ub_rows.append(("railsec", (u, v)))

        # 班列最大开行频次：x_r <= fmax
        for r in data.services:
            row = np.zeros(n)
            row[idx[("x", r)]] = 1.0
            Aub.append(row)
            bub.append(data.max_frequency)
            ub_rows.append(("xmax", r))

        # 节点换装能力：sum lambda*y_wp <= n_s
        for s in data.hubs:
            row = np.zeros(n)
            for od, plist in paths.items():
                for p in plist:
                    if p.lam.get(s, 0):
                        row[idx[("y", od, p.signature)]] += 1.0
            Aub.append(row)
            bub.append(data.hub_capacity)
            ub_rows.append(("hub", s))

        # 分支约束
        for cut in cuts:
            row = np.zeros(n)
            if cut.var_type == "x":
                key = ("x", cut.key)
            elif cut.var_type == "z":
                key = ("z", cut.key)
            elif cut.var_type == "y":
                od, sig = cut.key
                key = ("y", od, sig)
            else:
                raise ValueError(f"未知分支变量类型：{cut.var_type}")

            if key in idx:
                if cut.sense == "<=":
                    row[idx[key]] = 1.0
                    rhs = cut.rhs
                else:
                    row[idx[key]] = -1.0
                    rhs = -cut.rhs
            else:
                rhs = cut.rhs if cut.sense == "<=" else -cut.rhs

            Aub.append(row)
            bub.append(rhs)
            ub_rows.append(("branch", cut))

        bounds = [(0, None)] * n

        return (
            c,
            np.array(Aub),
            np.array(bub),
            np.array(Aeq),
            np.array(beq),
            bounds,
            idx,
            path_map,
            eq_rows,
            ub_rows,
        )

    def solve_rmp_lp(self, paths: Dict[str, List[Path]], cuts: List[BranchCut]) -> RMPResult:
        """求解当前 RMP LP 松弛并提取对偶变量。"""
        c, Aub, bub, Aeq, beq, bounds, idx, path_map, eq_rows, ub_rows = self.build_lp(paths, cuts)

        res = linprog(
            c,
            A_ub=Aub,
            b_ub=bub,
            A_eq=Aeq,
            b_eq=beq,
            bounds=bounds,
            method="highs",
        )

        if not res.success:
            return RMPResult(status="infeasible")

        x_val = {r: res.x[idx[("x", r)]] for r in self.data.services}
        z_val = {od: res.x[idx[("z", od)]] for od in self.data.od_data}
        y_val = {(od, sig): res.x[idx[("y", od, sig)]] for (od, sig), p in path_map.items()}

        # 等式约束对偶
        pi = {}
        for i, (typ, key) in enumerate(eq_rows):
            if typ == "balance":
                pi[key] = res.eqlin.marginals[i]

        # 不等式约束对偶
        cap, minscale, hub, branch_y = {}, {}, {}, {}
        for i, (typ, key) in enumerate(ub_rows):
            marginal = res.ineqlin.marginals[i]
            if typ == "cap":
                cap[key] = marginal
            elif typ == "minscale":
                minscale[key] = marginal
            elif typ == "hub":
                hub[key] = marginal
            elif typ == "branch" and key.var_type == "y":
                # build_lp represents >= rows as -y <= -rhs
                coefficient = 1.0 if key.sense == "<=" else -1.0
                branch_y[key.key] = branch_y.get(key.key, 0.0) + coefficient * marginal

        duals = DualPack(pi=pi, cap=cap, minscale=minscale, hub=hub, branch_y=branch_y)

        solution = {
            "obj": float(res.fun),
            "x": x_val,
            "z": z_val,
            "y": y_val,
            "paths": {od: list(paths[od]) for od in paths},
            "status": "lp_optimal",
        }

        return RMPResult(status="optimal", obj=float(res.fun), solution=solution, duals=duals)

    def column_generation(self, node: BPNode) -> RMPResult:
        """在当前分支节点内执行列生成。"""
        for _ in range(self.max_cg_iter):
            rmp = self.solve_rmp_lp(node.paths, node.cuts)
            if rmp.status != "optimal":
                return rmp

            added = 0
            for od in self.data.od_data:
                forbidden = {p.signature for p in node.paths[od]}
                # A <= 0 branch on a path-flow variable forbids that path in
                # every subsequent pricing problem at the node.
                forbidden.update(
                    cut.key[1]
                    for cut in node.cuts
                    if cut.var_type == "y"
                    and cut.key[0] == od
                    and cut.sense == "<="
                    and cut.rhs <= INT_EPS
                )
                new_path, rc = pricing_label_algorithm(
                    data=self.data,
                    od=od,
                    duals=rmp.duals,
                    forbidden=forbidden,
                    branch_y_dual=rmp.duals.branch_y,
                )
                if new_path is not None and rc < -self.cg_eps:
                    node.paths[od].append(new_path)
                    added += 1
                    self.columns_generated += 1

            if added == 0:
                return rmp

        return self.solve_rmp_lp(node.paths, node.cuts)

    def choose_branch_variable(self, sol: Dict[str, Any]) -> Optional[Tuple[str, Any, float]]:
        """按 x_r、z_w、y_wp 顺序选择分支变量。"""
        candidates = []

        for r, v in sol["x"].items():
            if not is_integer_value(v):
                candidates.append((abs(frac_part(v) - 0.5), "x", r, v))
        if candidates:
            _, typ, key, val = min(candidates, key=lambda t: t[0])
            return typ, key, val

        candidates = []
        for od, v in sol["z"].items():
            if not is_integer_value(v):
                candidates.append((abs(frac_part(v) - 0.5), "z", od, v))
        if candidates:
            _, typ, key, val = min(candidates, key=lambda t: t[0])
            return typ, key, val

        candidates = []
        for key, v in sol["y"].items():
            if abs(v) > INT_EPS and not is_integer_value(v):
                candidates.append((abs(frac_part(v) - 0.5), "y", key, v))
        if candidates:
            _, typ, key, val = min(candidates, key=lambda t: t[0])
            return typ, key, val

        return None

    def solve(self) -> Dict[str, Any]:
        """分支定价主循环。"""
        self.start_time = time.time()
        self.nodes_explored = 0

        root_paths = initial_path_pool(self.data, k_each_od=8)
        ub = self.all_road_upper_bound()
        best_solution = self.all_road_solution(root_paths)

        root = BPNode(
            node_id=next(self.node_counter),
            depth=0,
            cuts=[],
            paths={od: list(root_paths[od]) for od in root_paths},
            lb=-INF,
        )

        active = []
        heapq.heappush(active, (root.lb, root.depth, root.node_id, root))

        while active:
            if time.time() - self.start_time >= self.time_limit:
                break
            if self.nodes_explored >= self.max_nodes:
                break

            # 全局 gap 判断
            if active[0][0] > -INF / 2 and ub < INF / 2:
                global_lb = active[0][0]
                gap = max(0.0, (ub - global_lb) / max(1.0, abs(ub)))
                if gap <= self.gap_tol:
                    break

            _, _, _, node = heapq.heappop(active)
            self.nodes_explored += 1

            rmp = self.column_generation(node)
            if rmp.status != "optimal":
                continue

            node.lb = rmp.obj

            # 下界剪枝
            if node.lb >= ub - EPS:
                continue

            # 整数性检查
            branch_var = self.choose_branch_variable(rmp.solution)

            if branch_var is None:
                if rmp.obj < ub - EPS:
                    ub = rmp.obj
                    best_solution = rmp.solution
                    best_solution["status"] = "integer_feasible"
                continue

            # 生成左右分支
            typ, key, val = branch_var
            lo = math.floor(val)
            hi = math.ceil(val)

            for cut in [BranchCut(typ, key, "<=", lo), BranchCut(typ, key, ">=", hi)]:
                child = BPNode(
                    node_id=next(self.node_counter),
                    depth=node.depth + 1,
                    cuts=node.cuts + [cut],
                    paths={od: list(node.paths[od]) for od in node.paths},
                    lb=node.lb,
                )
                heapq.heappush(active, (child.lb, child.depth, child.node_id, child))

        best_solution["nodes_explored"] = self.nodes_explored
        best_solution["columns_generated"] = self.columns_generated
        best_solution["time_used"] = time.time() - self.start_time
        if active and active[0][0] > -INF / 2:
            self.final_lower_bound = active[0][0]
        elif best_solution.get("status") == "integer_feasible":
            self.final_lower_bound = best_solution["obj"]
        best_solution["lower_bound"] = self.final_lower_bound
        best_solution["relative_gap"] = (
            None
            if self.final_lower_bound <= -INF / 2
            else max(0.0, (best_solution["obj"] - self.final_lower_bound) / max(1.0, abs(best_solution["obj"])))
        )
        return best_solution

    def calc_carbon(self, sol: Dict[str, Any]) -> float:
        """计算系统碳排放量。"""
        total = 0.0

        # 全公路直运碳排放
        for od, zval in sol["z"].items():
            total += zval * self.data.od_data[od]["road_direct"] * self.data.e_road

        path_dict = {}
        for od, plist in sol["paths"].items():
            for p in plist:
                path_dict[(od, p.signature)] = p

        # 联运路径碳排放
        for key, yval in sol["y"].items():
            if abs(yval) <= INT_EPS:
                continue
            p = path_dict[key]
            total += yval * (
                p.d_road * self.data.e_road
                + p.d_rail * self.data.e_rail
                + len(p.lam) * self.data.tau
            )

        return total

    def calc_rail_share(self, sol: Dict[str, Any]) -> float:
        """按货物周转量口径计算铁路分担率。"""
        total_turnover = 0.0
        rail_turnover = 0.0

        for od, zval in sol["z"].items():
            total_turnover += zval * self.data.od_data[od]["road_direct"]

        path_dict = {}
        for od, plist in sol["paths"].items():
            for p in plist:
                path_dict[(od, p.signature)] = p

        for key, yval in sol["y"].items():
            if abs(yval) <= INT_EPS:
                continue
            p = path_dict[key]
            total_turnover += yval * (p.d_road + p.d_rail)
            rail_turnover += yval * p.d_rail

        return 0.0 if total_turnover <= INT_EPS else rail_turnover / total_turnover

    def format_path(self, p: Path, use_city_name: bool = False) -> str:
        """将路径转为可读文本。"""
        parts = []
        for e in p.edges:
            if use_city_name:
                u = self.data.city_names.get(e.u, str(e.u))
                v = self.data.city_names.get(e.v, str(e.v))
            else:
                u = str(e.u)
                v = str(e.v)

            if e.kind == "road":
                parts.append(f"{u}-{v}(road)")
            else:
                parts.append(f"{u}-{v}({e.service})")

        return " -> ".join(parts)

    def print_solution(self, sol: Dict[str, Any]) -> None:
        """输出最终求解结果。"""
        print("=" * 100)
        print("Real Case: Column Generation + Branch-and-Price")
        print(f"Status: {sol.get('status')}")
        print(f"Objective value: {sol['obj']:.2f} CNY")
        print(f"Carbon emissions: {self.calc_carbon(sol):.2f} kg CO2")
        print(f"Rail modal share: {100.0 * self.calc_rail_share(sol):.2f}%")
        print(f"Explored nodes: {sol.get('nodes_explored')}")
        print(f"Time used: {sol.get('time_used', 0.0):.2f} s")
        print("=" * 100)

        print("\nScheduled rail service frequencies:")
        for r in sorted(sol["x"], key=lambda x: int(x[1:])):
            v = sol["x"][r]
            if abs(v) > INT_EPS:
                print(f"  {r}: {round(float(v))}")

        path_dict = {}
        for od, plist in sol["paths"].items():
            for p in plist:
                path_dict[(od, p.signature)] = p

        print("\nOD transport schemes:")
        for od in sorted(self.data.od_data):
            o = int(self.data.od_data[od]["origin"])
            d = int(self.data.od_data[od]["dest"])
            print(f"\n{od}: {self.data.city_names[o]} -> {self.data.city_names[d]}")

            zval = sol["z"].get(od, 0.0)
            if abs(zval) > INT_EPS:
                print(f"  All-road direct: flow = {float(zval):.6f}")

            rows = []
            for key, yval in sol["y"].items():
                if key[0] != od:
                    continue
                if abs(yval) <= INT_EPS:
                    continue
                p = path_dict[key]
                rows.append((self.format_path(p, use_city_name=False), yval, p.cost * yval))

            rows.sort(key=lambda t: -t[1])

            for desc, flow, var_cost in rows:
                print(
                    f"  Intermodal: {desc}; "
                    f"flow = {float(flow):.6f}; "
                    f"variable cost = {float(var_cost):.2f}"
                )


# =========================
# 十、主程序入口
# =========================
if __name__ == "__main__":
    # 构建案例数据
    parser = argparse.ArgumentParser(description="Column generation / branch-and-price; no bundled case data.")
    parser.add_argument("--data-dir", type=FilePath, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--full-path-baseline", action="store_true")
    args = parser.parse_args()
    try:
        data = build_case(args.data_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        print("External data validated; no optimization performed.")
        sys.exit(0)

    # 创建分支定价求解器
    solver = BranchAndPriceSolver(
        data=data,
        cg_eps=1e-4,
        gap_tol=1e-4,
        time_limit=600.0,
        max_nodes=10000,
        max_cg_iter=200,
    )

    # 求解模型
    solution = solver.solve()

    # 输出结果
    solver.print_solution(solution)

    if args.full_path_baseline:
        print("\nEnumerating the complete path pool for the exact full-path baseline...")
        full_paths = enumerate_full_path_pool(data)
        # Keep the reported benchmark reproducible: the MILP solve itself is
        # limited to 60 s; path enumeration time is included in wall-clock time.
        baseline = solve_full_path_milp(data, full_paths, time_limit=60.0, mip_rel_gap=1e-4)
        print(json.dumps(baseline, ensure_ascii=False, indent=2))
        with open("full_path_baseline.json", "w", encoding="utf-8") as handle:
            json.dump(baseline, handle, ensure_ascii=False, indent=2)
