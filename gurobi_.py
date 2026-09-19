# -*- coding: utf-8 -*-
"""Direct Gurobi benchmark with external case data only.

Requires the companion data-free main module in this directory.
Full enumeration uses the original path restrictions; --competitive-only
is a filtered-path model. No case values or datasets are bundled.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None


DEFAULT_SOURCE = Path(__file__).resolve().with_name("论文代码_无内嵌数据.py")
MAX_ROAD_ARCS = 3
MAX_RAIL_SERVICES = 4
DEFAULT_TOTAL_TIME_LIMIT = 2 * 60 * 60  # 整个程序默认最多运行 2 小时
RESULT_DIR = Path("gurobi_results")


def load_source(source: Path):
    """Load companion structures and external-data reader, not its main solver."""
    spec = importlib.util.spec_from_file_location("paper_case_data", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法载入源文件：{source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OverallTimeLimitReached(RuntimeError):
    """整个程序达到运行时间上限。"""


def ensure_time_remaining(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise OverallTimeLimitReached


def enumerate_paths(module, data, od: str, competitive_only: bool, deadline: float):
    """完整 DFS 枚举原定价程序资源上限内的简单联运路径。"""
    origin = int(data.od_data[od]["origin"])
    dest = int(data.od_data[od]["dest"])
    cutoff = module.direct_road_cost(data, od) if competitive_only else math.inf
    result = []

    def dfs(node, visited: Set[int], used_services: Set[str], road_count: int,
            state, edges: Tuple, trans_nodes: Tuple[int, ...], cost: float,
            has_rail: bool):
        ensure_time_remaining(deadline)
        if node == dest:
            if has_rail:
                result.append(module.make_path(data, od, edges, trans_nodes))
            return

        for edge in data.outgoing.get(node, []):
            if edge.v in visited:
                continue
            next_road_count = road_count + (1 if edge.kind == "road" else 0)
            if next_road_count > MAX_ROAD_ARCS:
                continue
            next_services = set(used_services)
            if edge.kind == "rail":
                next_services.add(edge.service)
            if len(next_services) > MAX_RAIL_SERVICES:
                continue

            transfer = module.trans_required(state, edge.state)
            add_cost = (module.road_cost(data, edge.dist) if edge.kind == "road"
                        else module.rail_cost(data, edge.dist))
            if transfer:
                if node not in data.hubs:
                    continue
                add_cost += module.trans_cost(data)
            if cost + add_cost > cutoff + 1e-9:
                continue

            dfs(
                edge.v,
                visited | {edge.v},
                next_services,
                next_road_count,
                edge.state,
                edges + (edge,),
                trans_nodes + ((node,) if transfer else tuple()),
                cost + add_cost,
                has_rail or edge.kind == "rail",
            )

    dfs(origin, {origin}, set(), 0, None, tuple(), tuple(), 0.0, False)
    return result


def build_and_solve(module, data, paths: Dict[str, List], args):
    ensure_time_remaining(args.deadline)
    model_build_started = time.perf_counter()
    model = gp.Model("direct_intermodal_MIP")
    model.Params.MIPGap = args.mip_gap
    model.Params.Threads = args.threads

    # 与原模型一致：x、z、y 均为非负整数变量。
    x = model.addVars(data.services.keys(), vtype=GRB.INTEGER, lb=0, name="x")
    z = model.addVars(data.od_data.keys(), vtype=GRB.INTEGER, lb=0, name="z")
    y = {}
    for od, plist in paths.items():
        for k, _ in enumerate(plist):
            y[od, k] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y[{od},{k}]")
    model.update()

    model.setObjective(
        gp.quicksum(data.train_fixed_cost * x[r] for r in data.services)
        + gp.quicksum(module.direct_road_cost(data, od) * z[od] for od in data.od_data)
        + gp.quicksum(p.cost * y[od, k] for od, plist in paths.items()
                      for k, p in enumerate(plist)),
        GRB.MINIMIZE,
    )

    # (1) OD 需求平衡：sum_p y_wp + z_w = q_w
    for od, info in data.od_data.items():
        model.addConstr(
            z[od] + gp.quicksum(y[od, k] for k in range(len(paths[od]))) == info["demand"],
            name=f"demand[{od}]",
        )

    # (2) 班列服务各运行区段容量：sum delta*y <= Q*x
    for r, sections in data.service_sections.items():
        for u, v in sections:
            model.addConstr(
                gp.quicksum(y[od, k] for od, plist in paths.items()
                            for k, p in enumerate(plist) if p.delta.get((r, u, v), 0))
                <= data.train_capacity * x[r],
                name=f"service_capacity[{r},{u},{v}]",
            )

    # (3) 最小开行规模：Qmin*x <= sum theta*y（完全沿用原代码定义）
    for r in data.services:
        model.addConstr(
            data.min_loading * x[r]
            <= gp.quicksum(y[od, k] for od, plist in paths.items()
                           for k, p in enumerate(plist) if r in p.services_used),
            name=f"minimum_loading[{r}]",
        )

    # (4) 铁路物理区段通过能力：sum beta*x <= U
    for u, v in data.physical_rail_sections:
        model.addConstr(
            gp.quicksum(x[r] for r, sections in data.service_sections.items()
                        if (u, v) in sections) <= data.rail_section_capacity,
            name=f"rail_section_capacity[{u},{v}]",
        )

    # (5) 最大开行频次
    for r in data.services:
        model.addConstr(x[r] <= data.max_frequency, name=f"maximum_frequency[{r}]")

    # (6) 中间换装节点能力
    for s in data.hubs:
        model.addConstr(
            gp.quicksum(y[od, k] for od, plist in paths.items()
                        for k, p in enumerate(plist) if p.lam.get(s, 0))
            <= data.hub_capacity,
            name=f"hub_capacity[{s}]",
        )

    model_build_seconds = time.perf_counter() - model_build_started
    # 模型构建占用的时间也计入总时限，求解器只使用剩余时间。
    ensure_time_remaining(args.deadline)
    model.Params.TimeLimit = max(0.001, args.deadline - time.perf_counter())
    model.optimize()
    # model.Runtime 是 Gurobi 内部记录的纯 optimize() 时间。
    return model, x, z, y, model_build_seconds, float(model.Runtime)


def print_solution(module, data, paths, model, x, z, y, timings, competitive_only):
    lines = []

    def emit(text=""):
        print(text)
        lines.append(str(text))

    emit("=" * 90)
    emit("求解方式：Gurobi 直接 MIP（无列生成、无定价、无手写分支树）")
    emit("路径模式：" + ("经济路径筛选（非完整路径集）" if competitive_only else "完整枚举"))
    emit(f"Gurobi状态：{model.Status}")
    emit(f"路径数：{sum(map(len, paths.values()))}")
    emit("\n运行时间记录：")
    emit(f"  数据读取及初始化：{timings['data']:.6f} 秒")
    emit(f"  路径生成：        {timings['paths']:.6f} 秒")
    emit(f"  Gurobi模型构建：  {timings['build']:.6f} 秒")
    emit(f"  Gurobi纯求解：    {timings['solve']:.6f} 秒")
    emit(f"  程序总运行时间：  {timings['total']:.6f} 秒")
    if model.SolCount == 0:
        emit("没有可输出的可行解。")
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        record = RESULT_DIR / "gurobi_run_record.txt"
        record.write_text("\n".join(lines), encoding="utf-8")
        emit(f"运行记录已保存：{record}")
        return
    emit(f"目标值：{model.ObjVal:.2f}")
    if model.Status != GRB.OPTIMAL:
        emit(f"当前下界：{model.ObjBound:.2f}")
        emit(f"MIP Gap：{100.0 * model.MIPGap:.4f}%")

    emit("\n班列开行频次：")
    for r in sorted(data.services, key=lambda q: int(q[1:])):
        if x[r].X > 0.5:
            emit(f"  {r}: {round(x[r].X)}")

    emit("\nOD运输方案：")
    for od in data.od_data:
        emit(f"\n{od}: 全公路货量 = {z[od].X:.0f}")
        used = [(k, var.X) for (w, k), var in y.items() if w == od and var.X > 0.5]
        used.sort(key=lambda item: -item[1])
        for k, flow in used:
            p = paths[od][k]
            desc = " -> ".join(
                f"{e.u}-{e.v}({'road' if e.kind == 'road' else e.service})" for e in p.edges
            )
            emit(f"  联运货量={flow:.0f}; 单位成本={p.cost:.2f}; {desc}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    record = RESULT_DIR / "gurobi_run_record.txt"
    record.write_text("\n".join(lines), encoding="utf-8")
    emit(f"\n运行记录已保存：{record}")


def main():
    global RESULT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Path to companion module exposing build_case(data_dir)")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TOTAL_TIME_LIMIT,
        help="整个程序的运行时间上限（秒），默认 7200 秒，即 2 小时",
    )
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--competitive-only", action="store_true",
                        help="仅保留单位成本不高于全公路的路径；速度快但不是完整路径集")
    args = parser.parse_args()
    RESULT_DIR = args.output_dir

    started = time.perf_counter()
    args.deadline = started + args.time_limit
    data_started = time.perf_counter()
    module = load_source(args.source)
    try:
        data = module.build_case(args.data_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        print("External data validated; no Gurobi optimization performed.")
        return
    if gp is None:
        parser.error("Install gurobipy and configure a valid Gurobi license to solve.")
    ensure_time_remaining(args.deadline)
    data_seconds = time.perf_counter() - data_started
    paths_started = time.perf_counter()
    paths = {}
    try:
        for od in data.od_data:
            paths[od] = enumerate_paths(
                module, data, od, args.competitive_only, args.deadline
            )
            print(f"{od} 已生成路径 {len(paths[od])} 条")
    except OverallTimeLimitReached:
        elapsed = time.perf_counter() - started
        print(f"程序已达到整体运行时间上限 {args.time_limit:.0f} 秒，自动停止。")
        print(f"实际运行时间：{elapsed:.2f} 秒")
        return
    paths_seconds = time.perf_counter() - paths_started

    try:
        model, x, z, y, build_seconds, solve_seconds = build_and_solve(
            module, data, paths, args
        )
    except OverallTimeLimitReached:
        elapsed = time.perf_counter() - started
        print(f"程序已达到整体运行时间上限 {args.time_limit:.0f} 秒，自动停止。")
        print(f"实际运行时间：{elapsed:.2f} 秒")
        return
    timings = {
        "data": data_seconds,
        "paths": paths_seconds,
        "build": build_seconds,
        "solve": solve_seconds,
        "total": time.perf_counter() - started,
    }
    print_solution(module, data, paths, model, x, z, y, timings, args.competitive_only)


if __name__ == "__main__":
    main()
