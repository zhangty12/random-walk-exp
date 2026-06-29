#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from custom_random_walk import load_extension


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def package_version(package_names: Sequence[str]) -> Optional[str]:
    for name in package_names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return None


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_device(torch: Any, device_arg: str) -> Any:
    device = torch.device(device_arg)
    if device.type != "cuda":
        raise SystemExit("This comparison requires a CUDA device.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available, but the config requests a CUDA run.")
    torch.cuda.set_device(device.index if device.index is not None else 0)
    return device


def environment_info(torch: Any, device: Any) -> Dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_cluster": package_version(["torch-cluster", "torch_cluster"]),
        "device": str(device),
        "cuda_device_name": props.name,
        "cuda_capability": "%d.%d" % (props.major, props.minor),
        "cuda_total_memory_gb": round(props.total_memory / 1024**3, 3),
    }


def synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def row_col_to_rowptr(torch: Any, row: Any, num_nodes: int) -> Any:
    degree = torch.bincount(row, minlength=num_nodes)
    rowptr = torch.empty(num_nodes + 1, dtype=torch.long, device=row.device)
    rowptr[0] = 0
    torch.cumsum(degree, 0, out=rowptr[1:])
    return rowptr.contiguous()


def rowptr_to_row(torch: Any, rowptr: Any) -> Any:
    num_nodes = rowptr.numel() - 1
    degree = rowptr[1:] - rowptr[:-1]
    return torch.arange(num_nodes, dtype=torch.long, device=rowptr.device).repeat_interleave(
        degree
    )


def sort_by_row(torch: Any, row: Any, col: Any) -> Tuple[Any, Any]:
    perm = torch.argsort(row)
    return row[perm].contiguous(), col[perm].contiguous()


def build_random_graph(torch: Any, graph_cfg: Dict[str, Any], device: Any, seed: int) -> Dict[str, Any]:
    num_nodes = int(graph_cfg["num_nodes"])
    avg_degree = float(graph_cfg["avg_degree"])
    if num_nodes < 2:
        raise ValueError("random graph needs at least two nodes")

    undirected_edges = max(1, int(round(num_nodes * avg_degree / 2.0)))
    generator = torch.Generator(device=str(device))
    generator.manual_seed(seed)

    src = torch.randint(
        0, num_nodes, (undirected_edges,), dtype=torch.long, device=device, generator=generator
    )
    offset = torch.randint(
        1, num_nodes, (undirected_edges,), dtype=torch.long, device=device, generator=generator
    )
    dst = (src + offset) % num_nodes

    row = torch.cat([src, dst], dim=0)
    col = torch.cat([dst, src], dim=0)
    row, col = sort_by_row(torch, row, col)
    rowptr = row_col_to_rowptr(torch, row, num_nodes)
    return {
        "rowptr": rowptr,
        "col": col,
        "num_nodes": num_nodes,
        "num_edges": int(col.numel()),
        "avg_degree": float(col.numel()) / float(num_nodes),
        "graph_type": "random",
    }


def build_power_law_graph(
    torch: Any, graph_cfg: Dict[str, Any], device: Any, seed: int
) -> Dict[str, Any]:
    num_nodes = int(graph_cfg["num_nodes"])
    attachment_edges = int(graph_cfg["attachment_edges"])
    if attachment_edges < 1:
        raise ValueError("attachment_edges must be positive")
    if num_nodes <= attachment_edges + 1:
        raise ValueError("num_nodes must be larger than attachment_edges + 1")

    rng = random.Random(seed)
    initial_nodes = attachment_edges + 1
    rows: List[int] = []
    cols: List[int] = []
    repeated: List[int] = []

    for i in range(initial_nodes):
        for j in range(i + 1, initial_nodes):
            rows.extend([i, j])
            cols.extend([j, i])
            repeated.extend([i, j])

    for new_node in range(initial_nodes, num_nodes):
        targets = set()
        while len(targets) < attachment_edges:
            targets.add(repeated[rng.randrange(len(repeated))])

        for dst in targets:
            rows.extend([new_node, dst])
            cols.extend([dst, new_node])
            repeated.extend([new_node, dst])

    row = torch.tensor(rows, dtype=torch.long, device=device)
    col = torch.tensor(cols, dtype=torch.long, device=device)
    row, col = sort_by_row(torch, row, col)
    rowptr = row_col_to_rowptr(torch, row, num_nodes)
    return {
        "rowptr": rowptr,
        "col": col,
        "num_nodes": num_nodes,
        "num_edges": int(col.numel()),
        "avg_degree": float(col.numel()) / float(num_nodes),
        "graph_type": "power_law",
    }


def build_graph(torch: Any, graph_cfg: Dict[str, Any], device: Any, seed: int) -> Dict[str, Any]:
    graph_type = graph_cfg["type"]
    if graph_type == "random":
        return build_random_graph(torch, graph_cfg, device, seed)
    if graph_type == "power_law":
        return build_power_law_graph(torch, graph_cfg, device, seed)
    raise ValueError("unknown graph type: %s" % graph_type)


def make_starts(
    torch: Any, num_nodes: int, num_starts: int, device: Any, mode: str, seed: int
) -> Any:
    if mode == "range":
        return (torch.arange(num_starts, dtype=torch.long, device=device) % num_nodes).contiguous()
    if mode == "random":
        generator = torch.Generator(device=str(device))
        generator.manual_seed(seed)
        return torch.randint(
            0, num_nodes, (num_starts,), dtype=torch.long, device=device, generator=generator
        ).contiguous()
    raise ValueError("unknown start mode: %s" % mode)


def time_cuda_once(torch: Any, device: Any, fn: Callable[[], Any]) -> Tuple[Any, float]:
    synchronize(torch, device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    output = fn()
    end_event.record()
    end_event.synchronize()
    return output, float(start_event.elapsed_time(end_event))


def benchmark_callable(
    torch: Any,
    device: Any,
    fn: Callable[[], Any],
    warmup: int,
    iters: int,
) -> Tuple[Dict[str, Any], Any]:
    last_output = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        last_output = fn()
    synchronize(torch, device)

    times_ms: List[float] = []
    for _ in range(iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        last_output = fn()
        end_event.record()
        end_event.synchronize()
        times_ms.append(float(start_event.elapsed_time(end_event)))

    synchronize(torch, device)
    peak_memory = torch.cuda.max_memory_allocated(device)
    stats = {
        "warmup": int(warmup),
        "iters": int(iters),
        "latency_ms_mean": statistics.mean(times_ms),
        "latency_ms_std": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "latency_ms_min": min(times_ms),
        "latency_ms_p50": percentile(times_ms, 50),
        "latency_ms_p90": percentile(times_ms, 90),
        "latency_ms_p95": percentile(times_ms, 95),
        "latency_ms_p99": percentile(times_ms, 99),
        "latency_ms_max": max(times_ms),
        "peak_memory_gb": round(float(peak_memory) / 1024**3, 6),
    }
    return stats, last_output


def output_description(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        return [output_description(item) for item in output]
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "device": str(output.device),
    }


def make_library_callable(
    torch: Any,
    rowptr: Any,
    col: Any,
    start: Any,
    walk_length: int,
    p: float,
    q: float,
    num_nodes: int,
) -> Callable[[], Any]:
    try:
        import torch_cluster  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "torch_cluster is required for mode 'library'. Install the wheel matching "
            "your PyTorch/CUDA build or remove 'library' from the config modes."
        ) from exc

    try:
        op = torch.ops.torch_cluster.random_walk

        def call_op() -> Any:
            return op(rowptr, col, start, int(walk_length), float(p), float(q))

        return call_op
    except (AttributeError, RuntimeError):
        from torch_cluster import random_walk

        row = rowptr_to_row(torch, rowptr).contiguous()

        def call_wrapper() -> Any:
            return random_walk(
                row,
                col,
                start,
                int(walk_length),
                p=float(p),
                q=float(q),
                coalesced=False,
                num_nodes=int(num_nodes),
            )

        return call_wrapper


def make_binary_search_callable(
    module: Any,
    rowptr: Any,
    col: Any,
    start: Any,
    walk_length: int,
    p: float,
    q: float,
    linear_threshold: int,
    seed: int,
) -> Callable[[], Any]:
    def call_new_impl() -> Any:
        return module.random_walk(
            rowptr,
            col,
            start,
            int(walk_length),
            float(p),
            float(q),
            int(linear_threshold),
            int(seed),
        )

    return call_new_impl


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: str, results: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "graph_name",
        "graph_type",
        "mode",
        "num_nodes",
        "num_edges",
        "avg_degree",
        "num_starts",
        "walk_length",
        "p",
        "q",
        "linear_threshold",
        "sort_csr_ms",
        "latency_ms_mean",
        "latency_ms_std",
        "latency_ms_min",
        "latency_ms_p50",
        "latency_ms_p90",
        "latency_ms_p95",
        "latency_ms_p99",
        "latency_ms_max",
        "transitions_per_second_p50",
        "walks_per_second_p50",
        "peak_memory_gb",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in results:
            cfg = item["config"]
            stats = item["stats"]
            writer.writerow(
                {
                    "graph_name": cfg["graph_name"],
                    "graph_type": cfg["graph_type"],
                    "mode": item["mode"],
                    "num_nodes": cfg["num_nodes"],
                    "num_edges": cfg["num_edges"],
                    "avg_degree": cfg["avg_degree"],
                    "num_starts": cfg["num_starts"],
                    "walk_length": cfg["walk_length"],
                    "p": cfg["p"],
                    "q": cfg["q"],
                    "linear_threshold": cfg["linear_threshold"],
                    "sort_csr_ms": cfg["sort_csr_ms"],
                    **{field: stats[field] for field in fields if field in stats},
                }
            )


def print_summary(results: Sequence[Dict[str, Any]]) -> None:
    if not results:
        return
    print("\nSummary:")
    print(
        "%-20s %-14s %-14s %8s %8s %12s %12s %14s"
        % ("graph", "mode", "p/q", "nodes", "starts", "p50_ms", "p90_ms", "steps/s")
    )
    for item in results:
        cfg = item["config"]
        stats = item["stats"]
        print(
            "%-20s %-14s %-14s %8d %8d %12.4f %12.4f %14.3e"
            % (
                cfg["graph_name"],
                item["mode"],
                "%.3g/%.3g" % (cfg["p"], cfg["q"]),
                cfg["num_nodes"],
                cfg["num_starts"],
                stats["latency_ms_p50"],
                stats["latency_ms_p90"],
                stats["transitions_per_second_p50"],
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare torch_cluster.random_walk with a threshold-adaptive sorted-CSR CUDA implementation."
    )
    parser.add_argument("--config", default="compare_random_walk_config.json")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cuda:1.")
    parser.add_argument("--json", default=None, help="Override JSON output path.")
    parser.add_argument("--csv", default=None, help="Override CSV output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required to run this benchmark.") from exc

    config_path = Path(args.config)
    config = load_json(str(config_path))
    device = make_device(torch, args.device or config.get("device", "cuda:0"))
    torch.manual_seed(int(config.get("seed", 12345)))
    torch.cuda.manual_seed_all(int(config.get("seed", 12345)))

    env = environment_info(torch, device)
    print("Environment:")
    print(json.dumps(env, indent=2, sort_keys=True))

    extension_cfg = config.get("extension", {})
    module = load_extension(
        name=extension_cfg.get("name", "binary_search_random_walk_ext"),
        verbose=bool(extension_cfg.get("verbose", False)),
        extra_cuda_cflags=extension_cfg.get("extra_cuda_cflags", []),
    )

    bench_cfg = config.get("benchmark", {})
    modes = list(bench_cfg.get("modes", ["library", "adaptive"]))
    warmup = int(bench_cfg.get("warmup", 10))
    iters = int(bench_cfg.get("iters", 50))
    walk_length = int(bench_cfg.get("walk_length", 20))
    start_mode = str(bench_cfg.get("start_mode", "random"))
    linear_threshold = int(bench_cfg.get("linear_threshold", 32))
    base_seed = int(config.get("seed", 12345))

    all_results: List[Dict[str, Any]] = []

    for graph_index, graph_cfg in enumerate(config["graphs"]):
        graph_seed = base_seed + graph_index * 1009
        graph_name = graph_cfg["name"]
        print("\nBuilding graph %s" % graph_name)
        synchronize(torch, device)
        build_start = time.perf_counter()
        graph = build_graph(torch, graph_cfg, device, graph_seed)
        synchronize(torch, device)
        graph_build_seconds = time.perf_counter() - build_start

        sorted_col, sort_csr_ms = time_cuda_once(
            torch, device, lambda: module.sort_csr_col(graph["rowptr"], graph["col"])
        )
        sorted_col = sorted_col.contiguous()
        synchronize(torch, device)

        print(
            "Graph %s: nodes=%d edges=%d avg_degree=%.3f build=%.3fs sort=%.3fms"
            % (
                graph_name,
                graph["num_nodes"],
                graph["num_edges"],
                graph["avg_degree"],
                graph_build_seconds,
                sort_csr_ms,
            )
        )

        num_starts = int(graph_cfg["num_starts"])
        start = make_starts(
            torch,
            graph["num_nodes"],
            num_starts,
            device,
            start_mode,
            graph_seed + 17,
        )
        synchronize(torch, device)

        for pq_index, pq_cfg in enumerate(config["node2vec"]):
            p = float(pq_cfg["p"])
            q = float(pq_cfg["q"])
            transitions = num_starts * walk_length

            callables: Dict[str, Callable[[], Any]] = {}
            if "library" in modes:
                callables["library"] = make_library_callable(
                    torch=torch,
                    rowptr=graph["rowptr"],
                    col=sorted_col,
                    start=start,
                    walk_length=walk_length,
                    p=p,
                    q=q,
                    num_nodes=graph["num_nodes"],
                )
            for custom_mode in [mode for mode in modes if mode in ("adaptive", "binary_search")]:
                callables[custom_mode] = make_binary_search_callable(
                    module=module,
                    rowptr=graph["rowptr"],
                    col=sorted_col,
                    start=start,
                    walk_length=walk_length,
                    p=p,
                    q=q,
                    linear_threshold=linear_threshold,
                    seed=graph_seed + pq_index * 37,
                )

            for mode, fn in callables.items():
                cfg = {
                    "graph_name": graph_name,
                    "graph_type": graph["graph_type"],
                    "num_nodes": graph["num_nodes"],
                    "num_edges": graph["num_edges"],
                    "avg_degree": graph["avg_degree"],
                    "num_starts": num_starts,
                    "walk_length": walk_length,
                    "p": p,
                    "q": q,
                    "linear_threshold": linear_threshold,
                    "sort_csr_ms": sort_csr_ms,
                    "graph_build_seconds": graph_build_seconds,
                    "start_mode": start_mode,
                }
                print(
                    "Benchmarking %-14s graph=%s p=%.4g q=%.4g"
                    % (mode, graph_name, p, q)
                )
                stats, output = benchmark_callable(
                    torch=torch,
                    device=device,
                    fn=fn,
                    warmup=warmup,
                    iters=iters,
                )
                p50_seconds = stats["latency_ms_p50"] / 1000.0
                stats["transitions_per_second_p50"] = transitions / p50_seconds
                stats["walks_per_second_p50"] = num_starts / p50_seconds
                stats["output"] = output_description(output)
                result = {"mode": mode, "config": cfg, "stats": stats}
                all_results.append(result)
                print(
                    "%s: p50=%.4f ms p90=%.4f ms transitions/s=%.3e"
                    % (
                        mode,
                        stats["latency_ms_p50"],
                        stats["latency_ms_p90"],
                        stats["transitions_per_second_p50"],
                    )
                )

    print_summary(all_results)

    output_cfg = config.get("output", {})
    json_path = args.json or output_cfg.get("json")
    csv_path = args.csv or output_cfg.get("csv")
    payload = {
        "environment": env,
        "config_path": str(config_path),
        "config": config,
        "results": all_results,
    }
    if json_path:
        write_json(json_path, payload)
        print("\nWrote JSON: %s" % json_path)
    if csv_path:
        write_csv(csv_path, all_results)
        print("Wrote CSV: %s" % csv_path)


if __name__ == "__main__":
    main()
