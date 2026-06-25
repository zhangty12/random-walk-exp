#!/usr/bin/env python3
"""
Micro-profile torch_cluster.random_walk on CUDA GPUs such as NVIDIA A100.

The script reports per-call latency and transition throughput for three modes:

  op                  Calls the low-level torch_cluster op with prebuilt CSR.
  wrapper_precoalesced Calls torch_cluster.random_walk on sorted row/col.
  wrapper_coalesced    Calls torch_cluster.random_walk with internal sorting.

Examples:
  python3 profile_random_walk.py --device cuda:0 --num-nodes 1000000 \
    --degree 16 --num-starts 1048576 --walk-length 20 \
    --modes op wrapper_precoalesced --iters 100 --warmup 20 \
    --json results/a100_rw.json --csv results/a100_rw.csv

  nsys profile -t cuda,nvtx -o results/rw_nsys \
    python3 profile_random_walk.py --device cuda:0 --modes op --nvtx
"""

from __future__ import annotations

import argparse
import csv
import inspect
import itertools
import json
import math
import os
import statistics
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_int_list(value: str) -> List[int]:
    try:
        parsed = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


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
    try:
        from importlib import metadata
    except ImportError:
        import importlib_metadata as metadata  # type: ignore

    for name in package_names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return None


def import_dependencies() -> Tuple[Any, Callable[..., Any]]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is not installed in this Python environment.") from exc

    try:
        import torch_cluster  # noqa: F401
        from torch_cluster import random_walk
    except ImportError as exc:
        raise SystemExit(
            "torch_cluster is not installed. Install the wheel matching your "
            "PyTorch/CUDA build before running this benchmark."
        ) from exc

    return torch, random_walk


def make_device(torch: Any, device_arg: str) -> Any:
    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available, but --device is set to CUDA.")
        torch.cuda.set_device(device.index if device.index is not None else 0)
    return device


def environment_info(torch: Any, device: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_cluster": package_version(["torch-cluster", "torch_cluster"]),
        "device": str(device),
    }

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "cuda_device_name": props.name,
                "cuda_capability": "%d.%d" % (props.major, props.minor),
                "cuda_total_memory_gb": round(props.total_memory / 1024**3, 3),
                "cuda_device_count": torch.cuda.device_count(),
            }
        )
    return info


def set_reproducibility(torch: Any, seed: int, device: Any) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def load_edge_index(
    torch: Any,
    path: str,
    device: Any,
    num_nodes: Optional[int],
) -> Tuple[Any, Any, Any, int]:
    loaded = torch.load(path, map_location=device)
    if isinstance(loaded, dict):
        if "edge_index" not in loaded:
            raise ValueError("edge-index file is a dict but has no 'edge_index' key")
        edge_index = loaded["edge_index"]
    else:
        edge_index = loaded

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")

    row = edge_index[0].to(device=device, dtype=torch.long).contiguous()
    col = edge_index[1].to(device=device, dtype=torch.long).contiguous()

    if num_nodes is None:
        max_node = torch.maximum(row.max(), col.max()).item()
        num_nodes = int(max_node) + 1

    perm = torch.argsort(row)
    row = row[perm].contiguous()
    col = col[perm].contiguous()
    rowptr = row_col_to_rowptr(torch, row, int(num_nodes))
    return row, col, rowptr, int(num_nodes)


def build_fixed_degree_graph(
    torch: Any,
    num_nodes: int,
    degree: int,
    device: Any,
    need_row: bool,
) -> Tuple[Optional[Any], Any, Any]:
    num_edges = num_nodes * degree
    col = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long, device=device)
    rowptr = torch.arange(
        0,
        num_edges + 1,
        degree,
        dtype=torch.long,
        device=device,
    )

    row = None
    if need_row:
        row = torch.arange(num_nodes, dtype=torch.long, device=device).repeat_interleave(
            degree
        )
    return row, col.contiguous(), rowptr.contiguous()


def row_col_to_rowptr(torch: Any, row: Any, num_nodes: int) -> Any:
    deg = row.new_zeros(num_nodes)
    deg.scatter_add_(0, row, torch.ones_like(row))
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])
    return rowptr


def make_starts(
    torch: Any,
    num_nodes: int,
    num_starts: int,
    device: Any,
    mode: str,
) -> Any:
    if mode == "range":
        return (torch.arange(num_starts, dtype=torch.long, device=device) % num_nodes).contiguous()
    if mode == "random":
        return torch.randint(0, num_nodes, (num_starts,), dtype=torch.long, device=device)
    raise ValueError("unknown start mode: %s" % mode)


def output_description(output: Any) -> Any:
    if isinstance(output, tuple):
        return [
            {"shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device)}
            for x in output
        ]
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "device": str(output.device),
    }


def make_callables(
    torch: Any,
    random_walk: Callable[..., Any],
    modes: Sequence[str],
    row: Optional[Any],
    col: Any,
    rowptr: Any,
    start: Any,
    num_nodes: int,
    walk_length: int,
    p: float,
    q: float,
    return_edge_indices: bool,
) -> Dict[str, Callable[[], Any]]:
    callables: Dict[str, Callable[[], Any]] = {}
    wrapper_accepts_return_edge_indices = (
        "return_edge_indices" in inspect.signature(random_walk).parameters
    )

    if "op" in modes:
        op = torch.ops.torch_cluster.random_walk

        def call_op() -> Any:
            return op(rowptr, col, start, int(walk_length), float(p), float(q))

        callables["op"] = call_op

    if "wrapper_precoalesced" in modes:
        if row is None:
            raise ValueError("wrapper_precoalesced needs row; internal error")

        def call_wrapper_precoalesced() -> Any:
            kwargs = {
                "p": float(p),
                "q": float(q),
                "coalesced": False,
                "num_nodes": int(num_nodes),
            }
            if wrapper_accepts_return_edge_indices:
                kwargs["return_edge_indices"] = return_edge_indices
            return random_walk(row, col, start, int(walk_length), **kwargs)

        callables["wrapper_precoalesced"] = call_wrapper_precoalesced

    if "wrapper_coalesced" in modes:
        if row is None:
            raise ValueError("wrapper_coalesced needs row; internal error")

        def call_wrapper_coalesced() -> Any:
            kwargs = {
                "p": float(p),
                "q": float(q),
                "coalesced": True,
                "num_nodes": int(num_nodes),
            }
            if wrapper_accepts_return_edge_indices:
                kwargs["return_edge_indices"] = return_edge_indices
            return random_walk(row, col, start, int(walk_length), **kwargs)

        callables["wrapper_coalesced"] = call_wrapper_coalesced

    return callables


def benchmark_callable(
    torch: Any,
    device: Any,
    fn: Callable[[], Any],
    mode: str,
    warmup: int,
    iters: int,
    nvtx: bool,
) -> Tuple[Dict[str, Any], Any]:
    last_output = None

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        last_output = fn()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times_ms: List[float] = []

    if device.type == "cuda":
        for i in range(iters):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            if nvtx:
                torch.cuda.nvtx.range_push("%s_iter_%d" % (mode, i))
            start_event.record()
            last_output = fn()
            end_event.record()
            end_event.synchronize()
            if nvtx:
                torch.cuda.nvtx.range_pop()
            times_ms.append(start_event.elapsed_time(end_event))
    else:
        for i in range(iters):
            if nvtx:
                raise ValueError("--nvtx requires CUDA")
            start_time = time.perf_counter()
            last_output = fn()
            times_ms.append((time.perf_counter() - start_time) * 1000.0)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device)
    else:
        peak_memory = None

    stats = {
        "warmup": warmup,
        "iters": iters,
        "latency_ms_mean": statistics.mean(times_ms),
        "latency_ms_std": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "latency_ms_min": min(times_ms),
        "latency_ms_p50": percentile(times_ms, 50),
        "latency_ms_p90": percentile(times_ms, 90),
        "latency_ms_p95": percentile(times_ms, 95),
        "latency_ms_p99": percentile(times_ms, 99),
        "latency_ms_max": max(times_ms),
        "peak_memory_gb": None
        if peak_memory is None
        else round(float(peak_memory) / 1024**3, 6),
    }
    return stats, last_output


def run_profiler(
    torch: Any,
    device: Any,
    fn: Callable[[], Any],
    mode: str,
    trace_dir: str,
    steps: int,
    record_shapes: bool,
    with_stack: bool,
    row_limit: int,
) -> Dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile, record_function

    os.makedirs(trace_dir, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    sort_by = "self_cpu_time_total"
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        sort_by = "self_cuda_time_total"

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=True,
        with_stack=with_stack,
    ) as prof:
        for _ in range(steps):
            with record_function("%s_random_walk" % mode):
                fn()
            prof.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    trace_path = os.path.join(
        trace_dir,
        "%s_%s.json" % (mode, datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    prof.export_chrome_trace(trace_path)
    table = prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
    return {"trace_path": trace_path, "table": table}


def print_summary(results: Sequence[Dict[str, Any]]) -> None:
    if not results:
        return

    print("\nSummary, sorted by median latency:")
    header = (
        "mode",
        "nodes",
        "degree",
        "starts",
        "walk_len",
        "p50_ms",
        "p90_ms",
        "steps/s",
        "walks/s",
    )
    print(
        "%-22s %10s %8s %10s %8s %12s %12s %14s %14s"
        % header
    )

    for item in sorted(results, key=lambda x: x["stats"]["latency_ms_p50"]):
        cfg = item["config"]
        stats = item["stats"]
        print(
            "%-22s %10d %8s %10d %8d %12.4f %12.4f %14.3e %14.3e"
            % (
                item["mode"],
                cfg["num_nodes"],
                str(cfg.get("degree")),
                cfg["num_starts"],
                cfg["walk_length"],
                stats["latency_ms_p50"],
                stats["latency_ms_p90"],
                stats["transitions_per_second_p50"],
                stats["walks_per_second_p50"],
            )
        )


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: str, results: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "mode",
        "num_nodes",
        "degree",
        "num_edges",
        "num_starts",
        "walk_length",
        "p",
        "q",
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
                    "mode": item["mode"],
                    "num_nodes": cfg["num_nodes"],
                    "degree": cfg.get("degree"),
                    "num_edges": cfg["num_edges"],
                    "num_starts": cfg["num_starts"],
                    "walk_length": cfg["walk_length"],
                    "p": cfg["p"],
                    "q": cfg["q"],
                    **{k: stats[k] for k in fields if k in stats},
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Micro-profile torch_cluster.random_walk on CUDA."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--num-nodes", type=positive_int, default=1_000_000)
    parser.add_argument("--degree", type=positive_int, default=16)
    parser.add_argument("--num-starts", type=positive_int, default=1_048_576)
    parser.add_argument("--walk-length", type=positive_int, default=20)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--start-mode", choices=("random", "range"), default="random")
    parser.add_argument(
        "--edge-index",
        default=None,
        help="Optional .pt file containing edge_index tensor [2, E] or dict['edge_index'].",
    )

    parser.add_argument("--sweep-num-nodes", type=parse_int_list, default=None)
    parser.add_argument("--sweep-degree", type=parse_int_list, default=None)
    parser.add_argument("--sweep-num-starts", type=parse_int_list, default=None)
    parser.add_argument("--sweep-walk-length", type=parse_int_list, default=None)

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("op", "wrapper_precoalesced", "wrapper_coalesced"),
        default=("op", "wrapper_precoalesced"),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=positive_int, default=100)
    parser.add_argument("--return-edge-indices", action="store_true")
    parser.add_argument("--nvtx", action="store_true")

    parser.add_argument("--json", default=None, help="Write full results to JSON.")
    parser.add_argument("--csv", default=None, help="Write summary rows to CSV.")

    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Optional directory for torch.profiler Chrome trace JSON files.",
    )
    parser.add_argument("--profile-steps", type=positive_int, default=10)
    parser.add_argument("--profile-record-shapes", action="store_true")
    parser.add_argument("--profile-with-stack", action="store_true")
    parser.add_argument("--profiler-rows", type=positive_int, default=20)
    return parser


def sweep_values(args: argparse.Namespace) -> Iterable[Tuple[int, int, int, int]]:
    num_nodes_values = args.sweep_num_nodes or [args.num_nodes]
    degree_values = args.sweep_degree or [args.degree]
    num_starts_values = args.sweep_num_starts or [args.num_starts]
    walk_length_values = args.sweep_walk_length or [args.walk_length]
    return itertools.product(
        num_nodes_values,
        degree_values,
        num_starts_values,
        walk_length_values,
    )


def main() -> None:
    args = build_parser().parse_args()
    torch, random_walk = import_dependencies()
    device = make_device(torch, args.device)
    set_reproducibility(torch, args.seed, device)

    env = environment_info(torch, device)
    print("Environment:")
    print(json.dumps(env, indent=2, sort_keys=True))

    all_results: List[Dict[str, Any]] = []
    need_row = any(mode.startswith("wrapper_") for mode in args.modes)

    loaded_graph: Optional[Tuple[Any, Any, Any, int]] = None
    if args.edge_index is not None:
        print("\nLoading edge_index from %s" % args.edge_index)
        loaded_graph = load_edge_index(torch, args.edge_index, device, args.num_nodes)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for num_nodes, degree, num_starts, walk_length in sweep_values(args):
        if loaded_graph is not None:
            row, col, rowptr, actual_num_nodes = loaded_graph
            num_nodes = actual_num_nodes
            degree_for_report: Optional[float] = None
        else:
            print(
                "\nBuilding synthetic graph: num_nodes=%d degree=%d"
                % (num_nodes, degree)
            )
            row, col, rowptr = build_fixed_degree_graph(
                torch,
                num_nodes=num_nodes,
                degree=degree,
                device=device,
                need_row=need_row,
            )
            degree_for_report = float(degree)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        start = make_starts(torch, num_nodes, num_starts, device, args.start_mode)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        num_edges = int(col.numel())
        transitions = int(num_starts) * int(walk_length)
        cfg = {
            "num_nodes": int(num_nodes),
            "degree": degree_for_report,
            "num_edges": num_edges,
            "num_starts": int(num_starts),
            "walk_length": int(walk_length),
            "p": float(args.p),
            "q": float(args.q),
            "start_mode": args.start_mode,
            "edge_index": args.edge_index,
        }

        callables = make_callables(
            torch=torch,
            random_walk=random_walk,
            modes=args.modes,
            row=row,
            col=col,
            rowptr=rowptr,
            start=start,
            num_nodes=int(num_nodes),
            walk_length=int(walk_length),
            p=float(args.p),
            q=float(args.q),
            return_edge_indices=bool(args.return_edge_indices),
        )

        for mode, fn in callables.items():
            print("\nBenchmarking %s with config: %s" % (mode, cfg))
            try:
                stats, output = benchmark_callable(
                    torch=torch,
                    device=device,
                    fn=fn,
                    mode=mode,
                    warmup=args.warmup,
                    iters=args.iters,
                    nvtx=args.nvtx,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Benchmark mode '%s' failed. If only 'op' fails, this "
                    "torch_cluster build may not expose the rowptr low-level "
                    "op; rerun with --modes wrapper_precoalesced."
                    % mode
                ) from exc

            p50_seconds = stats["latency_ms_p50"] / 1000.0
            stats["transitions_per_second_p50"] = transitions / p50_seconds
            stats["walks_per_second_p50"] = int(num_starts) / p50_seconds
            stats["output"] = output_description(output)

            result = {"mode": mode, "config": dict(cfg), "stats": stats}

            if args.trace_dir is not None:
                print("Running torch.profiler for %s" % mode)
                profiler = run_profiler(
                    torch=torch,
                    device=device,
                    fn=fn,
                    mode=mode,
                    trace_dir=args.trace_dir,
                    steps=args.profile_steps,
                    record_shapes=args.profile_record_shapes,
                    with_stack=args.profile_with_stack,
                    row_limit=args.profiler_rows,
                )
                result["profiler"] = profiler
                print(profiler["table"])
                print("Trace: %s" % profiler["trace_path"])

            all_results.append(result)

            print(
                "%s: p50=%.4f ms, p90=%.4f ms, transitions/s=%.3e, walks/s=%.3e"
                % (
                    mode,
                    stats["latency_ms_p50"],
                    stats["latency_ms_p90"],
                    stats["transitions_per_second_p50"],
                    stats["walks_per_second_p50"],
                )
            )

    print_summary(all_results)

    payload = {"environment": env, "args": vars(args), "results": all_results}
    if args.json:
        write_json(args.json, payload)
        print("\nWrote JSON: %s" % args.json)
    if args.csv:
        write_csv(args.csv, all_results)
        print("Wrote CSV: %s" % args.csv)


if __name__ == "__main__":
    main()
