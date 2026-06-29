#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from compare_random_walk import (
    benchmark_callable,
    environment_info,
    make_binary_search_callable,
    make_device,
    make_library_callable,
    make_starts,
    output_description,
    print_summary,
    row_col_to_rowptr,
    sort_by_row,
    synchronize,
    time_cuda_once,
    write_json,
)
from custom_random_walk import load_extension


DATASETS: Dict[str, Dict[str, str]] = {
    "twitter_combined": {
        "url": "https://snap.stanford.edu/data/twitter_combined.txt.gz",
        "description": "SNAP Social circles: Twitter combined ego-network edge list",
    },
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def parse_pq_pairs(value: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        fields = [item.strip() for item in chunk.split(",")]
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(
                "expected p,q pairs separated by semicolons, e.g. '1,1;0.25,4'"
            )
        p = float(fields[0])
        q = float(fields[1])
        if p <= 0.0 or q <= 0.0:
            raise argparse.ArgumentTypeError("p and q must be positive")
        pairs.append((p, q))
    if not pairs:
        raise argparse.ArgumentTypeError("expected at least one p,q pair")
    return pairs


def dataset_filename(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = os.path.basename(parsed.path)
    if not name:
        raise ValueError("could not infer a filename from URL: %s" % url)
    return name


def download_file(url: str, path: Path, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        print("Using existing download: %s" % path)
        return

    tmp_path = path.with_name(path.name + ".tmp")
    print("Downloading %s" % url)
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as out:
        shutil.copyfileobj(response, out)
    tmp_path.replace(path)
    print("Wrote %s" % path)


def resolve_edge_list(args: argparse.Namespace) -> Tuple[Path, str, str]:
    if args.edge_list is not None:
        path = Path(args.edge_list)
        if not path.exists():
            raise FileNotFoundError(path)
        return path, "local", args.graph_name or path.name

    dataset = DATASETS[args.dataset]
    url = args.url or dataset["url"]
    path = Path(args.data_dir) / dataset_filename(url)
    download_file(url, path, force=bool(args.force_download))
    return path, url, args.graph_name or args.dataset


def open_edge_text(path: Path) -> Iterable[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield line
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield line


def parse_edge_list(
    path: Path,
    make_undirected: bool,
    keep_self_loops: bool,
    keep_duplicates: bool,
    max_raw_edges: Optional[int],
) -> Tuple[List[int], List[int], int, Dict[str, Any]]:
    node_map: Dict[int, int] = {}
    rows: List[int] = []
    cols: List[int] = []
    seen = None if keep_duplicates else set()
    raw_edges = 0
    skipped_self_loops = 0
    duplicate_edges = 0

    def node_id(raw: int) -> int:
        mapped = node_map.get(raw)
        if mapped is None:
            mapped = len(node_map)
            node_map[raw] = mapped
        return mapped

    def append_edge(src: int, dst: int) -> None:
        nonlocal duplicate_edges
        if seen is not None:
            edge = (src, dst)
            if edge in seen:
                duplicate_edges += 1
                return
            seen.add(edge)
        rows.append(src)
        cols.append(dst)

    for line in open_edge_text(path):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue

        raw_src = int(fields[0])
        raw_dst = int(fields[1])
        raw_edges += 1
        if max_raw_edges is not None and raw_edges > max_raw_edges:
            break

        if raw_src == raw_dst and not keep_self_loops:
            skipped_self_loops += 1
            continue

        src = node_id(raw_src)
        dst = node_id(raw_dst)
        append_edge(src, dst)
        if make_undirected and src != dst:
            append_edge(dst, src)

    if not rows:
        raise ValueError("no edges were parsed from %s" % path)

    stats = {
        "raw_edges_read": raw_edges if max_raw_edges is None else min(raw_edges, max_raw_edges),
        "skipped_self_loops": skipped_self_loops,
        "duplicate_edges_removed": duplicate_edges,
        "make_undirected": make_undirected,
        "keep_duplicates": keep_duplicates,
    }
    return rows, cols, len(node_map), stats


def degree_summary(torch: Any, rowptr: Any) -> Dict[str, Any]:
    degree = (rowptr[1:] - rowptr[:-1]).detach().cpu()
    degree_float = degree.to(dtype=torch.float64)
    quantile_points = torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float64)
    quantiles = torch.quantile(degree_float, quantile_points)
    return {
        "degree_min": int(degree.min().item()),
        "degree_mean": float(degree_float.mean().item()),
        "degree_max": int(degree.max().item()),
        "degree_p50": float(quantiles[0].item()),
        "degree_p90": float(quantiles[1].item()),
        "degree_p95": float(quantiles[2].item()),
        "degree_p99": float(quantiles[3].item()),
    }


def build_real_graph(
    torch: Any,
    path: Path,
    device: Any,
    make_undirected: bool,
    keep_self_loops: bool,
    keep_duplicates: bool,
    max_raw_edges: Optional[int],
) -> Dict[str, Any]:
    rows, cols, num_nodes, parse_stats = parse_edge_list(
        path=path,
        make_undirected=make_undirected,
        keep_self_loops=keep_self_loops,
        keep_duplicates=keep_duplicates,
        max_raw_edges=max_raw_edges,
    )

    row = torch.tensor(rows, dtype=torch.long, device=device)
    col = torch.tensor(cols, dtype=torch.long, device=device)
    row, col = sort_by_row(torch, row, col)
    rowptr = row_col_to_rowptr(torch, row, num_nodes)
    synchronize(torch, device)

    num_edges = int(col.numel())
    summary = degree_summary(torch, rowptr)
    return {
        "rowptr": rowptr,
        "col": col,
        "num_nodes": int(num_nodes),
        "num_edges": num_edges,
        "avg_degree": float(num_edges) / float(num_nodes),
        "graph_type": "real_power_law",
        "parse_stats": parse_stats,
        **summary,
    }


def write_results_csv(path: str, results: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "graph_name",
        "dataset_source",
        "mode",
        "num_nodes",
        "num_edges",
        "avg_degree",
        "degree_max",
        "degree_p50",
        "degree_p90",
        "degree_p95",
        "degree_p99",
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
                    "dataset_source": cfg["dataset_source"],
                    "mode": item["mode"],
                    "num_nodes": cfg["num_nodes"],
                    "num_edges": cfg["num_edges"],
                    "avg_degree": cfg["avg_degree"],
                    "degree_max": cfg["degree_max"],
                    "degree_p50": cfg["degree_p50"],
                    "degree_p90": cfg["degree_p90"],
                    "degree_p95": cfg["degree_p95"],
                    "degree_p99": cfg["degree_p99"],
                    "num_starts": cfg["num_starts"],
                    "walk_length": cfg["walk_length"],
                    "p": cfg["p"],
                    "q": cfg["q"],
                    "linear_threshold": cfg["linear_threshold"],
                    "sort_csr_ms": cfg["sort_csr_ms"],
                    **{field: stats[field] for field in fields if field in stats},
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare torch_cluster.random_walk and the custom adaptive CSR "
            "implementation on real-world power-law edge-list data."
        )
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="twitter_combined")
    parser.add_argument("--url", default=None, help="Override the dataset download URL.")
    parser.add_argument("--edge-list", default=None, help="Use a local edge-list file instead.")
    parser.add_argument("--graph-name", default=None)
    parser.add_argument("--data-dir", default="data/real_powerlaw")
    parser.add_argument("--force-download", action="store_true")

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-starts", type=positive_int, default=262144)
    parser.add_argument("--walk-length", type=positive_int, default=20)
    parser.add_argument("--start-mode", choices=("random", "range"), default="random")
    parser.add_argument("--pq", type=parse_pq_pairs, default=parse_pq_pairs("1,1;0.25,4;1,2;4,0.25"))
    parser.add_argument("--linear-threshold", type=nonnegative_int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=positive_int, default=50)
    parser.add_argument("--modes", nargs="+", choices=("library", "adaptive"), default=("library", "adaptive"))

    parser.add_argument(
        "--directed",
        action="store_true",
        help="Keep the raw edge direction. By default, reciprocal edges are added.",
    )
    parser.add_argument("--keep-self-loops", action="store_true")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument(
        "--max-raw-edges",
        type=positive_int,
        default=None,
        help="Optional smoke-test limit on raw edges read before symmetrization.",
    )

    parser.add_argument("--extension-name", default="binary_search_random_walk_ext")
    parser.add_argument("--extension-verbose", action="store_true")
    parser.add_argument("--json", default="results/compare_real_powerlaw_random_walk.json")
    parser.add_argument("--csv", default="results/compare_real_powerlaw_random_walk.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required to run this benchmark.") from exc

    device = make_device(torch, args.device)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    env = environment_info(torch, device)
    print("Environment:")
    print(json.dumps(env, indent=2, sort_keys=True))

    edge_list_path, dataset_source, graph_name = resolve_edge_list(args)
    make_undirected = not bool(args.directed)

    print("\nLoading graph %s from %s" % (graph_name, edge_list_path))
    graph_start = time.perf_counter()
    graph = build_real_graph(
        torch=torch,
        path=edge_list_path,
        device=device,
        make_undirected=make_undirected,
        keep_self_loops=bool(args.keep_self_loops),
        keep_duplicates=bool(args.keep_duplicates),
        max_raw_edges=args.max_raw_edges,
    )
    graph_build_seconds = time.perf_counter() - graph_start

    module = load_extension(
        name=args.extension_name,
        verbose=bool(args.extension_verbose),
    )
    sorted_col, sort_csr_ms = time_cuda_once(
        torch, device, lambda: module.sort_csr_col(graph["rowptr"], graph["col"])
    )
    sorted_col = sorted_col.contiguous()
    synchronize(torch, device)

    print(
        "Graph %s: nodes=%d edges=%d avg_degree=%.3f max_degree=%d p99_degree=%.1f build=%.3fs sort=%.3fms"
        % (
            graph_name,
            graph["num_nodes"],
            graph["num_edges"],
            graph["avg_degree"],
            graph["degree_max"],
            graph["degree_p99"],
            graph_build_seconds,
            sort_csr_ms,
        )
    )
    print("Parse stats:")
    print(json.dumps(graph["parse_stats"], indent=2, sort_keys=True))

    start = make_starts(
        torch,
        graph["num_nodes"],
        int(args.num_starts),
        device,
        args.start_mode,
        int(args.seed) + 17,
    )
    synchronize(torch, device)

    all_results: List[Dict[str, Any]] = []
    for pq_index, (p, q) in enumerate(args.pq):
        callables: Dict[str, Any] = {}
        if "library" in args.modes:
            callables["library"] = make_library_callable(
                torch=torch,
                rowptr=graph["rowptr"],
                col=sorted_col,
                start=start,
                walk_length=int(args.walk_length),
                p=float(p),
                q=float(q),
                num_nodes=graph["num_nodes"],
            )
        if "adaptive" in args.modes:
            callables["adaptive"] = make_binary_search_callable(
                module=module,
                rowptr=graph["rowptr"],
                col=sorted_col,
                start=start,
                walk_length=int(args.walk_length),
                p=float(p),
                q=float(q),
                linear_threshold=int(args.linear_threshold),
                seed=int(args.seed) + pq_index * 37,
            )

        transitions = int(args.num_starts) * int(args.walk_length)
        for mode, fn in callables.items():
            cfg = {
                "graph_name": graph_name,
                "graph_type": graph["graph_type"],
                "dataset_source": dataset_source,
                "edge_list": str(edge_list_path),
                "make_undirected": make_undirected,
                "keep_duplicates": bool(args.keep_duplicates),
                "num_nodes": graph["num_nodes"],
                "num_edges": graph["num_edges"],
                "avg_degree": graph["avg_degree"],
                "degree_mean": graph["degree_mean"],
                "degree_max": graph["degree_max"],
                "degree_p50": graph["degree_p50"],
                "degree_p90": graph["degree_p90"],
                "degree_p95": graph["degree_p95"],
                "degree_p99": graph["degree_p99"],
                "num_starts": int(args.num_starts),
                "walk_length": int(args.walk_length),
                "p": float(p),
                "q": float(q),
                "linear_threshold": int(args.linear_threshold),
                "sort_csr_ms": sort_csr_ms,
                "graph_build_seconds": graph_build_seconds,
                "start_mode": args.start_mode,
            }
            print(
                "\nBenchmarking %-8s graph=%s p=%.4g q=%.4g"
                % (mode, graph_name, p, q)
            )
            stats, output = benchmark_callable(
                torch=torch,
                device=device,
                fn=fn,
                warmup=int(args.warmup),
                iters=int(args.iters),
            )
            p50_seconds = stats["latency_ms_p50"] / 1000.0
            stats["transitions_per_second_p50"] = transitions / p50_seconds
            stats["walks_per_second_p50"] = int(args.num_starts) / p50_seconds
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

    payload = {
        "environment": env,
        "args": {
            **vars(args),
            "pq": [[p, q] for p, q in args.pq],
        },
        "dataset_catalog": DATASETS,
        "graph": {k: v for k, v in graph.items() if k not in ("rowptr", "col")},
        "results": all_results,
    }
    if args.json:
        write_json(args.json, payload)
        print("\nWrote JSON: %s" % args.json)
    if args.csv:
        write_results_csv(args.csv, all_results)
        print("Wrote CSV: %s" % args.csv)


if __name__ == "__main__":
    main()
