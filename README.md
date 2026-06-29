# torch_cluster.random_walk A100 Micro Profiling

This directory contains a focused benchmark for `torch_cluster.random_walk` on
CUDA GPUs such as NVIDIA A100.

It also contains configurable comparison benchmarks for a custom CUDA/C++
node2vec random-walk implementation that uses sorted CSR adjacency lists and
threshold-adaptive second-order adjacency checks.

## Quick Start

Install PyTorch and `torch-cluster` wheels that match the CUDA runtime on the
A100 node, then run:

```bash
python3 profile_random_walk.py \
  --device cuda:0 \
  --num-nodes 1000000 \
  --degree 16 \
  --num-starts 1048576 \
  --walk-length 20 \
  --modes op wrapper_precoalesced \
  --warmup 20 \
  --iters 100 \
  --json results/a100_random_walk.json \
  --csv results/a100_random_walk.csv
```

The default graph is a synthetic fixed-out-degree graph. Each benchmark samples
`num_starts * walk_length` transitions per call and reports latency plus
transition throughput.

## Modes

`op`

Calls the low-level `torch.ops.torch_cluster.random_walk(rowptr, col, ...)`
operation with prebuilt CSR. This is the closest view of the underlying
random-walk op cost.

`wrapper_precoalesced`

Calls `torch_cluster.random_walk(row, col, ...)` with sorted edge lists and
`coalesced=False`. This includes the wrapper's rowptr/degree preparation, but
skips sorting.

`wrapper_coalesced`

Calls `torch_cluster.random_walk(row, col, ...)` with `coalesced=True`. This
includes sorting and rowptr preparation inside the wrapper.

## Sweeps

```bash
python3 profile_random_walk.py \
  --device cuda:0 \
  --sweep-degree 4,8,16,32 \
  --sweep-walk-length 10,20,40 \
  --num-nodes 1000000 \
  --num-starts 1048576 \
  --modes op \
  --csv results/sweep.csv
```

## Real Graph Input

You can pass a `.pt` file containing either an `edge_index` tensor with shape
`[2, num_edges]` or a dict with an `edge_index` key:

```bash
python3 profile_random_walk.py \
  --device cuda:0 \
  --edge-index data/edge_index.pt \
  --num-starts 1048576 \
  --walk-length 20 \
  --modes op wrapper_precoalesced
```

The input graph is sorted once before timing. That sorting cost is not included
in the benchmark.

## Kernel-Level Profiling

For Nsight Systems, enable NVTX ranges:

```bash
nsys profile -t cuda,nvtx -o results/rw_nsys \
  python3 profile_random_walk.py \
    --device cuda:0 \
    --modes op \
    --warmup 20 \
    --iters 200 \
    --nvtx
```

For PyTorch profiler Chrome traces:

```bash
python3 profile_random_walk.py \
  --device cuda:0 \
  --modes op wrapper_precoalesced \
  --trace-dir traces \
  --profile-steps 10
```

Open the emitted JSON trace in `chrome://tracing` or Perfetto.

## Practical Notes

- Run one benchmark process per GPU.
- Keep the GPU clocks and power state stable if comparing runs.
- Use `op` when you want the random-walk operation itself.
- Use `wrapper_precoalesced` or `wrapper_coalesced` when you want end-to-end
  Python API overhead.
- Increase `--num-starts` enough to make each timed call large relative to
  Python launch overhead.

## Adaptive CSR Implementation

The custom implementation lives in:

- `csrc/binary_search_random_walk.cpp`
- `csrc/binary_search_random_walk_cuda.cu`
- `custom_random_walk.py`

It accepts CSR inputs:

```text
rowptr: shape [num_nodes + 1], dtype torch.long, CUDA
col:    shape [num_edges], dtype torch.long, CUDA
start:  shape [num_walks], dtype torch.long, CUDA
```

The comparison script sorts `col[rowptr[x]:rowptr[x + 1]]` once for every
vertex, then benchmarks only the random-walk kernel. The sorting latency is
reported separately as `sort_csr_ms`.

Run the full configured comparison:

```bash
python3 compare_random_walk.py --config compare_random_walk_config.json
```

The config file controls graph families, graph sizes, concurrent walks per run,
`p/q` combinations, warmup/iteration counts, and output paths. The default
configuration tests reciprocal-edge random graphs with small average degree and
Barabasi-Albert-style power-law graphs:

```json
{
  "benchmark": {
    "modes": ["library", "adaptive"],
    "walk_length": 20,
    "linear_threshold": 32
  },
  "node2vec": [
    {"p": 1.0, "q": 1.0},
    {"p": 0.25, "q": 4.0},
    {"p": 1.0, "q": 2.0},
    {"p": 4.0, "q": 0.25}
  ]
}
```

For the node2vec path, `linear_threshold` is the degree cutoff `B`. Vertices
with degree greater than `B` are treated as long rows; all other positive-degree
vertices are treated as short rows.

- If both the current vertex `u` and previous vertex `t` are long, the kernel
  keeps rejection sampling and uses binary search in `t`'s sorted row.
- If `u` is long and `t` is short, the kernel keeps rejection sampling and scans
  `t`'s row linearly.
- If `u` is short and `t` is long, the kernel computes all candidate weights for
  neighbors of `u` with binary searches into `t`, then samples directly.
- If both rows are short, the kernel computes candidate weights with a merge
  over the two sorted rows, then samples directly.

The generated benchmark graphs include reciprocal edges so this adjacency check
is comparable with the existing `torch_cluster.random_walk` behavior on
undirected graph semantics.

## Real-World Power-Law Comparison

Use `compare_real_powerlaw_random_walk.py` to run the same library-vs-custom
comparison on edge-list data. By default it downloads the SNAP Twitter combined
ego-network edge list, adds reciprocal edges, removes duplicate directed edges,
sorts the CSR rows once, and benchmarks the library and adaptive kernels:

```bash
python3 compare_real_powerlaw_random_walk.py \
  --dataset twitter_combined \
  --device cuda:0 \
  --num-starts 262144 \
  --walk-length 20 \
  --linear-threshold 32 \
  --json results/twitter_random_walk.json \
  --csv results/twitter_random_walk.csv
```

To use another real graph, pass a whitespace-delimited local edge list:

```bash
python3 compare_real_powerlaw_random_walk.py \
  --edge-list data/my_graph.txt.gz \
  --graph-name my_graph \
  --device cuda:0
```
