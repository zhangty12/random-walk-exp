# torch_cluster.random_walk A100 Micro Profiling

This directory contains a focused benchmark for `torch_cluster.random_walk` on
CUDA GPUs such as NVIDIA A100.

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
