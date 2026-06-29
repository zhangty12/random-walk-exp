#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <curand_kernel.h>
#include <torch/types.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;

__global__ void fill_edge_rows_kernel(
    const int64_t* __restrict__ rowptr,
    int64_t* __restrict__ edge_rows,
    int64_t num_nodes) {
    for (int64_t v = blockIdx.x * blockDim.x + threadIdx.x; v < num_nodes;
         v += blockDim.x * gridDim.x) {
        const int64_t begin = rowptr[v];
        const int64_t end = rowptr[v + 1];
        for (int64_t e = begin; e < end; ++e) {
            edge_rows[e] = v;
        }
    }
}

struct RowColLess {
    template <typename Tuple>
    __host__ __device__ bool operator()(const Tuple& lhs, const Tuple& rhs) const {
        const int64_t lhs_row = thrust::get<0>(lhs);
        const int64_t rhs_row = thrust::get<0>(rhs);
        if (lhs_row < rhs_row) {
            return true;
        }
        if (rhs_row < lhs_row) {
            return false;
        }
        return thrust::get<1>(lhs) < thrust::get<1>(rhs);
    }
};

__device__ __forceinline__ int64_t sample_edge(
    curandStatePhilox4_32_10_t* state,
    int64_t begin,
    int64_t degree) {
    return begin + static_cast<int64_t>(curand(state) % static_cast<unsigned long long>(degree));
}

__device__ __forceinline__ bool row_contains_linear(
    const int64_t* __restrict__ rowptr,
    const int64_t* __restrict__ col,
    int64_t row_vertex,
    int64_t target) {
    const int64_t begin = rowptr[row_vertex];
    const int64_t end = rowptr[row_vertex + 1];

    for (int64_t e = begin; e < end; ++e) {
        if (col[e] == target) {
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ bool row_contains_binary(
    const int64_t* __restrict__ rowptr,
    const int64_t* __restrict__ col,
    int64_t row_vertex,
    int64_t target) {
    const int64_t begin = rowptr[row_vertex];
    const int64_t end = rowptr[row_vertex + 1];

    int64_t lo = begin;
    int64_t hi = end;
    while (lo < hi) {
        const int64_t mid = lo + ((hi - lo) >> 1);
        const int64_t value = col[mid];
        if (value < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo < end && col[lo] == target;
}

__device__ __forceinline__ double transition_weight(
    int64_t candidate,
    int64_t previous,
    bool is_previous_neighbor,
    double unnormalized_return,
    double unnormalized_neighbor,
    double unnormalized_outward) {
    if (candidate == previous) {
        return unnormalized_return;
    }
    return is_previous_neighbor ? unnormalized_neighbor : unnormalized_outward;
}

__device__ __forceinline__ int64_t weighted_sample_with_binary_search(
    const int64_t* __restrict__ rowptr,
    const int64_t* __restrict__ col,
    curandStatePhilox4_32_10_t* state,
    int64_t current_begin,
    int64_t current_end,
    int64_t previous,
    double unnormalized_return,
    double unnormalized_neighbor,
    double unnormalized_outward) {
    int64_t selected = current_begin;
    double total_weight = 0.0;
    for (int64_t e = current_begin; e < current_end; ++e) {
        const int64_t candidate = col[e];
        const bool is_previous_neighbor =
            candidate != previous && row_contains_binary(rowptr, col, previous, candidate);
        const double weight = transition_weight(
            candidate,
            previous,
            is_previous_neighbor,
            unnormalized_return,
            unnormalized_neighbor,
            unnormalized_outward);
        total_weight += weight;
        if (static_cast<double>(curand_uniform(state)) * total_weight <= weight) {
            selected = e;
        }
    }
    return selected;
}

__device__ __forceinline__ int64_t weighted_sample_with_merge(
    const int64_t* __restrict__ rowptr,
    const int64_t* __restrict__ col,
    curandStatePhilox4_32_10_t* state,
    int64_t current_begin,
    int64_t current_end,
    int64_t previous,
    double unnormalized_return,
    double unnormalized_neighbor,
    double unnormalized_outward) {
    const int64_t previous_begin = rowptr[previous];
    const int64_t previous_end = rowptr[previous + 1];

    int64_t selected = current_begin;
    double total_weight = 0.0;
    int64_t previous_pos = previous_begin;
    for (int64_t e = current_begin; e < current_end; ++e) {
        const int64_t candidate = col[e];
        while (previous_pos < previous_end && col[previous_pos] < candidate) {
            ++previous_pos;
        }
        const bool is_previous_neighbor =
            candidate != previous && previous_pos < previous_end && col[previous_pos] == candidate;
        const double weight = transition_weight(
            candidate,
            previous,
            is_previous_neighbor,
            unnormalized_return,
            unnormalized_neighbor,
            unnormalized_outward);
        total_weight += weight;
        if (static_cast<double>(curand_uniform(state)) * total_weight <= weight) {
            selected = e;
        }
    }
    return selected;
}

__global__ void random_walk_uniform_kernel(
    const int64_t* __restrict__ rowptr,
    const int64_t* __restrict__ col,
    const int64_t* __restrict__ start,
    int64_t* __restrict__ node_out,
    int64_t* __restrict__ edge_out,
    int64_t num_walks,
    int64_t walk_length,
    unsigned long long seed) {
    const int64_t walk_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_id >= num_walks) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(walk_id), 0, &state);

    int64_t current = start[walk_id];
    node_out[walk_id * (walk_length + 1)] = current;

    for (int64_t step = 0; step < walk_length; ++step) {
        const int64_t begin = rowptr[current];
        const int64_t end = rowptr[current + 1];
        const int64_t degree = end - begin;

        int64_t next = current;
        int64_t edge = -1;
        if (degree > 0) {
            edge = sample_edge(&state, begin, degree);
            next = col[edge];
        }

        edge_out[walk_id * walk_length + step] = edge;
        node_out[walk_id * (walk_length + 1) + step + 1] = next;
        current = next;
    }
}

__global__ void random_walk_node2vec_kernel(
    const int64_t* __restrict__ rowptr,
    const int64_t* __restrict__ col,
    const int64_t* __restrict__ start,
    int64_t* __restrict__ node_out,
    int64_t* __restrict__ edge_out,
    int64_t num_walks,
    int64_t walk_length,
    double p,
    double q,
    int64_t linear_threshold,
    unsigned long long seed) {
    const int64_t walk_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (walk_id >= num_walks) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(walk_id), 0, &state);

    const double unnormalized_return = 1.0 / p;
    const double unnormalized_neighbor = 1.0;
    const double unnormalized_outward = 1.0 / q;
    const double max_weight =
        fmax(unnormalized_neighbor, fmax(unnormalized_return, unnormalized_outward));
    const double accept_return = unnormalized_return / max_weight;
    const double accept_neighbor = unnormalized_neighbor / max_weight;
    const double accept_outward = unnormalized_outward / max_weight;

    int64_t previous = start[walk_id];
    int64_t current = previous;
    node_out[walk_id * (walk_length + 1)] = current;

    if (walk_length <= 0) {
        return;
    }

    int64_t begin = rowptr[current];
    int64_t end = rowptr[current + 1];
    int64_t degree = end - begin;

    int64_t edge = -1;
    int64_t next = current;
    if (degree > 0) {
        edge = sample_edge(&state, begin, degree);
        next = col[edge];
    }

    edge_out[walk_id * walk_length] = edge;
    node_out[walk_id * (walk_length + 1) + 1] = next;
    previous = current;
    current = next;

    for (int64_t step = 1; step < walk_length; ++step) {
        begin = rowptr[current];
        end = rowptr[current + 1];
        degree = end - begin;
        const int64_t previous_degree = rowptr[previous + 1] - rowptr[previous];

        if (degree == 0) {
            edge = -1;
            next = current;
        } else if (degree == 1) {
            edge = begin;
            next = col[edge];
        } else if (degree > linear_threshold) {
            while (true) {
                edge = sample_edge(&state, begin, degree);
                next = col[edge];

                double accept_probability = accept_outward;
                if (next == previous) {
                    accept_probability = accept_return;
                } else {
                    const bool is_previous_neighbor =
                        previous_degree > linear_threshold
                            ? row_contains_binary(rowptr, col, previous, next)
                            : row_contains_linear(rowptr, col, previous, next);
                    if (is_previous_neighbor) {
                        accept_probability = accept_neighbor;
                    }
                }

                const double draw = static_cast<double>(curand_uniform(&state));
                if (draw <= accept_probability) {
                    break;
                }
            }
        } else if (previous_degree > linear_threshold) {
            edge = weighted_sample_with_binary_search(
                rowptr,
                col,
                &state,
                begin,
                end,
                previous,
                unnormalized_return,
                unnormalized_neighbor,
                unnormalized_outward);
            next = col[edge];
        } else {
            edge = weighted_sample_with_merge(
                rowptr,
                col,
                &state,
                begin,
                end,
                previous,
                unnormalized_return,
                unnormalized_neighbor,
                unnormalized_outward);
            next = col[edge];
        }

        edge_out[walk_id * walk_length + step] = edge;
        node_out[walk_id * (walk_length + 1) + step + 1] = next;
        previous = current;
        current = next;
    }
}

}  // namespace

torch::Tensor sort_csr_col_cuda(torch::Tensor rowptr, torch::Tensor col) {
    const c10::cuda::CUDAGuard device_guard(rowptr.device());
    auto sorted_col = col.clone();
    const int64_t num_nodes = rowptr.numel() - 1;
    const int64_t num_edges = col.numel();
    if (num_nodes == 0 || num_edges == 0) {
        return sorted_col;
    }

    auto edge_rows = at::empty_like(sorted_col);
    const int64_t blocks = std::min<int64_t>((num_nodes + kThreads - 1) / kThreads, 4096);
    auto stream = at::cuda::getCurrentCUDAStream();
    fill_edge_rows_kernel<<<blocks, kThreads, 0, stream.stream()>>>(
        rowptr.data_ptr<int64_t>(),
        edge_rows.data_ptr<int64_t>(),
        num_nodes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto row_begin = thrust::device_pointer_cast(edge_rows.data_ptr<int64_t>());
    auto col_begin = thrust::device_pointer_cast(sorted_col.data_ptr<int64_t>());
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(row_begin, col_begin));
    auto zip_end = zip_begin + num_edges;
    thrust::sort(thrust::cuda::par.on(stream.stream()), zip_begin, zip_end, RowColLess());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sorted_col;
}

std::vector<torch::Tensor> random_walk_binary_search_cuda(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor start,
    int64_t walk_length,
    double p,
    double q,
    int64_t linear_threshold,
    int64_t seed) {
    const c10::cuda::CUDAGuard device_guard(rowptr.device());
    const int64_t num_walks = start.numel();
    auto node_out = at::empty({num_walks, walk_length + 1}, start.options());
    auto edge_out = at::empty({num_walks, walk_length}, start.options());

    if (num_walks == 0) {
        return {node_out, edge_out};
    }

    const int64_t blocks = (num_walks + kThreads - 1) / kThreads;
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto kernel_seed = static_cast<unsigned long long>(seed);

    if (p == 1.0 && q == 1.0) {
        random_walk_uniform_kernel<<<blocks, kThreads, 0, stream.stream()>>>(
            rowptr.data_ptr<int64_t>(),
            col.data_ptr<int64_t>(),
            start.data_ptr<int64_t>(),
            node_out.data_ptr<int64_t>(),
            edge_out.data_ptr<int64_t>(),
            num_walks,
            walk_length,
            kernel_seed);
    } else {
        random_walk_node2vec_kernel<<<blocks, kThreads, 0, stream.stream()>>>(
            rowptr.data_ptr<int64_t>(),
            col.data_ptr<int64_t>(),
            start.data_ptr<int64_t>(),
            node_out.data_ptr<int64_t>(),
            edge_out.data_ptr<int64_t>(),
            num_walks,
            walk_length,
            p,
            q,
            linear_threshold,
            kernel_seed);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {node_out, edge_out};
}
