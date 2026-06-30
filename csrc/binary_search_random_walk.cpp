#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

void sort_csr_col_cuda_launcher(
    const int64_t* rowptr,
    int64_t* sorted_col,
    int64_t* edge_rows,
    int64_t num_nodes,
    int64_t num_edges,
    cudaStream_t stream);

void random_walk_binary_search_cuda_launcher(
    const int64_t* rowptr,
    const int64_t* col,
    const int64_t* start,
    int64_t* node_out,
    int64_t* edge_out,
    int64_t num_walks,
    int64_t walk_length,
    double p,
    double q,
    int64_t linear_threshold,
    unsigned long long seed,
    cudaStream_t stream);

torch::Tensor sort_csr_col_cuda(torch::Tensor rowptr, torch::Tensor col) {
    const c10::cuda::CUDAGuard device_guard(rowptr.device());
    auto sorted_col = col.clone();
    const int64_t num_nodes = rowptr.numel() - 1;
    const int64_t num_edges = col.numel();
    if (num_nodes == 0 || num_edges == 0) {
        return sorted_col;
    }

    auto edge_rows = at::empty_like(sorted_col);
    auto stream = at::cuda::getCurrentCUDAStream();
    sort_csr_col_cuda_launcher(
        rowptr.data_ptr<int64_t>(),
        sorted_col.data_ptr<int64_t>(),
        edge_rows.data_ptr<int64_t>(),
        num_nodes,
        num_edges,
        stream.stream());
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

    auto stream = at::cuda::getCurrentCUDAStream();
    random_walk_binary_search_cuda_launcher(
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
        static_cast<unsigned long long>(seed),
        stream.stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {node_out, edge_out};
}

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_LONG(x) TORCH_CHECK((x).scalar_type() == torch::kLong, #x " must be torch.long")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x); \
    CHECK_LONG(x)

torch::Tensor sort_csr_col(torch::Tensor rowptr, torch::Tensor col) {
    CHECK_INPUT(rowptr);
    CHECK_INPUT(col);
    TORCH_CHECK(rowptr.dim() == 1, "rowptr must be 1-D");
    TORCH_CHECK(col.dim() == 1, "col must be 1-D");
    TORCH_CHECK(rowptr.numel() >= 1, "rowptr must contain at least one entry");
    return sort_csr_col_cuda(rowptr, col);
}

std::vector<torch::Tensor> random_walk(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor start,
    int64_t walk_length,
    double p,
    double q,
    int64_t linear_threshold,
    int64_t seed) {
    CHECK_INPUT(rowptr);
    CHECK_INPUT(col);
    CHECK_INPUT(start);
    TORCH_CHECK(rowptr.dim() == 1, "rowptr must be 1-D");
    TORCH_CHECK(col.dim() == 1, "col must be 1-D");
    TORCH_CHECK(start.dim() == 1, "start must be 1-D");
    TORCH_CHECK(rowptr.numel() >= 1, "rowptr must contain at least one entry");
    TORCH_CHECK(walk_length >= 1, "walk_length must be >= 1");
    TORCH_CHECK(p > 0.0, "p must be positive");
    TORCH_CHECK(q > 0.0, "q must be positive");
    TORCH_CHECK(linear_threshold >= 0, "linear_threshold must be non-negative");
    return random_walk_binary_search_cuda(
        rowptr, col, start, walk_length, p, q, linear_threshold, seed);
}

}  // namespace

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "sort_csr_col",
        &sort_csr_col,
        "Sort CSR column entries independently inside each row (CUDA)");
    m.def(
        "random_walk",
        &random_walk,
        "node2vec random walk with threshold-adaptive sorted-CSR adjacency checks (CUDA)",
        py::arg("rowptr"),
        py::arg("col"),
        py::arg("start"),
        py::arg("walk_length"),
        py::arg("p"),
        py::arg("q"),
        py::arg("linear_threshold") = 32,
        py::arg("seed") = 12345);
}
