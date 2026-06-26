#include <torch/extension.h>

#include <cstdint>
#include <vector>

torch::Tensor sort_csr_col_cuda(torch::Tensor rowptr, torch::Tensor col);

std::vector<torch::Tensor> random_walk_binary_search_cuda(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor start,
    int64_t walk_length,
    double p,
    double q,
    int64_t linear_threshold,
    int64_t seed);

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
        "node2vec random walk with sorted-CSR binary-search adjacency checks (CUDA)",
        py::arg("rowptr"),
        py::arg("col"),
        py::arg("start"),
        py::arg("walk_length"),
        py::arg("p"),
        py::arg("q"),
        py::arg("linear_threshold") = 32,
        py::arg("seed") = 12345);
}
