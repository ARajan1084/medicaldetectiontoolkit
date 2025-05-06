// nms_3d_cuda.cpp
// Refactored 3D Non-Maximum Suppression GPU implementation
// Uses ATen/C++ API and torch::extension

#include <torch/extension.h>
#include "cuda/nms_kernel.h"
#include <vector>

#define THREADS_PER_BLOCK 64

// boxes: [N,6] tensor on CUDA (x1,y1,x2,y2,z1,z2), sorted by score
// Returns (keep, num_out) on CPU
std::tuple<at::Tensor, at::Tensor> gpu_nms_3d(
    at::Tensor boxes,
    float nms_overlap_thresh
) {
    TORCH_CHECK(boxes.is_cuda(), "boxes must be a CUDA tensor");
    boxes = boxes.contiguous();
    const int64_t N = boxes.size(0);
    const int64_t D = boxes.size(1);
    TORCH_CHECK(D == 6, "boxes should have shape [N,6]");

    // compute number of bitmask blocks
    const int64_t col_blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // allocate mask on GPU (uint64_t blocks)
    auto mask = torch::empty({N, col_blocks}, boxes.options().dtype(torch::kUInt64));
    uint64_t* mask_ptr_raw = mask.data_ptr<uint64_t>();
    unsigned long long* mask_ptr = reinterpret_cast<unsigned long long*>(mask_ptr_raw);

    // launch CUDA kernel (_nms defined in nms_kernel.h/.cu)
    _nms(
        static_cast<int>(N),
        boxes.data_ptr<float>(),
        mask_ptr,
        nms_overlap_thresh
    );

    // move mask to CPU for bitwise scan
    auto mask_cpu = mask.cpu();
    auto mask_acc = mask_cpu.accessor<uint64_t,2>();

    // prepare outputs on CPU
    std::vector<unsigned long long> remv(col_blocks, 0ULL);
    at::Tensor keep = torch::empty({N}, torch::kLong);
    int64_t num_to_keep = 0;

    for (int64_t i = 0; i < N; ++i) {
        int64_t nblock  = i / THREADS_PER_BLOCK;
        int64_t inblock = i % THREADS_PER_BLOCK;
        if (!(remv[nblock] & (1ULL << inblock))) {
            keep[num_to_keep++] = i;
            // apply mask for this i
            for (int64_t j = nblock; j < col_blocks; ++j) {
                remv[j] |= mask_acc[i][j];
            }
        }
    }

    // finalize keep tensor
    keep = keep.slice(0, 0, num_to_keep);
    at::Tensor num_out = torch::full({}, num_to_keep, torch::kLong);

    return std::make_tuple(keep, num_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gpu_nms_3d",
        &gpu_nms_3d,
        "3D Non-Maximum Suppression (CUDA)",
        py::arg("boxes"),
        py::arg("nms_overlap_thresh")
    );
}
