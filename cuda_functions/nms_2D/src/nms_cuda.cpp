#include <torch/extension.h>
#include "cuda/nms_kernel.h"
#include <vector>

#define THREADS_PER_BLOCK 64

// GPU NMS: boxes is a [N×4] CUDA float Tensor, sorted by score.
// Returns (keep, num_out) where
//  - keep is a CPU LongTensor of length num_out with the kept indices
//  - num_out is a CPU LongTensor scalar with the number kept
std::tuple<at::Tensor, at::Tensor> gpu_nms(
    at::Tensor boxes,
    float nms_overlap_thresh
) {
    TORCH_CHECK(boxes.device().is_cuda(), "boxes must be a CUDA tensor");
    boxes = boxes.contiguous();
    const int64_t boxes_num = boxes.size(0);
    const int64_t col_blocks = (boxes_num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Allocate mask on GPU: shape [boxes_num, col_blocks], dtype unsigned long long
    auto mask = torch::empty(
        {boxes_num, col_blocks},
        boxes.options().dtype(torch::kLong)  // will reinterpret bits as ull later
    );
    auto mask_ptr = reinterpret_cast<unsigned long long*>(mask.data_ptr<int64_t>());

    // Launch the CUDA kernel (_nms is provided by nms_kernel.h/.cu)
    _nms(
        static_cast<int>(boxes_num),
        boxes.data_ptr<float>(),
        mask_ptr,
        nms_overlap_thresh
    );

    // Bring mask back to CPU for bit-mask processing
    auto mask_cpu = mask.cpu().to(torch::kUInt64);
    auto mask_acc = mask_cpu.accessor<unsigned long long,2>();

    // Prepare output containers on CPU
    at::Tensor keep    = torch::empty({boxes_num}, torch::kLong);
    std::vector<unsigned long long> remv(col_blocks, 0ULL);
    int64_t num_to_keep = 0;

    for (int64_t i = 0; i < boxes_num; ++i) {
        int64_t nblock  = i / THREADS_PER_BLOCK;
        int64_t inblock = i % THREADS_PER_BLOCK;

        if (!(remv[nblock] & (1ULL << inblock))) {
            keep[num_to_keep++] = i;
            // mark all boxes that overlap with i
            for (int64_t j = nblock; j < col_blocks; ++j) {
                remv[j] |= mask_acc[i][j];
            }
        }
    }

    // Slice keep to actual length
    keep = keep.slice(/*dim=*/0, /*start=*/0, /*end=*/num_to_keep);

    // Return keep and the count
    at::Tensor num_out = torch::full({}, num_to_keep, torch::kLong);
    return std::make_tuple(keep, num_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gpu_nms",
        &gpu_nms,
        "Non-Maximum Suppression (CUDA)",
        py::arg("boxes"),
        py::arg("nms_overlap_thresh")
    );
}
