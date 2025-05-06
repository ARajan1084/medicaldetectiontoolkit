// nms_3d.cpp
// Refactored 3D Non-Maximum Suppression CPU implementation
// Originally from nms.c, now using ATen/C++ API and torch::extension

#include <torch/extension.h>
#include <algorithm>
#include <vector>

int cpu_nms_3d(
    at::Tensor keep_out,      // LongTensor (CPU) for kept indices
    at::Tensor num_out,       // LongTensor (CPU) scalar for count
    at::Tensor boxes,         // FloatTensor (CPU) of shape [N, 6]: [x1, y1, x2, y2, z1, z2]
    at::Tensor order,         // LongTensor (CPU) of shape [N]
    at::Tensor areas,         // FloatTensor (CPU) of shape [N]
    float nms_overlap_thresh
) {
    // Contiguity checks
    TORCH_CHECK(keep_out.is_contiguous(), "keep_out must be contiguous");
    TORCH_CHECK(boxes.is_contiguous(),   "boxes must be contiguous");
    TORCH_CHECK(order.is_contiguous(),   "order must be contiguous");
    TORCH_CHECK(areas.is_contiguous(),   "areas must be contiguous");

    auto boxes_cont = boxes;
    const auto N = boxes_cont.size(0);
    const auto D = boxes_cont.size(1);
    TORCH_CHECK(D == 6, "boxes must have shape [N,6]");

    // Raw pointers
    auto keep_ptr   = keep_out.data_ptr<long>();
    auto num_ptr    = num_out.data_ptr<long>();
    auto boxes_ptr  = boxes_cont.data_ptr<float>();
    auto order_ptr  = order.data_ptr<long>();
    auto areas_ptr  = areas.data_ptr<float>();

    // suppressed flags
    at::Tensor suppressed = torch::zeros({N}, torch::kUInt8);
    auto sup_ptr = suppressed.data_ptr<uint8_t>();

    long keep_count = 0;
    for (long ii = 0; ii < N; ++ii) {
        long i = order_ptr[ii];
        if (sup_ptr[i])
            continue;
        keep_ptr[keep_count++] = i;

        // load box i coords
        float ix1 = boxes_ptr[i*6 + 0];
        float iy1 = boxes_ptr[i*6 + 1];
        float ix2 = boxes_ptr[i*6 + 2];
        float iy2 = boxes_ptr[i*6 + 3];
        float iz1 = boxes_ptr[i*6 + 4];
        float iz2 = boxes_ptr[i*6 + 5];
        float iarea = areas_ptr[i];

        for (long jj = ii + 1; jj < N; ++jj) {
            long j = order_ptr[jj];
            if (sup_ptr[j])
                continue;
            // intersection coords
            float xx1 = std::max(ix1, boxes_ptr[j*6 + 0]);
            float yy1 = std::max(iy1, boxes_ptr[j*6 + 1]);
            float xx2 = std::min(ix2, boxes_ptr[j*6 + 2]);
            float yy2 = std::min(iy2, boxes_ptr[j*6 + 3]);
            float zz1 = std::max(iz1, boxes_ptr[j*6 + 4]);
            float zz2 = std::min(iz2, boxes_ptr[j*6 + 5]);

            float w = std::max(0.0f, xx2 - xx1 + 1.0f);
            float h = std::max(0.0f, yy2 - yy1 + 1.0f);
            float d = std::max(0.0f, zz2 - zz1 + 1.0f);
            float inter = w * h * d;
            float ovr = inter / (iarea + areas_ptr[j] - inter);

            if (ovr >= nms_overlap_thresh)
                sup_ptr[j] = 1;
        }
    }

    // write out number kept
    *num_ptr = keep_count;
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_nms_3d", &cpu_nms_3d,
          "Non-Maximum Suppression 3D (CPU)",
          py::arg("keep_out"), py::arg("num_out"),
          py::arg("boxes"), py::arg("order"),
          py::arg("areas"), py::arg("nms_overlap_thresh"));
}
