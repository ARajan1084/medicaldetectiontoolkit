#include <torch/extension.h>
#include <algorithm>
#include <vector>

int cpu_nms(
    at::Tensor keep_out,
    at::Tensor num_out,
    at::Tensor boxes,
    at::Tensor order,
    at::Tensor areas,
    float nms_overlap_thresh
) {
    // Runtime checks
    TORCH_CHECK(keep_out.is_contiguous(), "keep_out must be contiguous");
    TORCH_CHECK(boxes.is_contiguous(),   "boxes must be contiguous");
    TORCH_CHECK(order.is_contiguous(),   "order must be contiguous");
    TORCH_CHECK(areas.is_contiguous(),   "areas must be contiguous");

    // Sizes
    const auto boxes_num = boxes.size(0);
    const auto boxes_dim = boxes.size(1);

    // Raw pointers
    auto keep_ptr  = keep_out.data_ptr<long>();
    auto num_ptr   = num_out.data_ptr<long>();
    auto boxes_ptr = boxes.data_ptr<float>();
    auto order_ptr = order.data_ptr<long>();
    auto areas_ptr = areas.data_ptr<float>();

    // A temporary mask of suppressed indices
    at::Tensor suppressed = torch::zeros({boxes_num}, torch::kUInt8);
    auto sup_ptr = suppressed.data_ptr<uint8_t>();

    long num_to_keep = 0;
    for (long _i = 0; _i < boxes_num; ++_i) {
        long i = order_ptr[_i];
        if (sup_ptr[i]) {
            continue;
        }
        keep_ptr[num_to_keep++] = i;

        // load box i
        float ix1   = boxes_ptr[i * boxes_dim + 0];
        float iy1   = boxes_ptr[i * boxes_dim + 1];
        float ix2   = boxes_ptr[i * boxes_dim + 2];
        float iy2   = boxes_ptr[i * boxes_dim + 3];
        float iarea = areas_ptr[i];

        // compare with all lower-scored boxes j
        for (long _j = _i + 1; _j < boxes_num; ++_j) {
            long j = order_ptr[_j];
            if (sup_ptr[j]) {
                continue;
            }
            float xx1 = std::max(ix1, boxes_ptr[j * boxes_dim + 0]);
            float yy1 = std::max(iy1, boxes_ptr[j * boxes_dim + 1]);
            float xx2 = std::min(ix2, boxes_ptr[j * boxes_dim + 2]);
            float yy2 = std::min(iy2, boxes_ptr[j * boxes_dim + 3]);

            float w = std::max(0.0f, xx2 - xx1 + 1.0f);
            float h = std::max(0.0f, yy2 - yy1 + 1.0f);
            float inter = w * h;
            float ovr = inter / (iarea + areas_ptr[j] - inter);

            if (ovr >= nms_overlap_thresh) {
                sup_ptr[j] = 1;
            }
        }
    }

    // write out the keep count
    *num_ptr = num_to_keep;
    return 1;
}

// Bind to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "cpu_nms",
        &cpu_nms,
        "Non-Maximum Suppression (CPU)",
        py::arg("keep_out"),
        py::arg("num_out"),
        py::arg("boxes"),
        py::arg("order"),
        py::arg("areas"),
        py::arg("nms_overlap_thresh")
    );
}