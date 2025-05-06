// crop_and_resize.cpp
// Refactored 2D Crop & Resize CPU implementation
// Uses ATen/C++ API and torch::extension

#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <stdexcept>

// Per‐box crop & resize helper (unchanged core logic)
static void CropAndResizePerBox(
    const float * image_data,
    int batch_size,
    int depth,
    int image_height,
    int image_width,

    const float * boxes_data,
    const int * box_index_data,
    int start_box,
    int limit_box,

    float * crops_data,
    int crop_height,
    int crop_width,
    float extrapolation_value
) {
    const int image_channel_sz = image_height * image_width;
    const int image_batch_sz   = depth * image_channel_sz;

    const int channel_sz = crop_height * crop_width;
    const int crop_sz    = depth * channel_sz;

    #pragma omp parallel for
    for (int b = start_box; b < limit_box; ++b) {
        const float * box = boxes_data + b * 4;
        float y1 = box[0], x1 = box[1], y2 = box[2], x2 = box[3];
        int batch_index = box_index_data[b];
        if (batch_index < 0 || batch_index >= batch_size) {
            throw std::out_of_range("box_index out of range");
        }

        float height_scale = (crop_height > 1)
            ? (y2 - y1) * (image_height - 1) / (crop_height - 1)
            : 0.f;
        float width_scale = (crop_width > 1)
            ? (x2 - x1) * (image_width  - 1) / (crop_width  - 1)
            : 0.f;

        for (int y = 0; y < crop_height; ++y) {
            float in_y = (crop_height > 1)
                ? y1 * (image_height - 1) + y * height_scale
                : 0.5f * (y1 + y2) * (image_height - 1);

            bool y_oob = (in_y < 0.f || in_y > image_height - 1);
            int top_y    = static_cast<int>(std::floor(in_y));
            int bottom_y = static_cast<int>(std::ceil(in_y));
            float y_lerp = in_y - top_y;

            for (int x = 0; x < crop_width; ++x) {
                float in_x = (crop_width > 1)
                    ? x1 * (image_width - 1) + x * width_scale
                    : 0.5f * (x1 + x2) * (image_width - 1);

                bool x_oob = (in_x < 0.f || in_x > image_width - 1);
                int left_x    = static_cast<int>(std::floor(in_x));
                int right_x   = static_cast<int>(std::ceil(in_x));
                float x_lerp  = in_x - left_x;

                for (int d = 0; d < depth; ++d) {
                    float v = extrapolation_value;
                    if (!y_oob && !x_oob) {
                        const float * src = image_data
                            + batch_index * image_batch_sz
                            + d * image_channel_sz;
                        float tl = src[top_y    * image_width + left_x];
                        float tr = src[top_y    * image_width + right_x];
                        float bl = src[bottom_y * image_width + left_x];
                        float br = src[bottom_y * image_width + right_x];
                        float top_val    = tl + (tr - tl) * x_lerp;
                        float bottom_val = bl + (br - bl) * x_lerp;
                        v = top_val + (bottom_val - top_val) * y_lerp;
                    }
                    // write to crops[b,d,y,x]
                    crops_data[crop_sz * b + channel_sz * d + y * crop_width + x] = v;
                }
            }
        }
    }
}

// Forward wrapper
at::Tensor crop_and_resize_forward(
    at::Tensor image,         // [B, D, H, W], float, CPU
    at::Tensor boxes,         // [N, 4], float, CPU
    at::Tensor box_index,     // [N],   int64 (or int32), CPU
    float extrapolation_value,
    int crop_height,
    int crop_width
) {
    TORCH_CHECK(image.device().is_cpu(),   "image must be a CPU tensor");
    TORCH_CHECK(boxes.device().is_cpu(),   "boxes must be a CPU tensor");
    TORCH_CHECK(box_index.device().is_cpu(),"box_index must be a CPU tensor");
    image = image.contiguous();
    boxes = boxes.contiguous();
    box_index = box_index.contiguous().to(torch::kInt32);

    int batch_size  = image.size(0);
    int depth       = image.size(1);
    int image_h     = image.size(2);
    int image_w     = image.size(3);
    int num_boxes   = boxes.size(0);

    auto crops = torch::empty(
        {num_boxes, depth, crop_height, crop_width},
        image.options()
    );
    float * image_ptr     = image.data_ptr<float>();
    float * boxes_ptr     = boxes.data_ptr<float>();
    int   * index_ptr     = box_index.data_ptr<int>();
    float * crops_ptr     = crops.data_ptr<float>();

    // zero init
    std::fill_n(crops_ptr, num_boxes * depth * crop_height * crop_width, 0.f);

    CropAndResizePerBox(
        image_ptr,
        batch_size,
        depth,
        image_h,
        image_w,
        boxes_ptr,
        index_ptr,
        0,
        num_boxes,
        crops_ptr,
        crop_height,
        crop_width,
        extrapolation_value
    );

    return crops;
}

// Backward wrapper
at::Tensor crop_and_resize_backward(
    at::Tensor grads,        // [N, D, Hc, Wc], float, CPU
    at::Tensor boxes,        // [N, 4], float, CPU
    at::Tensor box_index,    // [N],   int64/int32, CPU
    int batch_size,
    int image_h,
    int image_w
) {
    TORCH_CHECK(grads.device().is_cpu(),     "grads must be a CPU tensor");
    TORCH_CHECK(boxes.device().is_cpu(),     "boxes must be a CPU tensor");
    TORCH_CHECK(box_index.device().is_cpu(), "box_index must be a CPU tensor");
    grads     = grads.contiguous();
    boxes     = boxes.contiguous();
    box_index = box_index.contiguous().to(torch::kInt32);

    int depth       = grads.size(1);
    int crop_h      = grads.size(2);
    int crop_w      = grads.size(3);
    int num_boxes   = grads.size(0);

    // allocate gradients w.r.t. image: zeros
    auto grads_image = torch::zeros(
        {batch_size, depth, image_h, image_w},
        grads.options()
    );

    // pointers
    const float * grads_ptr       = grads.data_ptr<float>();
    const float * boxes_ptr       = boxes.data_ptr<float>();
    const int   * index_ptr       = box_index.data_ptr<int>();
    float       * grads_image_ptr = grads_image.data_ptr<float>();

    const int image_channel_sz = image_h * image_w;
    const int image_batch_sz   = depth * image_channel_sz;
    const int channel_sz       = crop_h * crop_w;
    const int crop_sz          = depth * channel_sz;

    for (int b = 0; b < num_boxes; ++b) {
        const float * box = boxes_ptr + b * 4;
        float y1 = box[0], x1 = box[1], y2 = box[2], x2 = box[3];
        int batch_idx = index_ptr[b];
        if (batch_idx < 0 || batch_idx >= batch_size) {
            throw std::out_of_range("box_index out of range");
        }

        float height_scale = (crop_h > 1)
            ? (y2 - y1) * (image_h - 1) / (crop_h - 1)
            : 0.f;
        float width_scale = (crop_w > 1)
            ? (x2 - x1) * (image_w - 1) / (crop_w - 1)
            : 0.f;

        for (int y = 0; y < crop_h; ++y) {
            float in_y = (crop_h > 1)
                ? y1 * (image_h - 1) + y * height_scale
                : 0.5f * (y1 + y2) * (image_h - 1);
            if (in_y < 0 || in_y > image_h - 1) continue;
            int top_y    = static_cast<int>(std::floor(in_y));
            int bottom_y = static_cast<int>(std::ceil(in_y));
            float y_lerp = in_y - top_y;

            for (int x = 0; x < crop_w; ++x) {
                float in_x = (crop_w > 1)
                    ? x1 * (image_w - 1) + x * width_scale
                    : 0.5f * (x1 + x2) * (image_w - 1);
                if (in_x < 0 || in_x > image_w - 1) continue;
                int left_x  = static_cast<int>(std::floor(in_x));
                int right_x = static_cast<int>(std::ceil(in_x));
                float x_lerp= in_x - left_x;

                for (int d = 0; d < depth; ++d) {
                    float g = grads_ptr[crop_sz * b + channel_sz * d + y * crop_w + x];
                    float * im_ptr = grads_image_ptr
                        + batch_idx * image_batch_sz
                        + d * image_channel_sz;

                    float dtop    = (1 - y_lerp) * g;
                    im_ptr[top_y * image_w + left_x]  += (1 - x_lerp) * dtop;
                    im_ptr[top_y * image_w + right_x] += x_lerp * dtop;

                    float dbottom = y_lerp * g;
                    im_ptr[bottom_y * image_w + left_x]  += (1 - x_lerp) * dbottom;
                    im_ptr[bottom_y * image_w + right_x] += x_lerp * dbottom;
                }
            }
        }
    }

    return grads_image;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("crop_and_resize_forward", &crop_and_resize_forward,
          "Crop and resize forward (CPU)",
          py::arg("image"), py::arg("boxes"), py::arg("box_index"),
          py::arg("extrapolation_value"), py::arg("crop_height"), py::arg("crop_width"));

    m.def("crop_and_resize_backward", &crop_and_resize_backward,
          "Crop and resize backward (CPU)",
          py::arg("grads"), py::arg("boxes"), py::arg("box_index"),
          py::arg("batch_size"), py::arg("image_height"), py::arg("image_width"));
}
