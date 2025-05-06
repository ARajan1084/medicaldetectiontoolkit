// crop_and_resize_cuda.cpp
// Refactored 2D Crop & Resize GPU implementation
// Uses ATen/C++ API and torch::extension

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // for getCurrentCUDAStream()
#include "cuda/crop_and_resize_kernel.h"

// Forward: launches CUDA kernel, returns tensor of shape [N, C, Hc, Wc]
at::Tensor crop_and_resize_gpu_forward(
    at::Tensor image,           // [B, C, H, W] on CUDA
    at::Tensor boxes,           // [N, 4] (y1, x1, y2, x2) on CUDA
    at::Tensor box_index,       // [N] int32/64 on CUDA
    float extrapolation_value,
    int crop_height,
    int crop_width
) {
    TORCH_CHECK(image.is_cuda(),     "image must be a CUDA tensor");
    TORCH_CHECK(boxes.is_cuda(),     "boxes must be a CUDA tensor");
    TORCH_CHECK(box_index.is_cuda(), "box_index must be a CUDA tensor");

    image     = image.contiguous();
    boxes     = boxes.contiguous();
    box_index = box_index.contiguous().to(torch::kInt32);

    int batch_size  = image.size(0);
    int depth       = image.size(1);
    int image_h     = image.size(2);
    int image_w     = image.size(3);
    int num_boxes   = boxes.size(0);

    // allocate output
    auto crops = torch::zeros(
        {num_boxes, depth, crop_height, crop_width},
        image.options()
    );

    // current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // launch the CUDA kernel
    CropAndResizeLaucher(
        image.data_ptr<float>(),
        boxes.data_ptr<float>(),
        box_index.data_ptr<int>(),
        num_boxes, batch_size, image_h, image_w,
        crop_height, crop_width, depth, extrapolation_value,
        crops.data_ptr<float>(),
        stream
    );

    return crops;
}

// Backward: launches CUDA backprop kernel, returns dImage of shape [B, C, H, W]
at::Tensor crop_and_resize_gpu_backward(
    at::Tensor grads,         // [N, C, Hc, Wc] on CUDA
    at::Tensor boxes,         // [N, 4] on CUDA
    at::Tensor box_index,     // [N] on CUDA
    int batch_size,
    int image_h,
    int image_w
) {
    TORCH_CHECK(grads.is_cuda(),    "grads must be a CUDA tensor");
    TORCH_CHECK(boxes.is_cuda(),    "boxes must be a CUDA tensor");
    TORCH_CHECK(box_index.is_cuda(),"box_index must be a CUDA tensor");

    grads     = grads.contiguous();
    boxes     = boxes.contiguous();
    box_index = box_index.contiguous().to(torch::kInt32);

    int num_boxes   = grads.size(0);
    int depth       = grads.size(1);
    int crop_h      = grads.size(2);
    int crop_w      = grads.size(3);

    // allocate gradient w.r.t. image
    auto grads_image = torch::zeros(
        {batch_size, depth, image_h, image_w},
        grads.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // launch the CUDA backprop kernel
    CropAndResizeBackpropImageLaucher(
        grads.data_ptr<float>(),
        boxes.data_ptr<float>(),
        box_index.data_ptr<int>(),
        num_boxes, batch_size, image_h, image_w,
        crop_h, crop_w, depth,
        grads_image.data_ptr<float>(),
        stream
    );

    return grads_image;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "crop_and_resize_gpu_forward",
        &crop_and_resize_gpu_forward,
        "Crop and Resize forward (CUDA)",
        py::arg("image"),
        py::arg("boxes"),
        py::arg("box_index"),
        py::arg("extrapolation_value"),
        py::arg("crop_height"),
        py::arg("crop_width")
    );
    m.def(
        "crop_and_resize_gpu_backward",
        &crop_and_resize_gpu_backward,
        "Crop and Resize backward (CUDA)",
        py::arg("grads"),
        py::arg("boxes"),
        py::arg("box_index"),
        py::arg("batch_size"),
        py::arg("image_height"),
        py::arg("image_width")
    );
}
