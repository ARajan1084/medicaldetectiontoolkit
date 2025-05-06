// crop_and_resize_3d_cuda.cpp
// Refactored 3D-volume ROI Align GPU implementation
// Uses ATen/C++ API and torch::extension

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>  // for getCurrentCUDAStream()
#include "cuda/crop_and_resize_kernel.h"

// Forward: launches CUDA kernel for 3D volumes
at::Tensor crop_and_resize_3d_gpu_forward(
    at::Tensor image,           // [B, C, H, W, Z], float, CUDA
    at::Tensor boxes,           // [N, 4], float, CUDA
    at::Tensor box_index,       // [N],   int64/int32, CUDA
    float extrapolation_value,
    int crop_height,
    int crop_width,
    int crop_zdepth
) {
    TORCH_CHECK(image.is_cuda(),     "image must be a CUDA tensor");
    TORCH_CHECK(boxes.is_cuda(),     "boxes must be a CUDA tensor");
    TORCH_CHECK(box_index.is_cuda(), "box_index must be a CUDA tensor");

    image     = image.contiguous();
    boxes     = boxes.contiguous();
    box_index = box_index.contiguous().to(torch::kInt32);

    int batch_size   = image.size(0);
    int depth        = image.size(1);
    int image_h      = image.size(2);
    int image_w      = image.size(3);
    int image_z      = image.size(4);
    int num_boxes    = boxes.size(0);

    // allocate output tensor [N, C, crop_h, crop_w, crop_z]
    auto crops = torch::zeros(
        {num_boxes, depth, crop_height, crop_width, crop_zdepth},
        image.options()
    );

    // get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // launch CUDA kernel
    CropAndResizeLaucher(
        image.data_ptr<float>(),
        boxes.data_ptr<float>(),
        box_index.data_ptr<int>(),
        num_boxes, batch_size, image_h, image_w, image_z,
        crop_height, crop_width, crop_zdepth, depth,
        extrapolation_value,
        crops.data_ptr<float>(),
        stream
    );
    return crops;
}

// Backward: launches CUDA backprop kernel for 3D volumes
at::Tensor crop_and_resize_3d_gpu_backward(
    at::Tensor grads,           // [N, C, crop_h, crop_w, crop_z], float, CUDA
    at::Tensor boxes,           // [N, 4], float, CUDA
    at::Tensor box_index,       // [N],   int64/int32, CUDA
    int batch_size,
    int image_h,
    int image_w,
    int image_z
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
    int crop_z      = grads.size(4);

    // allocate gradient w.r.t. image: zeros
    auto grads_image = torch::zeros(
        {batch_size, depth, image_h, image_w, image_z},
        grads.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // launch backprop kernel
    CropAndResizeBackpropImageLaucher(
        grads.data_ptr<float>(),
        boxes.data_ptr<float>(),
        box_index.data_ptr<int>(),
        num_boxes, batch_size, image_h, image_w, image_z,
        crop_h, crop_w, crop_z, depth,
        grads_image.data_ptr<float>(),
        stream
    );
    return grads_image;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "crop_and_resize_3d_gpu_forward",
        &crop_and_resize_3d_gpu_forward,
        "3D ROI Align forward (CUDA)",
        py::arg("image"), py::arg("boxes"), py::arg("box_index"),
        py::arg("extrapolation_value"), py::arg("crop_height"),
        py::arg("crop_width"), py::arg("crop_zdepth")
    );
    m.def(
        "crop_and_resize_3d_gpu_backward",
        &crop_and_resize_3d_gpu_backward,
        "3D ROI Align backward (CUDA)",
        py::arg("grads"), py::arg("boxes"), py::arg("box_index"),
        py::arg("batch_size"), py::arg("image_height"),
        py::arg("image_width"), py::arg("image_zdepth")
    );
}
