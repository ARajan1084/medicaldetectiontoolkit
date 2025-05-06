import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# Directories
this_dir = os.path.dirname(os.path.realpath(__file__))
src_dir  = os.path.join(this_dir, 'src')
cuda_obj = os.path.join(src_dir, 'cuda', 'crop_and_resize_kernel.cu.o')

defines = []
ext_modules = []

# CPU-only 3D Crop & Resize extension
cpu_ext = CppExtension(
    name='_ext.crop_and_resize_3d_cpu',
    sources=[os.path.join(src_dir, 'crop_and_resize.cpp')],
    define_macros=defines,
    extra_compile_args=['-fopenmp', '-std=c99'],
)
ext_modules.append(cpu_ext)

# GPU-enabled 3D Crop & Resize extension
if torch.cuda.is_available():
    print('Including CUDA code.')
    defines += [('WITH_CUDA', None)]
    gpu_ext = CUDAExtension(
        name='_ext.crop_and_resize_3d',
        sources=[os.path.join(src_dir, 'crop_and_resize_gpu.cpp')],
        define_macros=defines,
        extra_objects=[cuda_obj],
        extra_compile_args={
            'cxx': ['-fopenmp', '-std=c99'],
            'nvcc': ['-arch=sm_120', '-Xcompiler', '-fPIC'],
        },
    )
    ext_modules.append(gpu_ext)

# Setup call
tsetup = setup(
    name='crop_and_resize_3d',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
