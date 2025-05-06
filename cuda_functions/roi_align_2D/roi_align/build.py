import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# Directories
this_dir = os.path.dirname(os.path.realpath(__file__))
src_dir  = os.path.join(this_dir, 'src')
cuda_obj = os.path.join(src_dir, 'cuda', 'crop_and_resize_kernel.cu.o')

# Common compile flags
cpu_compile_args = ['-fopenmp', '-std=c99']
nvcc_compile_args = ['-arch=sm_120', '-Xcompiler', '-fPIC']

ext_modules = []

# 1) CPU‐only extension
cpu_ext = CppExtension(
    name='_ext.crop_and_resize_cpu',
    sources=[os.path.join(src_dir, 'crop_and_resize.cpp')],
    extra_compile_args=cpu_compile_args,
)
ext_modules.append(cpu_ext)

# 2) CUDA extension (if available)
if torch.cuda.is_available():
    print('Including CUDA code.')
    gpu_ext = CUDAExtension(
        name='_ext.crop_and_resize',
        sources=[
            os.path.join(src_dir, 'crop_and_resize_gpu.cpp')
        ],
        extra_objects=[cuda_obj],
        extra_compile_args={
            'cxx': cpu_compile_args,
            'nvcc': nvcc_compile_args,
        },
    )
    ext_modules.append(gpu_ext)

# Setup
setup(
    name='crop_and_resize',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
