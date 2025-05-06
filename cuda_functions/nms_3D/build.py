# build.py for nms_3D extension
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# Paths
this_dir = os.path.dirname(os.path.realpath(__file__))
src_dir  = os.path.join(this_dir, 'src')
cuda_obj = os.path.join(src_dir, 'cuda', 'nms_kernel.cu.o')

# Common settings
include_dirs  = [src_dir]
define_macros = []
ext_modules    = []

# 1) CPU-only 3D NMS extension
cpu_ext = CppExtension(
    name='_ext.nms_3d_cpu',
    sources=[os.path.join(src_dir, 'nms_3d.cpp')],
    include_dirs=include_dirs,
    define_macros=define_macros,
    extra_compile_args=['-std=c++17'],
)
ext_modules.append(cpu_ext)

# 2) GPU-enabled 3D NMS extension
if torch.cuda.is_available():
    gpu_ext = CUDAExtension(
        name='_ext.nms_3d',
        sources=[os.path.join(src_dir, 'nms_3d_cuda.cpp')],
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_objects=[cuda_obj],
        extra_compile_args={
            'cxx': ['-std=c++17'],
            'nvcc': ['-arch=sm_120', '-Xcompiler', '-fPIC'],
        },
    )
    ext_modules.append(gpu_ext)

# setuptools setup
setup(
    name='nms3d',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
