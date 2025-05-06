# build.py
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

this_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.join(this_dir, 'src')
cuda_obj = os.path.join(src_dir, 'cuda', 'nms_kernel.cu.o')

include_dirs = [src_dir]
define_macros = []
ext_modules = []

# CPU-only NMS extension (always built)
cpu_extension = CppExtension(
    name='_ext.nms_cpu',
    sources=[os.path.join(src_dir, 'nms.cpp')],
    include_dirs=include_dirs,
    define_macros=define_macros,
    extra_compile_args=['-std=c++17'],
)
ext_modules.append(cpu_extension)

# GPU NMS extension (built when CUDA is available)
if torch.cuda.is_available():
    gpu_sources = [os.path.join(src_dir, 'nms_cuda.cpp')]
    gpu_extension = CUDAExtension(
        name='_ext.nms',
        sources=gpu_sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_objects=[cuda_obj],
        extra_compile_args={
            'cxx': ['-std=c++17'],
            'nvcc': ['-arch=sm_120', '-Xcompiler', '-fPIC'],
        },
    )
    ext_modules.append(gpu_extension)

# Final setup
tsetup = setup(
    name='nms',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
