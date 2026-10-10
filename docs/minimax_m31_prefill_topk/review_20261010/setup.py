from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
root = Path(__file__).resolve().parent
setup(name='m31_topk_review_probe', ext_modules=[CUDAExtension(
    'm31_topk_review_probe', sources=[str(root/'key_binding.cpp'), str(root/'key_probe.cu')],
    include_dirs=[str(root/'source')],
    extra_compile_args={'cxx':['-O3','-std=c++17'], 'nvcc':['-O3','-std=c++17','-ftz=true','-lineinfo','--ptxas-options=-v']})],
    cmdclass={'build_ext':BuildExtension})
