from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension 

setup(
    name='mulsoftmax',
    ext_modules=[
        CUDAExtension('mulsoftmax', [
            'mulsoftmax.cpp',
            'mulsoftmax_kernel.cu',
        ])        
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
