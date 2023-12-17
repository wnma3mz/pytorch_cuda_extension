from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension 

setup(
    name='mulsoft',
    ext_modules=[
        CUDAExtension('mulsoft', [
            'mulsoft.cpp',
            'mulsoft_kernel.cu',
        ])        
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
