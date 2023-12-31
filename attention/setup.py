from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension 

setup(
    name='attention',
    ext_modules=[
        CUDAExtension('attention', [
            'attention.cpp',
            'attention_kernel.cu',
        ])        
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
