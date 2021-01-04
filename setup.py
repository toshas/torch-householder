#!/usr/bin/env python
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


long_description = """
This package implements the Householder transformation algorithm for calculating orthogonal matrices and Stiefel frames 
with differentiable bindings to PyTorch. In particular, the package provides an enhanced drop-in replacement for the 
`torch.orgqr` function. 

APIs for orthogonal transformations have been around since LAPACK; however, their support in the deep learning 
frameworks is lacking. Recently, orthogonal constraints have become popular in deep learning as a way to regularize
models and improve training dynamics, and hence the need to backpropagate through orthogonal transformations arised.

PyTorch 1.7 implements matrix exponential function `torch.matrix_exp`, which can be repurposed to performing the 
orthogonal transformation when the input matrix is skew-symmetric. This is the baseline we use in Speed and Precision 
evaluation.   

Compared to `torch.matrix_exp`, the Householder transformation implemented in this package has the following advantages: 
- Orders of magnitude lower memory footprint
- Ability to transform non-square matrices (Stiefel frames)
- A significant speed-up for non-square matrices
- Better numerical precision for all matrix and batch sizes

Find more details and the most up-to-date information on the project webpage:
https://www.github.com/toshas/torch-householder
"""


setup(
    name='torch_householder',
    version='1.0.0',
    description='Efficient Householder transformation in PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    python_requires='>=3.6',
    packages=find_packages(),
    author='Anton Obukhov',
    license='BSD',
    url='https://www.github.com/toshas/torch-householder',
    ext_modules=[CppExtension(
        'torch_householder_cpp', [os.path.join('torch_householder', 'householder.cpp')],
    )],
    cmdclass={
        'build_ext': BuildExtension
    },
    keywords=[
        'pytorch', 'householder', 'orgqr', 'efficient', 'differentiable',
        'orthogonal', 'transformation', 'unitary', 'matrices'
    ],
)
