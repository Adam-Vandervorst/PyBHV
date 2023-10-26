from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os


dimension_arg = int(os.environ.get("DIMENSION", "8192"))
assert dimension_arg >= 512 and dimension_arg % 512 == 0

with open("bhv/dimension.py", "w") as f:
    f.write(f"DIMENSION = {dimension_arg}")

VERSION = '0.10.1'
DESCRIPTION = 'Boolean Hypervectors'
LONG_DESCRIPTION = 'Boolean Hypervectors with various operators for experiments in hyperdimensional computing (HDC).'

native = Extension("bhv.cnative",
                   sources=['bhv/cnative/bindings.cpp',
                            'bhv/cnative/TurboSHAKE_opt/TurboSHAKE.cpp',
                            'bhv/cnative/TurboSHAKE_opt/KeccakP-1600-opt64.cpp',
                            'bhv/cnative/TurboSHAKE_AVX512/TurboSHAKE.cpp',
                            'bhv/cnative/TurboSHAKE_AVX512/KeccakP-1600-AVX512.cpp',
                            ],
                   define_macros=[("DIMENSION", dimension_arg)],
                   include_dirs=['bhv/cnative', 'bhv/cnative/TurboSHAKEopt'],
                   extra_compile_args=['-std=c++2a', '-O3', '-march=native', '-Wall'],
                   language='c++',
                   optional=True)

setup(
    name="bhv",
    version=VERSION,
    author="Adam Vandervorst",
    author_email="contact@adamv.be",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/Adam-Vandervorst/PyBHV",
    packages=find_packages(),
    install_requires=[],
    extras_require={
        "pytorch": ["torch>=2.0.0"],
        "np": ["numpy>=1.24.2"],
    },
    keywords='ai binary hypervector hdc bsc',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',

        'License :: Free for non-commercial use',

        'Environment :: GPU :: NVIDIA CUDA',

        'Operating System :: POSIX :: Linux',

        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',

        'Typing :: Typed',
    ],
    ext_modules=[native]
)
