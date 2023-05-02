from setuptools import setup, find_packages

VERSION = '0.2.6'
DESCRIPTION = 'Boolean Hypervectors'
LONG_DESCRIPTION = 'Boolean Hypervectors with various operators for experiments in hyperdimensional computing (HDC).'

setup(
    name="bhv",
    version=VERSION,
    author="Adam Vandervorst",
    author_email="contact@adamv.be",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    extras_require={
        "torch": ["torch>=2.0.0"],
        "torch-hd": ["torch-hd~=5.0.1"],
        "numpy": ["numpy>=1.24.2"],
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
    ]
)
