# Python Boolean Hyper-Vectors [WIP]

A rich research framework for [hyperdimensional computing](https://en.wikipedia.org/wiki/Hyperdimensional_computing) on large boolean vectors supporting program transformation and multiple backends for computation (plain Python, C++, NumPy, PyTorch). Many metrics and utility functions aim to aid the intuitive understanding of this new paradigm, and there are multiple levels of functionality available from the data marshalling and the basic (XOR, MAJ, PERMUTE)-algebra to cryptography support. All vector operations are implemented in C(++) and make use of bit-packing and SIMD, subprograms can be optimized and compiled to these operations in Python or C, and parallelization and pipelining are planned.

If your application is a relatively direct pipeline, take a look at [HDCC](https://arxiv.org/abs/2304.12398).
If you want a more stable library, or want to work with another base field than the booleans, use [torch-hd](https://pypi.org/project/torch-hd/).

## Overview
The fundamental research includes finding algebras with interesting properties on top of large boolean vectors. To this extent the library has [laws used for testing](tests/laws.py) and an expansive [set of operators](bhv/abstract.py) including:
- Multiple types of fast random vector generation
- Random and indexed select between vectors
- Ability to slightly modify a vector, for example by flipping a fraction of its bits
- Permutation, roll, and swapping with multiple interfaces
- Hashing and encoding
- Majority with multiple implementation
- Sample, a cheap alternative to Majority
- AND, OR, XOR, and NOT operators
- Composite operations like SELECT (or MUX) and FLIP-FRAC (flipping a fraction of the bits)
- Hamming, jaccard, cosine, bit-error-rate, tversky, and mutual-information metrics
- A system for relatedness, unrelatedness, and standard deviations apart
- zscore and pvalue

Additionally, provided are
- [A symbolic implementation with simplification, analysis, plotting and pretty printing](bhv/symbolic.py)
- A native [C++ implementation](bhv/cnative)
- Law and unification backed expression simplification
- Compilation to operation sequences (circuits)
- Efficient bit-packed representation (saves 8x memory compared to the traditional NumPy and PyTorch bool!)
- [Three redundant implementations on NumPy](bhv/np.py) for performance and correctness
- A (performant) [plain Python implementation](bhv/vanilla.py)
- A minimal abstraction for permutations with caching and composition
- Very [basic embeddings](bhv/embedding.py) for other datatypes (more to come)
- Graph visualization of distances in hyperdimensional space ([see example](examples/viz_distances.py)).
- Boolean expression and network synthesis (e.g. [Cellular Automata](examples/ca_rules) and [perfectly random functions](benchmarks/exact_synthesis.py))
- Visualization and storage via [pbm](bhv/visualization.py) (e.g. [Cellular Automata](examples/ca_rules))
- A normal form and conversions between different implementations and storage methods
- [Linear and adiabatic variants](bhv/variants.py) and [example](examples/reasoning_by_analogy_linear.py)

## Installation
Make sure you have a recent Python version, 3.10 is recommended.

`pip install bhv`

If you only want to work with plain Python, you're good to go with `from bhv.vanilla VanillaBHV as BHV`.

For the native option, you need a modern C++ compiler and use `from bhv.native import NativePackedBHV as BHV`. The setup process should attempt to install this by default.

For interop with (the Python interface of) NumPy and PyTorch, you'll need
`pip install numpy` or `pip install torch` with respectively `from bhv.np import NumPyPacked64BHV as BHV` or `from bhv.np import TorchBoolBHV as BHV`. 

## Getting started
Some resources to get started with the library, if you're looking for a broader intro, please take a look at [hd-computing.com](https://www.hd-computing.com/).

### New to Hyperdimensional Computing
Basic uses (in the context of [neo-GOFAI](https://www.cs.cmu.edu/~cga/gofai/)) are given in [my presentation with a installation-free notebook](https://colab.research.google.com/drive/10XSpooxDAeYVMivF3W2-N0W5yEIUfdZl?usp=sharing). 

The fundamental angle is to start is with [Kanerva's initial paper](http://ww.robertdick.org/iesr/papers/kanerva09jan.pdf) together with the library.
For that, multiple resources are provided:
- [A notebook going over the very basics](examples/Kanerva09.ipynb)
- [The grandmother example](examples/grandmother_example.py)
- [A Google Colab "reasoning by analogy" hosted notebook](https://colab.research.google.com/drive/10gOc39TsM5CE-6u3kj2oe1t-8KZHr_bB?usp=sharing)
- [A guide to picking metrics](examples/Metric_Picker.ipynb)

As for a Machine Learning angle, you may enjoy:
- [A minimal implementation](examples/winnow_classification.py) of the [winnow algorithm](https://en.wikipedia.org/wiki/Winnow_(algorithm)) on a minimal problem
- [A minimal implementation of classification based on the majority operator](examples/majority_classification.py) on a minimal problem
- [A Google Colab hosted digit classification via plain majority notebook](https://colab.research.google.com/drive/1xYQAXxcdFw89RV5CsflcvFhx3zpmEUxk?usp=sharing)
- [Graph classification notebook](https://colab.research.google.com/drive/1NrmCc99GrkmHm_VLs5nv9Q7BCbCLs0ar?usp=sharing) re-implementing [GraphHD](https://arxiv.org/abs/2205.07826)

### Evaluating the library
If you like to dive into the code directly, I suggest the following entrypoints:
- [Finite State Machine example](examples/state_machine.py)
- [The base class AbstractBHV](bhv/abstract.py)
- [The most idiomatic implementation NumPyBoolBHV](bhv/np.py)

Example exploratory usages of the library:
- [Trying to improve upon linear hypervector search](benchmarks/lookup.py) and [its implementation](bhv/lookup.py)
- [Exact logic synthesis](https://en.wikipedia.org/wiki/Logic_synthesis), [the benchmarking code](benchmarks/exact_synthesis.py)
- [Cellular automata](https://en.wikipedia.org/wiki/Cellular_automaton), [the code (which produces pretty pictures)](examples/ca_rules.py), [a talk on them in relation to VSA](https://youtu.be/GZ9pNTQmrsY?t=17), and [comments on the relation to the library](https://youtu.be/GZ9pNTQmrsY?t=985)
- [Fiestal Cipher](https://en.wikipedia.org/wiki/Feistel_cipher), [the tests (for statistical properties)](tests/fiestal.py), and [the implementation](bhv/abstract.py)

## Note
This repository is (highly) active development, and a work-in-progress.
Do expect changes to the naming, and even features to be swapped for more elegant alternatives.

The codebase also works with PyPy. Use the vanilla Python implementation. The numeric operations are slower than on CPython, but the symbolic ones are way faster.

If you have any feedback, raise an informal issue, or email me at [contact@adamv.be](mailto:contact@adamv.be)

If the library is not as fast as possible, that's a bug, please report.
