# Python Boolean Hyper-Vectors

WIP repository for research into vector symbolic architectures like Multiply-Add-Permute and Binary Spatter Codes.
Can be used dependency-free.

## Contains
The fundamental research includes finding algebras with interesting properties on top of large boolean vectors. To this extent the library has [laws used for testing](tests/laws.py) and an expansive [set of operators](bhv/abstract.py) including:
- Multiple types of random vector generation
- Random and indexed select between vectors
- Ability to slightly modify a vector, for example by flipping a fraction of its bits
- Permutation, roll, and swapping with multiple interfaces
- Hashing and encoding
- Majority with multiple implementation
- Sample, a cheap alternative to Majority
- AND, OR, and XOR operators
- Composite operations like SELECT (or MUX) and FLIP-FRAC (flipping a fraction of the bits)
- Hamming, jaccard, cosine, and bit-error-rate "metrics"
- A system for relatedness, unrelatedness, and standard deviations apart
- zscore and pvalue

Additionally, provided are
- A symbolic implementation with analysis, plotting and pretty printing
- A native C++ implementation
- Law and unification backed expression simplification
- Compilation to operation sequences
- Efficient bit-packed representation
- Three redundant implementations on NumPy for performance and correctness
- A (performant) plain Python reference implementation
- A minimal abstraction for permutations with caching and composition
- Very basic embeddings for other datatypes (more to come)
- Graph visualization of distances in hyperdimensional space ([see example](examples/viz_distances.py)).
- Boolean expression and network synthesis

## Installation
Make sure you have an up-to-date Python version, 3.10 recommended.

`pip install bhv`

If you only want to work with plain Python, you're good to go with `from bhv.vanilla VanillaBHV as BHV`.

For the native vector option, you need a modern C++ compiler.

For interop, you'll need
`pip install numpy` or `pip install torch` with respectively `from bhv.np import NumPyPacked64BHV as BHV` or `from bhv.np import TorchBoolBHV as BHV`. 

## Getting started
One way to start is with going over [Kanerva's initial paper](http://ww.robertdick.org/iesr/papers/kanerva09jan.pdf) together with the library.
For that, multiple resources are provided:
- [A notebook going over the very basics](examples/Kanerva09.ipynb)
- [The grandmother example](examples/grandmother_example.py)
- [A Google Colab hosted reasoning by analogy example notebook](https://colab.research.google.com/drive/10gOc39TsM5CE-6u3kj2oe1t-8KZHr_bB?usp=sharing)
- [A guide to picking metrics](examples/Metric_Picker.ipynb)

If you like to dive into the code directly, I suggest the following entrypoints:
- [Finite State Machine example](examples/state_machine.py)
- [The base class AbstractBHV](bhv/abstract.py)
- [The most idiomatic implementation NumPyBoolBHV](bhv/np.py)

If you're comming at this from a Machine Learning angle, you may enjoy:
- [A minimal implementation](examples/winnow_classification.py) of the [winnow algorithm](https://en.wikipedia.org/wiki/Winnow_(algorithm)) on a minimal problem
- [A minimal implementation of classification based on the majority operator](examples/majority_classification.py) on a minimal problem
- [A Google Colab hosted digit classification via plain majority notebook](https://colab.research.google.com/drive/1xYQAXxcdFw89RV5CsflcvFhx3zpmEUxk?usp=sharing)
- [Graph classification notebook](https://colab.research.google.com/drive/1NrmCc99GrkmHm_VLs5nv9Q7BCbCLs0ar?usp=sharing) re-implementing [GraphHD](https://arxiv.org/abs/2205.07826)

## Note
This repository is highly active, and a work-in-progress.
Do expect changes to the naming, and even features to be swapped for more elegant alternatives.

Also works with PyPy. Use the vanilla Python implementation. The numeric operations are slower than on CPython, but the symbolic ones are way faster.
