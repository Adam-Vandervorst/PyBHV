# -*- coding: utf-8 -*-
"""
Following along with [Kanerva 2009](http://ww.robertdick.org/iesr/papers/kanerva09jan.pdf), starting from "Hyperdimensional Representation". Please do open these side-by-side, while there is quite a bit of text copied over, it's mostly formulas, where as the interesting insights can only be found in the paper.

Also the booleans {0, 1} are used instead of {-1, 1}.
"""

# Let's install the package
# pip install bhv==0.3.0

# Hypervectors here are called Boolean-Hyper-Vectors or BHVs for short.
# Import a BHV implementation
from bhv.vanilla import VanillaBHV as BHV
# Import the corresponding permutation datastructure, we'll get to this
from bhv.vanilla import VanillaPermutation as Perm

"""> We can measure distances between points in Euclidean
or Hamming metric. For binary spaces the Hamming distance is the simplest: it is the number of places at which
two binary vectors differ, and it is also the length of the
shortest path from one corner point to the other along the
edges of the hypercube. In fact, there are 2k such shortest
paths between two points that are k bits apart. Naturally,
the maximum Hamming distance is 10,000 bits, from any
point to its opposite point.

One example of opposite points is 101010... and 010101... which differs in every position.
The example of opposite points below is all zeros and all ones.
The absolute distance in number of bits can be retrieved by `hamming`.
"""

print(BHV.ONE.hamming(BHV.ZERO))
# Note that the hypervector dimension here is 8192 instead of 10000

"""

> Although the points are not concentrated or clustered
anywhere in the space—because every point is just like
every other point—the distances are highly concentrated
half-way into the space, or around the distance of 5,000
bits, or 0.5.

Let's generate two random BHVs and find their bit-error-rate (BER), this should be close to 0.5. The BER corresponds to the hamming distance over the number of dimensions. `rand`"""

print(BHV.rand().bit_error_rate(BHV.rand()))

"""

> It is easy to see that half the space is closer to a
point than 0.5 and the other half is further away, but it is
somewhat surprising that less than a millionth of the space
is closer than 0.476 and less than a thousand-millionth is
closer than 0.47; similarly, less than a millionth is further
than 0.524 away and less than a thousand-millionth is
further than 0.53. These figures are based on the binomial
distribution with mean 5,000 and standard deviation (STD)
50, and on its approximation with the normal distribution—
the distance from any point of the space to a randomly
drawn point follows the binomial distribution. 

This is a useful notion, and how likely a BHV is to occur randomly generated,
can be found by `zscore` (giving you the standard deviations) 
and `pvalue` (the absolute chance). 
"""

# very likely to be randomly generated, because it is
# Here -1.0 would mean "one standard deviation less 1s than expected"
print(BHV.rand().zscore())
# flipping just 5% extra on, makes it a 6 sigma+ outlier
print(BHV.rand().flip_frac_on(0.05).zscore())

"""

> We can go on
taking vectors at random without needing to worry about
running out of vectors—we run out of time before we run
out of vectors. We say that such vectors are unrelated.

All vectors we'll practically sample are all unrelated.
There's a `related` function that takes the safety margin, 6 sigma by default."""

random_bhvs = BHV.nrand(10)
for x in random_bhvs:
    for y in random_bhvs:
        if x != y:
            assert x.unrelated(y)

"""

> Measured in standard deviations, the bulk of the space, and
the unrelated vectors, are 100 STDs away from any given
vector.

So we expect two BHVs in the space to be 100 (or in our case, 90) STDs apart."""

print(BHV.rand().std_apart(BHV.rand()))

"""

> This peculiar distribution of the space makes hyperdimensional representation robust. When meaningful entities
are represented by 10,000-bit vectors, many of the bits can
be changed—more than a third—by natural variation in
stimulus and by random errors and noise, and the resulting
vector can still be identified with the correct one, in that it
is closer to the original ‘‘error-free’’ vector than to any
unrelated vector chosen so far, with near certainty.

Let's try and flip 1/3 bits, and see how many STDs they are apart.
This is a lot less than the previous test!"""

r = BHV.rand()

print(r.std_apart(r.flip_frac(1/3)))

"""

> The robustness is illustrated further by the following
example. Let us assume that two meaningful vectors A and
B are only 2,500 bits apart—when only 1/4 of their bits
differ. The probability of this happening by chance is about
zero, but a system can create such vectors when their
meanings are related; more on such relations will be said
later. So let us assume that 1/3 of the bits of A are changed
at random; will the resulting ‘‘noisy’’ A vector be closer to
B than to A—would it be falsely identified with B? It is
possible but most unlikely because the noisy vector would
be 4,166 bits away from B, on the average, and only 3,333
bits from A; the difference is 17 STDs. The (relative)
distance from the noisy A vector to B is given by d = e -
2de with d = 1/4 and e = 1/3. Thus, adding e amount of
noise to the first vector increases the distance to the second
vector by (1 - 2d)e on the average. Intuitively, most
directions that are away from A in hyperspace are also
away from B.

Let's follow the full example.


"""

A = BHV.rand()
A_corrupted = A.flip_frac(1/3)
B = A.flip_frac(1/4)

d = 1/4
e = 1/3
relative_distance = d + e - 2*d*e

print(relative_distance)
print(A_corrupted.bit_error_rate(B))

# Note this is not the 17 STDs from the paper because we're working in less dimensions
print(A_corrupted.std_apart(B) - A.std_apart(A_corrupted))

"""

> The similarity of patterns is the flip-side of distance. We
say that two patterns, vectors, points are similar to each
other when the distance between them is considerably
smaller than 0.5. We can now describe points of the space
and their neighborhoods as follows. Each point has a large
‘‘private’’ neighborhood in terms of distance: the volume of
space within, say, 1/3 or 3,333 bits is insignificant compared to the total space. The rest of the space—all the
unrelated ‘‘stuff’’—becomes significant only when the
distance approaches 0.5. In a certain probabilistic sense,
then, two points even as far as 0.45 apart are very close to
each other. Furthermore, the ‘‘private’’ neighborhoods of
any two unrelated points have points in common—there
are patterns that are closely related to any two unrelated
patterns. For example, a point C half-way between unrelated points A and B is very closely related to both, and
another half-way point D can be unrelated to the first, C.
This can be shown with as few as four dimensions:
A = 0000, B = 0011, C = 0001, and D = 0010. However, the ‘‘unusual’’ probabilities implied by these relative
distances require high dimensionality. This is significant
when representing objects and concepts with points of the
hyperspace, and significantly different from what we are
accustomed to in ordinary three-dimensional space.

Let's do the same experiment (with BHVs) and verify for ourselves that while both C and D are at a halfwaypoint, they're further removed from each other than from A B.

"""

A = BHV.ZERO
B = BHV.ONE
C = BHV.rand() # random is halfway between 0 and 1 on average
D = ~C # the inverse of C is unrelated to C

print(A.bit_error_rate(C), B.bit_error_rate(C))
print(A.bit_error_rate(D), B.bit_error_rate(D))
print(C.bit_error_rate(D))

"""

> The sum (and the mean) of random vectors has the
following important property: it is similar to each of the
vectors being added together. The similarity is very pronounced when only a few vectors are added

We're working with BHVs so we don't have a sum with the same properties, but there is a notion of mean. The mean is a "majority" operation plus a tiebreaker for even numbers of BHVs."""

vs = BHV.nrand(5)
vs_mean = BHV.majority(vs)

# while the random vectors are pairwise unrelated
for i in range(5):
    assert vs[i].unrelated(vs[(i + 1) % 5])
# they're all related to the mean
print(vs[0].bit_error_rate(vs_mean))
print(vs[0].std_apart(vs_mean))

#for v in vs:
#    assert v.sixsigma(vs_mean)

"""

> A very basic and simple multiplication of binary vectors is
by componentwise Exclusive-Or (XOR). The XOR of two vectors has 0s where the two agree and it has 1s where they
disagree. For example, 0011…10 XOR 0101…00 =
0110…10. Mathematically, the XOR is the arithmetic sum
modulo 2. The (1, -1)-binary system, also called bipolar,
is equivalent to the (0, 1)-binary system when the XOR is
replaced by ordinary multiplication. We will use the
notation A  B for the multiplication of the vectors A and
B—for their product-vector. Here * is the XOR unless
otherwise noted.
The XOR commutes, A \* B = B \* A, and is its own
inverse so that A \* A = O, where O is the vector of all 0s
(in algebra terms O is the unit vector because A \* O \= A).

Here we'll denote XOR with the carret `^`. The laws are thoroughly tested in the library, but let's look at a few examples here."""

A = BHV.rand()
B = BHV.rand()

# commutative
assert A ^ B == B ^ A
# it's own inverse
assert A ^ A == BHV.ZERO

"""

> Since the XOR-vector has 1s where the two vectors disagree, the number of 1s in it is the Hamming distance
between the two vectors. By denoting the number of 1s in a
binary vector X with |X| we can write the Hamming distance d between A and B as d(A, B) = |A * B|

Here we'll refrain from choosing a standard distance metric, but this fact is used everywere."""

A = BHV.rand()
B = BHV.rand()

assert A.hamming(B) == (A ^ B).active()

"""

> Multiplication can be thought of as a mapping of points
in the space. Multiplying the vector X with A maps it to
the vector X_A = A \* X which is as far from X as there are 1s
in A (i.e., d(A, X) = |X_A \* X| = |(A \* X) \* X| = |A \* X \* X| = |A|).

Another few important properties, set out in code."""

A = BHV.rand()
X = BHV.rand()

X_A = A ^ X

assert X_A.hamming(X) == A.active()

"""

> If A is a typical (random) vector of the space,
about half of its bits are 1s, and so X_A is in the part of the
space that is unrelated to X in terms of the distance criterion. Thus we can say that multiplication randomizes.

Let's check this """

A = BHV.rand()
X = BHV.rand()

X_A = A ^ X

print(X_A.std_apart(X))

"""

> Mapping with multiplication preserves distance. This is
seen readily by considering X_A = A \* X and Y_A = A \* Y;
taking their XOR, and noting that the two As cancel out
thus: X_A \* Y_A = (A \* X) \* (A \* Y) = A \* X \* A \* Y = X \* Y
Since the XOR-vector is the same, the Hamming distance
is the same: |X_A \* Y_A| = |X \* Y|

And the logical extension"""

A = BHV.rand()
X = BHV.rand()
Y = BHV.rand()

X_A = A ^ X
Y_A = A ^ Y

assert (X_A ^ Y_A) == X ^ Y

"""

> A very useful property of multiplication is that it
distributes over addition. That means, for example, that
A \* \[X + Y + Z\] = \[A \* X + A \* Y + A \* Z\]
The brackets \[…\] stand for normalization. Distributivity is
invaluable in analyzing these representations and in
understanding how they work and fail.

The BHVs do not support normalization in the sense that's meant here. Therefore, care has to be taken as to not introduce unwanted noise. Instead of adding multiple times, and normalizing, one should prefer the addition (here mean) of multiple items."""

A, X, Y, Z = BHV.nrand(4)

LHS = A ^ BHV.majority([X, Y, Z])
RHS = BHV.majority([A ^ X, A ^ Y, A ^ Z])

assert LHS == RHS

"""

> Permutations reorder the vector components and thus are
very simple; they are also very useful in constructing a
cognitive code. We will denote the permutation of a vector
with a multiplication by a matrix (the permutation matrix
Π), thus X_Π = ΠX. We can also describe the permutation
of n elements as the list of the integers 1, 2, 3, …, n in the
permuted order. A random permutation is then one where
the order of the list is random—it is a permutation chosen
randomly from the n! possible permutations.

When theory hits reality, it's not quite as pretty. Firstly, a 8192x8192 matrix is large (LA folks are laughing in the background) and wasteful. This can be resolved by using a denser representation (`NumPyWordPermutation` uses an array of indices, which is [still no optimal](https://hackmd.io/@dabo/rkP8Pcf9t#A-compact-data-structure-for-storing-a-permutation)). Secondly, the number of permutations 8192! is extremely large (28000+ digits). Therefore, implementions needn't support all permutations (`NumPyWordPermutation` only supports permutations on 64 bit words at the moment). Let's take a look at the basic properties with an explicit representation.
"""

Π = Perm.random()

X, Y = BHV.nrand(2)

assert Π(X) ^ Π(Y) == Π(X ^ Y)
assert Π(X).hamming(Π(Y)) == X.hamming(Y)

"""

> As mentioned above, we can map the same vector with two
different permutations and ask how similar the resulting
vectors are: by permuting X with Π and Γ, what is the
distance between ΠX and ΓX (...)? Unlike above with multiplication by
a vector, this depends on the vector X (e.g., the 0-vector is
unaffected by permutation).

Let's generate another permutation."""

Π, Γ = Perm.nrandom(2)

X = BHV.rand()

assert Π(X).unrelated(Γ(X))
