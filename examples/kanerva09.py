"""
Following along with http://ww.robertdick.org/iesr/papers/kanerva09jan.pdf

Note that the hypervector dimension here is 8192 instead of 10000

Also the booleans {0, 1} are used instead of {-1, 1}.

Hypervectors are called Boolean Hyper Vectors or BHVs for short.
"""

from bhv.np import NumPyPacked64BHV as BHV, NumPyWordPermutation as Perm


"""
We can measure distances between points in Euclidean
or Hamming metric. For binary spaces the Hamming distance is the simplest: it is the number of places at which
two binary vectors differ, and it is also the length of the
shortest path from one corner point to the other along the
edges of the hypercube. In fact, there are 2k such shortest
paths between two points that are k bits apart. Naturally,
the maximum Hamming distance is 10,000 bits, from any
point to its opposite point.

One example of opposite points is 101010... and 010101... which differs in every position.
The example of opposite points below is all zeros and all ones.
The absolute distance can be retrieved by `hamming` which is defined as 
`(self ^ other).active()`
"""

print(BHV.ONE.hamming(BHV.ZERO))

"""
The distance is often expressed
relative to the number of dimensions, so that here 10,000
bits equals 1.

This is a lot more convenient for us, too.
The relative distance can be retrieved by `bit_error_rate` (BER) which is defined as 
`(self ^ other).active_fraction()`
"""

print(BHV.ONE.bit_error_rate(BHV.ZERO))

"""
Although the points are not concentrated or clustered
anywhere in the space—because every point is just like
every other point—the distances are highly concentrated
half-way into the space, or around the distance of 5,000
bits, or 0.5.

Let's generate two random BHVs and find their BER, this should be close to 0.5.
"""

print(BHV.rand().bit_error_rate(BHV.rand()))

"""
It is easy to see that half the space is closer to a
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
print(BHV.rand().zscore())
# flipping just 5% extra on, makes it a 6 sigma+ outlier
print(BHV.rand().flip_frac_on(0.05).zscore())


"""
We can go on
taking vectors at random without needing to worry about
running out of vectors—we run out of time before we run
out of vectors. We say that such vectors are unrelated.

All vectors we'll practically sample are all unrelated.
There's a "related" function that fixes a certainty called `sixsigma`.
"""

random_bhvs = BHV.nrand(10)
for x in random_bhvs:
    for y in random_bhvs:
        if x != y:
            assert x.unrelated(y)

"""
Measured in standard deviations, the bulk of the space, and
the unrelated vectors, are 100 STDs away from any given
vector.

So we expect two BHVs in the space to be 100 (or in our case, 90) STDs apart.
"""

print(BHV.rand().std_apart(BHV.rand()))

"""
This peculiar distribution of the space makes hyperdimensional representation robust. When meaningful entities
are represented by 10,000-bit vectors, many of the bits can
be changed—more than a third—by natural variation in
stimulus and by random errors and noise, and the resulting
vector can still be identified with the correct one, in that it
is closer to the original ‘‘error-free’’ vector than to any
unrelated vector chosen so far, with near certainty.

Let's try and flip 1/3 bits, and see how many STDs they are apart.
This is a lot less than the previous test!
"""

r = BHV.rand()

print(r.std_apart(r.flip_frac(1/3)))

"""
The robustness is illustrated further by the following
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
distance from the noisy A vector to B is given by d ? e -
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

