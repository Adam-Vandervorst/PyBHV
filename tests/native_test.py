from time import monotonic_ns

from bhv.np import NumPyPacked64BHV as BHV
# from bhv.native import CNativePackedBHV as BHV


N = 201

t0 = monotonic_ns()

rs = [BHV.rand() for _ in range(N)]

t1 = monotonic_ns()
print("rand", t1 - t0)

m = BHV.majority(rs)

t2 = monotonic_ns()
print("majority", t2 - t1)

if False:
    ds = [r ^ m for r in rs]

    t3 = monotonic_ns()
    print("xor", t3 - t2)

    qs = [d.active() for d in ds]

    t4 = monotonic_ns()
    print("active", t4 - t3)
else:
    qs = [BHV.hamming(r, m) for r in rs]

    t3 = monotonic_ns()
    print("hamming", t3 - t2)

print(sum(qs)/N)

"""
NumPy, cnative, cpp
453862
91750
9197

majority
2381813
321277
15558

xor
280561
124080
63167

active
1382577
21120
7755

hamming
1759557
33983
2034

3866.9004975124376

"""