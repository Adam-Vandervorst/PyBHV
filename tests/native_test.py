from time import monotonic_ns

from bhv.np import NumPyPacked64BHV as BHV
# from bhv.native import NativePackedBHV as BHV


N = 201

t0 = monotonic_ns()

rs = BHV.nrand(N)

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
NumPy, native, cpp
rand
471245
167191
142936
60613

majority
45263512
753519
254592
228083

xor
294547
76813
99986
47869

active
773246
50594
29244
8045

hamming
950325
62586
82824
12533

3866.9004975124376

"""