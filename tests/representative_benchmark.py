from time import monotonic_ns
from bhv.np import NumPyPacked64BHV as BHV, DIMENSION

S = BHV.representative
# C = BHV._counting_representative
L = BHV._logic_representative


i = 1
while i < 1000000:
    i = i + 1 if i < 100 else int(i*1.111)

    vs = BHV.nrand(i)

    t0 = monotonic_ns()
    for _ in range(100): S(vs)
    t1 = monotonic_ns()
    # for _ in range(100): C(vs)
    t2 = monotonic_ns()
    for _ in range(100): L(vs)
    t3 = monotonic_ns()

    print(i, "sampling", (t1 - t0)//100000, "logic", (t3 - t2)//100000)
