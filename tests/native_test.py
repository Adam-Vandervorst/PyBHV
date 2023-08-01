from time import monotonic_ns

# from bhv.np import NumPyBoolBHV as BHV
from bhv.np import NumPyPacked64BHV as BHV
# from bhv.native import CNativePackedBHV as BHV

x = 0x7834d688d8827099
for i in range(5000000):
    x = x + (x % 7)

N = 201

t0 = monotonic_ns()

rs = [BHV.rand() for _ in range(N)]

t1 = monotonic_ns()
print("rand", t1 - t0)

ps = [r.roll_words(42) for r in rs]

t2 = monotonic_ns()
print("new permute", t2 - t1)

for r, p in zip(rs, ps):
    assert r == p.roll_words(-42)

t3 = monotonic_ns()
print("rpermute eq", t3 - t2)

m = BHV.majority(rs)

t4 = monotonic_ns()
print("majority", t4 - t3)

if False:
    ds = [r ^ m for r in rs]

    t5 = monotonic_ns()
    print("xor", t5 - t4)

    qs = [d.active() for d in ds]

    t6 = monotonic_ns()
    print("active", t6 - t5)
else:
    qs = [BHV.hamming(r, m) for r in rs]

    t5 = monotonic_ns()
    print("hamming", t5 - t4)

print(sum(qs)/N)
