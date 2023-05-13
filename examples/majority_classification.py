from bhv.vanilla import VanillaBHV as BHV

N = 199*2

xs = BHV.nrand(N)
ys = [i % 2 == 0 for i in range(N)]

pos = BHV.majority([x for x, y in zip(xs, ys) if y])
neg = BHV.majority([x for x, y in zip(xs, ys) if not y])

right = 0
for x, y in zip(xs, ys):
    spos = x.bit_error_rate(pos)
    sneg = x.bit_error_rate(neg)
    prediction = spos < sneg
    right += prediction == y
print(f"acc {right/N}")
