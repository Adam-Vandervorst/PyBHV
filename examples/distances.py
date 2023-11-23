# Note this is only going to make sense for DIMENSION=8192
from bhv.vanilla import VanillaBHV as BHV, DIMENSION

print("DIMENSION", DIMENSION)
print()


a, b, c, d = BHV.nrand(4)
abcd = BHV.majority([a, b, c, d])
abc = BHV.majority([a, b, c])
a1 = a.flip_frac_on(.1)

assert a.related(abcd) and a.related(abc) and a.related(a1)
assert not a.related(b) and not a.related(c) and not a.related(d)


ns, vs = zip(*filter(lambda v: isinstance(v[1], BHV), globals().items()))
print(f"standard deviations from a to {', '.join(ns)}: ")
print(*map(round, a.distribution(vs)))
print("closest to a:")
print(ns[a.closest(vs)])
print("top-2 closest to a:")
print(*map(ns.__getitem__, a.top(vs, 2)))
print("within 50σ of a:")
print(*map(ns.__getitem__, a.within_std(vs, 50)))
print("4σ closer to a than random:")
print(*map(ns.__getitem__, a.within_std(vs, -6, relative=True)))

print()

print(f"{BHV.std_to_frac(6):<7.2%} active fraction corresponding to 6σ")
print(f"{BHV.frac_to_std(0.033):<7.1f} standard deviations corresponding to 3% active fraction")
print(f"{BHV.std_to_frac(173):<7.2%} active fraction corresponding to 173σ")
print(f"{BHV.frac_to_std(.96):<7.1f} standard deviations corresponding to 96% active fraction")

print()

print(f"{BHV.std_to_frac(-6, relative=True):<7.2%} more/less difference compared to random, corresponding to -6σ")
print(f"{BHV.frac_to_std(-0.033, relative=True):<7.1f} more/less standard deviations compared to random, corresponding to -3.3%")
print(f"{BHV.std_to_frac(10, relative=True):<7.2%} more/less difference compared to random, corresponding to 10σ")
print(f"{BHV.frac_to_std(0.055, relative=True):<7.1f} more/less standard deviations compared to random, corresponding to 5.5%")

print()

print(f"{BHV.std_apart(a, b):<6.1f} standard deviations apart: d(a, b)")
print(f"{BHV.std_apart(a, a1):<6.1f} standard deviations apart: d(a, a1)")
print(f"{BHV.std_apart(a, ~a):<6.1f} standard deviations apart: d(a, ~a)")
print(f"{BHV.std_apart(a, BHV.ZERO):<6.1f} standard deviations apart: d(a, 0)")
print(f"{BHV.std_apart(a, BHV.ONE):<6.1f} standard deviations apart: d(a, 1)")

print()

print(f"{BHV.std_apart(a, b, relative=True):<6.1f} more/less standard deviations than two randoms: d(a, b)")
print(f"{BHV.std_apart(a, a1, relative=True):<6.1f} more/less standard deviations than two randoms: d(a, a1)")
print(f"{BHV.std_apart(a, ~a, relative=True):<6.1f} more/less standard deviations than two randoms: d(a, ~a)")
print(f"{BHV.std_apart(a, BHV.ZERO, relative=True):<6.1f} more/less standard deviations than two randoms: d(a, 0)")
print(f"{BHV.std_apart(a, BHV.ONE, relative=True):<6.1f} more/less standard deviations than two randoms: d(a, 1)")

print()

for k in [0., .02, .1, .25, .4, .48, .5, .52, .6, .75, .9, .98, 1.]:
    print(f"{k:.0%} between a and ~a:")
    print(" ", a.std_relation((BHV.HALF if k == .5 else BHV.random(k)).select(~a, a)))
