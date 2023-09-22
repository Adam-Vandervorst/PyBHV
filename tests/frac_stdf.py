from bhv.np import NumPyPacked64BHV as BHV

r = BHV.rand2(2)
s = r.flip_pow(3)

print(BHV.bit_error_rate(r, s))
print(BHV.jaccard(r, s))
print(BHV.cosine(r, s))

print()
s = r.flip_pow(2)

print(BHV.bit_error_rate(r, s))
print(BHV.jaccard(r, s))
print(BHV.cosine(r, s))

print()
s = r.flip_pow(1)

print(BHV.bit_error_rate(r, s))
print(BHV.jaccard(r, s))
print(BHV.cosine(r, s))

print()
a = BHV.rand()
b = BHV.rand()
z = BHV.ZERO.flip_frac_on(0.001)

print(BHV.bit_error_rate(a, a))
print(BHV.jaccard(a, a, distance=True))
print(BHV.cosine(a, a, distance=True))
print()
print(BHV.bit_error_rate(a, b))
print(BHV.jaccard(a, b, distance=True))
print(BHV.cosine(a, b, distance=True))
print()
print(BHV.bit_error_rate(BHV.ONE, z))
print(BHV.jaccard(BHV.ONE, z, distance=True))
print(BHV.cosine(BHV.ONE, z, distance=True))

