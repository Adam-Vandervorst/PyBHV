from bhv.np import NumPyPacked64BHV as BHV, DIMENSION


"""
Given some random vectors
"""
a, b, c = BHV.nrand(3)

"""
`active` counts the number of on-bits.
For rand vectors we expect this to be around DIMENSION/2
"""
print(a.active())
print(DIMENSION//2)

"""
Getting the bits that are on in both vectors by counting the active bits in their conjunction.
For rand vectors we expect this to be around DIMENSION/4
"""
ab = a & b
print(ab.active())
print(DIMENSION//4)



#
#
# r = BHV.rand2(2)
# s = r.flip_pow(3)
#
# print(BHV.bit_error_rate(r, s))
# print(BHV.jaccard(r, s))
# print(BHV.cosine(r, s))
#
# print()
# s = r.flip_pow(2)
#
# print(BHV.bit_error_rate(r, s))
# print(BHV.jaccard(r, s))
# print(BHV.cosine(r, s))
#
# print()
# s = r.flip_pow(1)
#
# print(BHV.bit_error_rate(r, s))
# print(BHV.jaccard(r, s))
# print(BHV.cosine(r, s))
#
# print()
# a = BHV.rand()
# b = BHV.rand()
# z = BHV.ZERO.flip_frac_on(0.001)
#
# print(BHV.bit_error_rate(a, a))
# print(BHV.jaccard(a, a, distance=True))
# print(BHV.cosine(a, a, distance=True))
# print()
# print(BHV.bit_error_rate(a, b))
# print(BHV.jaccard(a, b, distance=True))
# print(BHV.cosine(a, b, distance=True))
# print()
# print(BHV.bit_error_rate(BHV.ONE, z))
# print(BHV.jaccard(BHV.ONE, z, distance=True))
# print(BHV.cosine(BHV.ONE, z, distance=True))
#
