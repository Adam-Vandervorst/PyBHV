from bhv.np import NumPyPacked64BHV as BHV

a = BHV.rand()
for i in range(0, 21):
    p = i/20

    b = a.flip_frac(p)
    print(p)
    print("ber", 1. - a.bit_error_rate(b))
    print("cos", a.cosine(b))
    print("jac", a.jaccard(b))
    print("mut", a.mutual_information(b, distance=True))
    print("tve", a.tversky(b, .5, .5))
