from bhv.np import NumPyPacked64BHV as BHV


a, b = BHV.nrand(2)

for frac in [.1, .5, .9]:
    print(frac)
    a_sub = BHV.random(frac) & a
    b_sub = BHV.random(frac) & b

    print(a.bit_error_rate(a_sub), a.bit_error_rate(b_sub))
    print(a.cosine(a_sub), a.cosine(b_sub))
    print(a.jaccard(a_sub), a.jaccard(b_sub))
    print(a.mutual_information(a_sub), a.mutual_information(b_sub))
