from bhv.np import NumPyPacked64BHV as BHV
from bhv.visualization import DistanceGraph


a, b, c, d = BHV.nrand(4)

abc = BHV.majority([a, b, c])

a1 = a.flip_frac_on(.1)
a0 = a.flip_frac_off(.1)

cq = c.select(a0, a1)


DistanceGraph({
    'a': a,
    'b': b,
    'c': c,
    'abc': abc,
    'a1': a1,
    'a0': a0,
    'cq': cq,
}).graphviz()
