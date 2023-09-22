from bhv.vanilla import VanillaBHV as BHV
from bhv.visualization import DistanceGraph


def lolipop(p, q, U, neighborhood=3):
    return BHV.majority([r for r in U if not (q ^ r).unrelated(p, neighborhood)])


a, b, c, d = BHV.nrand(4)

abc = BHV.majority([a, b, c])

a1 = a.flip_frac_on(.1)
a0 = a.flip_frac_off(.1)

cq = c.select(a0, a1)

b_d = b ^ d
abc_d = abc ^ d

a_lp_b = lolipop(a, abc, [a, b, c, d, abc, a1, a0, cq, b_d, abc_d])

print("graph {")
DistanceGraph.from_scope(locals()).graphviz()
print("}")

