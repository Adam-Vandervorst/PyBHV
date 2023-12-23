# from bhv.np import NumPyPacked64BHV as BHV, DIMENSION
from bhv.native import NativePackedBHV as BHV, DIMENSION
# from bhv.vanilla import VanillaBHV as BHV, DIMENSION

vs = [BHV.level(i/DIMENSION) for i in range(DIMENSION + 1)]

for i in range(DIMENSION + 1):
    assert vs[i].active_fraction() == i/DIMENSION
    assert i == 0 or (vs[i-1] | vs[i]) == vs[i]
    assert (vs[i] & BHV.EVEN).active() == (i + 1)//2
    assert (vs[i] & BHV.ODD).active() == i//2
