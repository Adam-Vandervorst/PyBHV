import unittest
from bhv.np import NumPyBoolBHV as BHV
from bhv.symbolic import Var, SymbolicBHV
from statistics import fmean, median, harmonic_mean
from bhv.poibin import PoiBin

import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def inv_sigmoid(x):
    return -math.log((1 / x) - 1)


afs = [.35, .2, .46, .5, .8, .99]


# for afi in afs:
#     vi = BHV.random(afi)
#     for afj in afs:
#         vj = BHV.random(afj)
#         print((afi, afj, round((1. - ((1. - afi) * (1. - afj)))*100), round(100*(vi | vj).active_fraction())))
        # print((afi, afj, round((afi * afj)*100), round(100*(vi & vj).active_fraction())))
        # print((afi, afj, round(((afi * (1. - afj)) + ((1. - afi) * afj))*100), round(100*(vi ^ vj).active_fraction())))

for afi in afs:
    vi = BHV.random(afi)
    for afj in afs:
        vj = BHV.random(afj)
        for afk in afs:
            vk = BHV.random(afk)
            # print((afk, afi, afj, round(( afk*afi + (1. - afk)*afj )*100), round(100*vk.select(vi, vj).active_fraction())))
            # print((afk, afi, afj, round(( afk*afi + afk*afj + afi*afj - 2 * afk*afi*afj )*100), round(100*BHV.majority3(vi, vj, vk).active_fraction())))
            print((afk, afi, afj, round(( 1. - PoiBin([afk, afi, afj]).cdf(1) )*100), round(100*BHV.majority3(vi, vj, vk).active_fraction())))

