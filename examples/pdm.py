from random import uniform
from math import sin, cos, tau, radians, degrees, exp, log

from bhv.vanilla import VanillaBHV as BHV, DIMENSION
from bhv.visualization import Image

from scipy.special import iv  # Modified Bessel function of the first kind


def von_Mises(theta, kappa): return exp(kappa*cos(theta))/(tau*iv(0, kappa))
def index_to_angle(i): return tau*i/DIMENSION
def angle_to_index(a): return int(DIMENSION*a/tau)
def angle_difference(a1, a2): return 2*abs(sin((a1 - a2)/2))
def phase_difference(a1, a2): return (angle_difference(a1, a2)**2)/4


# v = BHV.from_bitstream(von_Mises(index_to_angle(i), 0., log(tau)) < uniform(0, 1) for i in range(DIMENSION))
v = BHV.from_bitstream(sin(index_to_angle(i)) < uniform(-1, 1) for i in range(DIMENSION))
# v = BHV.from_bitstream(sin(index_to_angle(i)) < uniform(0, 1)**2 for i in range(DIMENSION))

vs = [v.roll_bits(i) for i in range(DIMENSION)]


with open(f"pdm_circle.pbm", 'wb') as f:
    Image(vs).pbm(f, binary=True)


for d in range(360):
    v0 = vs[angle_to_index(radians(0))]
    vd = vs[angle_to_index(radians(d))]
    print(d, round(phase_difference(0, radians(d)), 4), round(2*v0.bit_error_rate(vd) - .5, 4))
    # print(d, round(angle_difference(0, radians(d)), 4), round(v0.cosine(vd, distance=True), 4))
