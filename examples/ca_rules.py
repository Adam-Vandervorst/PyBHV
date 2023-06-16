from collections import Counter
from bhv.symbolic import SymbolicBHV, Var
from bhv.np import NumPyBoolBHV as BHV, DIMENSION
import numpy as np


RULE = 30
ITERATIONS = 1000

def make_rule(r: int):
    mask = [b == '1' for b in bin(r)[2:].rjust(8, "0")]
    formula = SymbolicBHV.synth([Var("left"), Var("center"), Var("right")], mask)
    formula = formula.simplify()
    print("formula:", formula.show())
    return lambda x: formula.execute(vars={"left": x.roll_bits(1), "center": x, "right": x.roll_bits(-1)})

rule = make_rule(RULE)

v = None
# completely random
# last_v = BHV.rand()

# low fraction of on bits
last_v = BHV.random(.03)

# single on bit
# initial = np.zeros(DIMENSION, dtype=np.bool_)
# initial[64] = np.bool_(1)
# last_v = BHV(initial)


with open(f"rule{RULE}.pbm", 'w') as f:
    f.write(f"P1\n{DIMENSION} {ITERATIONS}\n")

    f.write(''.join(map(lambda i: '1' if i else '0', last_v.data)) + '\n')

    for i in range(ITERATIONS):
        v = rule(last_v)
        f.write(''.join(map(lambda i: '1' if i else '0', v.data)) + '\n')
        last_v = v

