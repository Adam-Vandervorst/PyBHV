from bhv.symbolic import SymbolicBHV, Var
from bhv.vanilla import VanillaBHV as BHV, DIMENSION
from bhv.visualization import Image


def make_rule(r: int):
    mask = [b == '1' for b in reversed(bin(r)[2:].rjust(8, "0"))]
    formula = SymbolicBHV.synth([Var("left"), Var("center"), Var("right")], mask)
    formula = formula.simplify()
    print("formula:", formula.show())
    return lambda x: formula.execute(vars={"left": x.roll_bits(1), "center": x, "right": x.roll_bits(-1)})


RULE = 30
ITERATIONS = 10000

rule = make_rule(RULE)

# completely random
# last_v = BHV.rand()

# low fraction of on bits
# last_v = BHV.random(.03)

# single on bit
last_v = BHV.from_bitstream(i == 420 for i in range(DIMENSION))

vs = [last_v]

for i in range(ITERATIONS):
    vs.append(rule(vs[-1]))

with open(f"rule{RULE}.pbm", 'wb') as f:
    Image(vs).pbm(f, binary=True)
