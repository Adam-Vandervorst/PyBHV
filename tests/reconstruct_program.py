from bhv.symbolic import SymbolicBHV as BHV, List, Var
from bhv.vanilla import VanillaBHV


a, b, c, d = Var("a"), Var("b"), Var("c"), Var("d")

abc = BHV.majority([a, b, c])

a1 = a.flip_frac_on(.1)
a0 = a.flip_frac_off(.1)

cq = c.select(a0, a1)

b_d = b ^ d
abc_d = abc ^ d

code = List("O", [cq, b_d, abc_d])\
    .simplify(expand_select_and_or=True)\
    .show_program(name="run", impl="VanillaBHV.")

assert code == """
def run(a, b, c, d):
    _0 = VanillaBHV.majority([a, b, c])
    _1 = _0 ^ d
    _2 = b ^ d
    _3 = VanillaBHV.random(0.2)
    _4 = a ^ _3
    _5 = a & _4
    _6 = ~c
    _7 = _6 & _5
    _8 = VanillaBHV.random(0.2)
    _9 = a ^ _3
    _10 = a | _4
    _11 = c & _10
    _12 = _11 | _7
    return [_12, _2, _1]
""".lstrip()
