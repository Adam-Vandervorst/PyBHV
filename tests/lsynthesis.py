from bhv.symbolic import SymbolicBHV as BHV, Var
from bhv.slice import Slice


def tos(i, b): return bin(i)[2:].rjust(b, '0')
def tomask(s): return [x == '1' for x in s]
def frommask(m): return ''.join("01"[x] for x in m)
def bitconfigs(n): return [tomask(tos(i, n)) for i in range(2**n)]


def synth(vs, t):
    if vs:
        return vs[0].select(
                synth(vs[1:], t[:len(t)//2]),
                synth(vs[1:], t[len(t)//2:]))
    else:
        return BHV.ONE if t[0] else BHV.ZERO


names = [Var("a"), Var("b"), Var("c")]
configurations = bitconfigs(3)
operators = 0
for target in bitconfigs(8):
    f = synth(names, target)
    f_ = f.simplify()

    # print(f_.show())

    retrieved = [f_.execute(vars=dict(a=Slice(a), b=Slice(b), c=Slice(c)), bhv=Slice).b for a, b, c in configurations]
    operators += f_.size()
    # print(frommask(target))
    # print(frommask(retrieved))
    assert target == retrieved

print("average operators:", operators/256)

print(tos(184, 8), synth(names, tomask(tos(184, 8))).simplify().show())
print(tos(110, 8), synth(names, tomask(tos(110, 8))).simplify().show())
print(tos(90, 8), synth(names, tomask(tos(90, 8))).simplify().show())
print(tos(30, 8), synth(names, tomask(tos(30, 8))).simplify().show())
print(tos(22, 8), synth(names, tomask(tos(22, 8))).simplify().show())

maj3 = [sum(c) >= 2 for c in configurations]
print(frommask(maj3), synth(names, maj3).simplify().show())
