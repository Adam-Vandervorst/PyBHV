from bhv.symbolic import SymbolicBHV as BHV, Var, List
from bhv.slice import Slice
from string import ascii_lowercase
from random import shuffle, sample, random


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


I = 8
O = 4
names = [Var(x) for x in ascii_lowercase[:I]]
fs = []

for j in range(O):
    target = [i % 2 == 0 for i in range(2**I)]
    shuffle(target)

    f = synth(names, target).simplify()
    fs.append(f)

print(List("O", fs).show())
# List("O", fs).graphviz(structural=True)


initial = names
layers = [initial]
sizes = [20, 20, 20, 20, O]

for size in sizes:
    layer = []
    for elem in range(size):
        r = random()
        if r < .75:
            i, j = sample(layers[-1], k=2)
            layer.append(i ^ j)
        else:
            i, = sample(layers[-1], k=1)
            layer.append(~i)
    layers.append(layer)

print(List("O", layers[-1]).show())
# List("O", layers[-1]).graphviz()
