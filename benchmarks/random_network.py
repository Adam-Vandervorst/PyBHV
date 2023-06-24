# N-input M-output computational vat implementations using different primitives
# NOTE: can fail due to tight bounds on the probabilistic properties, run multiple times
# from bhv.np import NumPyPacked64BHV as BHV
from bhv.native import NativePackedBHV as BHV
from bhv.symbolic import SymbolicBHV, Var, List

from time import monotonic
from string import ascii_lowercase
from random import shuffle, random, sample
from statistics import pstdev, fmean


repeat_pipeline = 3
repeat_executions = 10

I = 50
O = 50
sizes = [200]*20 + [O]
vat_type: '"MAJ3" | "XOR NOT", "SELECT EQ"' = "XOR NOT"

names = [Var(x) for x in ascii_lowercase[:I]]

construct_times = []
execution_times = []

for _ in range(repeat_pipeline):
    initial = names
    layers = [initial]

    t_constr = monotonic()
    for size in sizes:
        layer = []
        for elem in range(size):
            if vat_type == "MAJ3":
                i, j, k = sample(layers[-1], k=3)
                layer.append(SymbolicBHV.majority3(
                        i if random() < 2/3 else ~i,
                        j if random() < 2/3 else ~j,
                        k if random() < 2/3 else ~k))
            elif vat_type == "XOR NOT":
                r = random()
                if r < .8:
                    i, j = sample(layers[-1], k=2)
                    layer.append(i ^ j)
                else:
                    i, = sample(layers[-1], k=1)
                    layer.append(~i)
            elif vat_type == "SELECT EQ":
                r = random()
                if r < 1/3:
                    i, j, k = sample(layers[-1], k=3)
                    layer.append(SymbolicBHV.select(i, j, k))
                else:
                    i, j = sample(layers[-1], k=2)
                    layer.append(~(i ^ j))
        layers.append(layer)

    cf = List(layers[-1])

    t_exec = monotonic()
    construct_times.append(t_exec - t_constr)
    for _ in range(repeat_executions):
        # the random vector gen could be separated out from the timing
        inputs = {x: BHV.rand() for x in ascii_lowercase[:I]}
        result = cf.execute(vars=inputs, bhv=BHV)
        # the checking below should be separated out from the timing
        # since it boils down to `active`
        for v in result:
            assert v.zscore() <= 4
            for w in result:
                assert v is w or not v.related(w, 9)

    execution_times.append(monotonic() - t_exec)

print("construct:", fmean(construct_times), "+-", pstdev(construct_times))
# this actual includes hypervector computations
print("execution:", fmean(execution_times), "+-", pstdev(execution_times))
