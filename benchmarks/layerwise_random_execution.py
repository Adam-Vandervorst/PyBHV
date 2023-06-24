# N-input M-output stack of random computation (from different families)
# NOTE: can fail due to tight bounds on the probabilistic properties, run multiple times
# from bhv.np import NumPyPacked64BHV as BHV
from bhv.native import CNativePackedBHV as BHV

from time import monotonic
from random import shuffle, random, randrange, sample
from statistics import pstdev, fmean
from multiprocessing import Pool


repeat_pipeline = 3

I = 50
O = 50
sizes = [2000]*20 + [O]
vat_type: '"MAJ3" | "XOR NOT", "SELECT EQ"' = "MAJ3"


if vat_type == "MAJ3":
    def execute(_):
        i, j, k = sample(values[-1], k=3)
        return BHV.majority([
                i if random() < 2/3 else ~i,
                j if random() < 2/3 else ~j,
                k if random() < 2/3 else ~k])
elif vat_type == "XOR NOT":
    def execute(_):
        r = random()
        if r < .8:
            i, j = sample(values[-1], k=2)
            return i ^ j
        else:
            i, = sample(values[-1], k=1)
            return ~i
elif vat_type == "SELECT EQ":
    def execute(_):
        r = random()
        if r < 1/3:
            i, j, k = sample(values[-1], k=3)
            return BHV.select(i, j, k)
        else:
            i, j = sample(values[-1], k=2)
            return ~(i ^ j)


execution_times = []

for _ in range(repeat_pipeline):
    t_exec = monotonic()

    value = [BHV.rand() for _ in range(I)]
    values = [value]

    with Pool() as p:
        for size in sizes:
            values.append(p.map(execute, range(size)))
    # for size in sizes:
    #     values.append(list(map(execute, range(size))))

    result = values[-1]

    for v in result:
        assert v.zscore() <= 4
        for w in result:
            assert v is w or not v.related(w, 9)

    execution_times.append(monotonic() - t_exec)

print("execution:", fmean(execution_times), "+-", pstdev(execution_times))
