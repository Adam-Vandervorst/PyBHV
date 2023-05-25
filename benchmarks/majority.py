# Majority of a various number of inputs
from bhv.np import NumPyPacked64BHV as BHV
# from bhv.np import NumPyBoolBHV as BHV
from bhv.symbolic import SymbolicBHV, Var, List

from time import monotonic
from string import ascii_lowercase
from random import shuffle, random, randrange, sample
from statistics import pstdev, fmean
from multiprocessing import Pool


repeat_pipeline = 5

sizes = list(range(1, 2000))


distances = {s: [] for s in sizes}
execution_times = {s: [] for s in sizes}

for _ in range(repeat_pipeline):
    for size in sizes:
        rs = BHV.nrand(size)

        t_exec = monotonic()

        maj = BHV.majority(rs)

        execution_times[size].append(monotonic() - t_exec)

        distances[size].append([r.std_apart(maj, invert=True) for r in rs])


with open("results/majority_2000_np_packed64.csv", 'w') as f:
    f.write("size,mean_distance,std_distance,time\n")
    for size in sizes:
        print(size)
        md = [fmean(ds) for ds in distances[size]]
        sd = [pstdev(ds) for ds in distances[size]]
        print("mean distance:", fmean(md), "+-", pstdev(md))
        print("stdev distance:", fmean(sd), "+-", pstdev(sd))
        print("timing:", fmean(execution_times[size]), "+-", pstdev(execution_times[size]))
        f.write(f"{size},{fmean(md)},{fmean(sd)},{fmean(execution_times[size])}\n")
