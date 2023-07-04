# Majority of a various number of inputs
from bhv import DIMENSION, AbstractBHV
# from bhv.np import NumPyPacked64BHV as BHV
# from bhv.np import NumPyBoolBHV as BHV
from bhv.native import CNativePackedBHV as BHV

from time import monotonic
from statistics import pstdev, fmean


repeat_pipeline = 5

sizes = list(range(1, 2000))


distances = {s: [] for s in sizes}
execution_times = {s: [] for s in sizes}

for _ in range(repeat_pipeline):
    for size in sizes:
        rs = [BHV.rand() for _ in range(size)]

        t_exec = monotonic()

        maj = BHV.majority(rs)

        execution_times[size].append(monotonic() - t_exec)

        distances[size].append([AbstractBHV.frac_to_std(r.hamming(maj)/DIMENSION, invert=True) for r in rs])


with open("results/majority_2000_native_simd.csv", 'w') as f:
    f.write("size,mean_distance,std_distance,time\n")
    for size in sizes:
        print(size)
        md = [fmean(ds) for ds in distances[size]]
        sd = [pstdev(ds) for ds in distances[size]]
        print("mean distance:", fmean(md), "+-", pstdev(md))
        print("stdev distance:", fmean(sd), "+-", pstdev(sd))
        print("timing:", fmean(execution_times[size]), "+-", pstdev(execution_times[size]))
        f.write(f"{size},{fmean(md)},{fmean(sd)},{fmean(execution_times[size])}\n")
