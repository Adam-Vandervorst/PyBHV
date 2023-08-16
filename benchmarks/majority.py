# Majority of a various number of inputs
from bhv import DIMENSION, AbstractBHV
# from bhv.np import NumPyPacked64BHV as BHV
# from bhv.np import NumPyBoolBHV as BHV
from bhv.native import CNativePackedBHV as BHV

from time import monotonic
from statistics import pstdev, fmean


repeat_pipeline = 5

sizes = list(range(1, 2000, 2))
sizes += list(range(2001, 20000, 20))
sizes += list(range(20001, 200000, 200))
sizes += list(range(200001, 1000001, 2000))


distances = {s: [] for s in sizes}
execution_times = {s: [] for s in sizes}

rs = [BHV.rand() for _ in range(1000001)]

for i in range(repeat_pipeline):
    print(f"repetition {i + 1}/{repeat_pipeline}")
    for size in sizes:
        s = rs[:size]

        t_exec = monotonic()

        maj = BHV.majority(s)

        execution_times[size].append(monotonic() - t_exec)

        distances[size].append(fmean(AbstractBHV.frac_to_std(r.hamming(maj)/DIMENSION, invert=True) for r in s))

        del s


with open("results/majority_2000_native_simd.csv", 'w') as f:
    f.write("size,distance,time\n")
    for size in sizes:
        print(size)
        print("mean distance:", fmean(distances[size]), "+-", pstdev(distances[size]))
        print("timing:", fmean(execution_times[size]), "+-", pstdev(execution_times[size]))
        f.write(f"{size},{fmean(distances[size])},{fmean(execution_times[size])}\n")
