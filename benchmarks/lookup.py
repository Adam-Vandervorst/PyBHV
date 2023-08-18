# Similarity search
# from bhv.np import NumPyPacked64BHV as BHV
from bhv.native import NativePackedBHV as BHV
from bhv.lookup import StoreList, StoreRejectList, StoreChunks

from time import monotonic
from random import shuffle, random, randrange, sample
from statistics import pstdev, fmean


repeat_pipeline = 5
repeat_lookup = 10
thresholds = [0, 3, 6]
deviations = [0, 1, 2, 4]
sizes = [20, 200, 2000, 20000]

# e.g.
# hvs=[10, 20, 30, 50, 100]
# v=20 + 5
# threshold=10
# returns 20, 30

index_times = {s: [] for s in sizes}
lookup_times = {s: {t: [] for t in thresholds} for s in sizes}
distances = {s: {t: {d: [] for d in deviations} for t in thresholds} for s in sizes}

for _ in range(repeat_pipeline):
    for size in sizes:
        rs = BHV.nrand(size)
        ps = {deviation: [r.flip_frac(BHV.std_to_frac(deviation))
                          for r in sample(rs, repeat_lookup)]
              for deviation in deviations}

        t_index = monotonic()
        index = StoreChunks.auto_associative(rs)
        index_times[size].append(monotonic() - t_index)

        for threshold in thresholds:
            t_lookup = monotonic()
            for deviation in deviations:
                for p in ps[deviation]:
                    ms = list(index.related(p, threshold))
                    distances[size][threshold][deviation].append([index.hvs[m].std_apart(p) for m in ms])

            lookup_times[size][threshold].append(monotonic() - t_lookup)


for size in sizes:
    print("size", size)
    for threshold in thresholds:
        print(" threshold", threshold)
        print(" lookup", fmean(lookup_times[size][threshold]), "+-", pstdev(lookup_times[size][threshold]))
        for deviation in deviations:
            print("  deviation", deviation)
            print("  results", distances[size][threshold][deviation])
