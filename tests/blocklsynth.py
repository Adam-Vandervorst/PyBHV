from bhv.symbolic import SymbolicBHV as BHV, Var, List
from bhv.slice import Slice
from bhv.shared import bin_bitmask, nbs
from string import ascii_lowercase
from random import shuffle, sample, random, randrange
from statistics import pstdev, fmean


I = 8
O = 16
names = [Var(x) for x in ascii_lowercase[:I]]
fs = []

for j in range(O):
    target = [i % 2 == 0 for i in range(2**I)]
    shuffle(target)

    f = BHV.synth(names, target).simplify()
    assert target == f.truth_assignments(names)
    fs.append(f)

# print([f.expected_active_fraction(vars={x: .5 for x in ascii_lowercase[:I]}) for f in fs])
# print(List("O", fs).show())
# List("O", fs).graphviz(structural=True)


initial = names
layers = [initial]
# sizes = [20, 20, 20, 20, O]
sizes = [1]*30 + [O]


# for size in sizes:
#     layer = []
#     for elem in range(size):
#         r = random()
#         i, j, k = sample(layers[-1], k=3)
#
#         layer.append(BHV.majority3(
#                 i if random() < .666 else ~i,
#                 j if random() < .666 else ~j,
#                 k if random() < .666 else ~k))
#
#     layers.append(layer)

# for size in sizes:
#     layer = []
#     for elem in range(size):
#         r = random()
#         if r < .75:
#             i, j = sample(layers[-1], k=2)
#             layer.append(i ^ j)
#         else:
#             i, = sample(layers[-1], k=1)
#             layer.append(~i)
#     layers.append(layer)

# for size in sizes:
#     layer = []
#     for elem in range(size):
#         i, j, k = sample(layers[-1], k=3)
#         layer.append(BHV.select(i, ~j, k))
#     layers.append(layer)

print([f.expected_active_fraction(vars={x: .5 for x in ascii_lowercase[:I]}) for f in layers[-1]])
print(List(layers[-1]).size())
# List("O", layers[-1]).graphviz(compact_select=False)

for fk in [fs, layers[-1]]:
    truth_table = [fi.truth_assignments(names) for fi in fk]

    truth_tables_sims = [[sum(a ^ a_ for a, a_ in zip(ta, ta_)) for ta_ in truth_table] for ta in truth_table]
    mean_stdev = [(abs(.5 - fmean(diffs)/2**I), pstdev(diffs)/2**I) for diffs in truth_tables_sims]
    print(max(x for x, _ in mean_stdev), max(y for _, y in mean_stdev))

    truth_tables_sens = [fmean(sum(ta[i] ^ ta[j] for j in nbs(i, I))/I for i in range(2**I)) for ta in truth_table]
    print(abs(.5 - fmean(truth_tables_sens)), pstdev(truth_tables_sens))



    # for assignments in truth_table:
    #     print(bin_bitmask(assignments))

"""
8, 16

ref
0.04296875 0.125760625250041
0.0008544921875 0.019445229885344226

maj3 not [20, 20, 20, 20, O]
0.12060546875 0.2121091726734187
0.162841796875 0.03379297840161043

maj3 not [200, 200, 200, 200, O]
0.080078125 0.15209000803614153
0.120849609375 0.030429560571682487

xor not [20, 20, 20, 20, O]
0.0625 0.16535945694153692
0.0078125 0.12077831901359615

xor not [200, 200, 200, 200, O]
0.03125 0.12103072956898178
0.0 0.1875

ite not [200, 200, 200, 200, O]
0.05029296875 0.13149097664047601
0.0791015625 0.025633603024231812

ite not [20, 20, 20, 20, O]
0.1044921875 0.1579708702179055
0.124267578125 0.030601447254438183

maj3-9 1/3not [20, 20, 20, 20, O]
0.07861328125 0.18325514715650032
0.091552734375 0.03555998508028247
"""