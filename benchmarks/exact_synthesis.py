# execution of generated N-input M-output boolean functions
# currently the synthesis and reduction rules make the program consist of
# XOR, NOT, AND, OR
# NOTE: takes about a minute and doesn't show intermediate results
from bhv.np import NumPyPacked64BHV as BHV
# from bhv.native import NativePackedBHV as BHV
from bhv.symbolic import SymbolicBHV, Var, List

from time import monotonic
from string import ascii_lowercase
from random import shuffle
from statistics import pstdev, fmean


repeat_pipeline = 3
repeat_executions = 10

I = 8
O = 16

names = [Var(x) for x in ascii_lowercase[:I]]

synthesis_times = []
optimization_times = []
execution_times = []

for _ in range(repeat_pipeline):
    t_synth = monotonic()
    fs = []
    for j in range(O):
        target = [i % 2 == 0 for i in range(2**I)]
        shuffle(target)
        f = SymbolicBHV.synth(names, target)
        assert target == f.truth_assignments(names)
        fs.append(f)

    t_opt = monotonic()
    synthesis_times.append(t_opt - t_synth)
    cf = List(fs)
    print(cf.size())
    # could be commented out for more load on execution
    cf = cf.simplify(expand_select_and_or=True)
    print(cf.size())
    # could be commented out for more load on execution
    cf = cf.optimal_sharing()
    print(cf.size())

    t_exec = monotonic()
    optimization_times.append(t_exec - t_opt)
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

print("synth:", fmean(synthesis_times), "+-", pstdev(synthesis_times))
print("optim:", fmean(optimization_times), "+-", pstdev(optimization_times))
# only this actual includes hypervector computations
print("execu:", fmean(execution_times), "+-", pstdev(execution_times))
