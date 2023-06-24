from bhv.np import NumPyBoolBHV as BHV
import numpy as np


A = BHV.rand()
B = BHV.rand()
C = BHV.rand()

A.data[:4] = np.array([0, 1, 1, 1])
B.data[:4] = np.array([0, 0, 1, 1])
C.data[:4] = np.array([0, 0, 0, 1])


M = BHV.majority3(A, B, C)

print("MAJ3  ", M.data[:4])
S = BHV.sigma3(A, B, C)
print("SIGMA3", S.data[:4])


A = BHV.rand()
B = BHV.rand()
C = BHV.rand()
D = BHV.rand()
E = BHV.rand()

A.data[:6] = np.array([0, 1, 1, 1, 1, 1])
B.data[:6] = np.array([0, 0, 1, 1, 1, 1])
C.data[:6] = np.array([0, 0, 0, 1, 1, 1])
D.data[:6] = np.array([0, 0, 0, 0, 1, 1])
E.data[:6] = np.array([0, 0, 0, 0, 0, 1])

M = BHV.majority([A, B, C, D, E])
print("MAJ5  ", M.data[:6])
S = BHV.sigma([A, B, C, D, E])
print("SIGMA ", S.data[:6])


for N in [10]*10: #range(7, 20):
    # N = 9

    VS = BHV.nrand(N)

    # for i, V in enumerate(VS):
    #     a = np.zeros(N+1, dtype=np.bool_)
    #     a[i+1:] = 1
    #     V.data[:N+1] = a

    # M = BHV.majority(VS)
    # print("MAJ   ", M.data[:N+1])
    S = BHV.sigma(VS, stdevs=-1)
    print(f"SIGMA{N}", "".join(map("01".__getitem__, S.data[:N+1])), S.active_fraction())

#  . . . . . . .
#    x  >= .5
#  .5 - e <= x <= .5 + e
