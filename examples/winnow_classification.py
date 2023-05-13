# require NumPy
from bhv.np import NumPyBoolBHV as BHV, DIMENSION
import numpy as np

N = 22
EPOCHS = 30

xs = BHV.nrand(N)
ys = [i % 2 == 0 for i in range(N)]

threshold = DIMENSION//2
alpha = 2
ws = np.ones(DIMENSION, dtype=np.uint16)

for i in range(EPOCHS):
    right = 0
    for x, y in zip(xs, ys):
        total = np.dot(x.data, ws)
        prediction = total > threshold
        right += prediction == y
        if prediction != y:
            if y:
                ws = np.where(x.data == 1, ws*alpha, ws)
            else:
                ws = np.where(x.data == 1, ws/alpha, ws)
                # ws = np.where(x.data == 1, np.zeros_like(ws), ws)
    print(f"acc {right/N}")
