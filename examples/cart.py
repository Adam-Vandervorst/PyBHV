from math import hypot
from random import randrange, seed
from bhv.native import NativePackedBHV as BHV, DIMENSION


seed(42)

C, X, Y = BHV.nrand(3)
S = 100

def embed(x, y):
    b = BHV.select_random(C, X, x)
    t = BHV.select_random(Y, X ^ Y, x)
    return BHV.select_random(b, t, y)

def embed2(x, y):
    xp = BHV.select_random(C, X, x)
    yp = BHV.select_random(C, Y, y)
    return BHV.select_rand(xp, yp)

def embed3(x, y):
    xp = (~X).select_random(X, x/2 + .25)
    yp = (~Y).select_random(Y, y/2 + .25)
    return C.select(xp, yp)


def embed4(x, y):
    br = (~Y).select_rand(X)
    bl = (~Y).select_rand(~X)
    tr = (Y).select_rand(X)
    tl = (Y).select_rand(~X)

    th = BHV.select_random(tl, tr, x)
    bh = BHV.select_random(bl, br, x)
    lv = BHV.select_random(tl, bl, y)
    rv = BHV.select_random(tr, br, y)

    return C.select(BHV.select_random(th, bh, y),
                    BHV.select_random(lv, rv, x))

grid = [[embed4(x/S, y/S)
        for y in range(S+1)]
        for x in range(S+1)]


print("euclid tl tr", 1)
print("euclid tl br", 2**.5)

print("manhat tl tr", 1)
print("manhat tl br", 2)

print("hv tl tr", grid[0][0].bit_error_rate(grid[-1][0]))
print("hv tl br", grid[0][0].bit_error_rate(grid[-1][-1]))


# for s in range(20):
#     x1, y1 = (randrange(101), randrange(101))
#     x2, y2 = (randrange(101), randrange(101))

ps = [(0, 0), (0, 1), (1, 1), (.1, 0), (0, .9), (.5, 0), (.5, .5)]

for i, (x, y) in enumerate(ps):
    ps[i] = (int(x*S), int(y*S))

for x1, y1 in ps:
    for x2, y2 in ps:
        euclid = hypot((x2 - x1)/S, (y2 - y1)/S)
        manhat = (abs(x2 - x1) + abs(y2 - y1))/S

        # hvd = grid[x1][y1].cosine(grid[x2][y2], distance=True)
        # hvd = 2*grid[x1][y1].jaccard(grid[x2][y2], distance=True)
        hvd = grid[x1][y1].bit_error_rate(grid[x2][y2])

        print((x1, y1), (x2, y2))
        print(round(euclid, 3), round(manhat, 3), round(hvd, 3))
