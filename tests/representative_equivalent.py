from bhv.native import NativePackedBHV as BHV, DIMENSION
# from bhv.symbolic import Var, SymbolicBHV
# from bhv.np import NumPyBoolBHV as BHV, DIMENSION
import numpy as np
from random import choices


def L(vs):
    cls = type(vs[0])
    n = len(vs)

    def smallest_power2_larger(x):
        # returns log2(x) on x power of 2, else its adds the smallest d such that log2(x + d) is integer
        k = 0
        while 2 ** k < x:
            k += 1
        return k

    circumscribed = smallest_power2_larger(n)
    inscribed = circumscribed - 1

    rs = cls.nrand(circumscribed)

    def interval(p):
        lower = (int(p, 2) if p else 0)*2**(circumscribed - len(p))
        upper = min(n, lower + 2**(circumscribed - len(p)))
        return lower, upper

    def length(lu):
        d = lu[1] - lu[0]
        if d >= 1:
            return d
        else:
            return 0

    def si(p, l, circ, n):


        # if l == 0:
        #     return n
        if l > circ:
            return 0

        lower = p*2**(circ - l)
        upper = min(n, lower + 2**(circ - l))

        # if p == 0:
        #     return min(n, 2**(circumscribed - l))
        #
        # if p*2**(circumscribed - l) < n < (p + 1)*2**(circumscribed - l):
        #     return n - p*2**(circumscribed - l)

        d = upper - lower
        if lower < n:
            return d
        else:
            return 0

    def rec(p, pn, depth):
        L0, L1 = length(interval(p + '0')), length(interval(p + '1'))
        # L0, L1 = si(pn << 1, depth + 1, circumscribed, n), si(1 | (pn << 1), depth + 1, circumscribed, n)

        # print(L0, L1)
        # print(f"assert(si({pn}, {depth}, {circumscribed}, {n}) == {si(pn, depth, circumscribed, n)});")
        # print(f"assert(si(1 | ({pn} << 1), {depth} + 1, {circumscribed}, {n}) == {si(1 | (pn << 1), depth + 1, circumscribed, n)});")
        # assert si(pn, depth, circumscribed, n) == length(interval(p)), f"'{p}', {si(pn, depth, circumscribed, n)}, {length(interval(p))}, {interval(p)}, {n}"
        # assert si(pn << 1, depth + 1, circumscribed, n) == L0
        # assert si(1 | (pn << 1), depth + 1, circumscribed, n) == L1, f"'{p}', {si(1 | (pn << 1), depth + 1, circumscribed, n)}, {L1}, {interval(p + '1')}, {n}"

        # print(p, len(p), interval(p), interval(p + '0'), interval(p + '1'))

        if L0 and L1:
            # TODO could be replaced by fraction (created with AND, OR, and RAND)
            d = rs[len(p)] if L0 == L1 else cls.random(L1 / (L0 + L1))
            return d.select(rec(p + '1', 1 | (pn << 1), depth + 1), rec(p + '0', pn << 1, depth + 1))
        elif L0:
            return rec(p + '0', pn << 1, depth + 1)
        elif L1:
            return rec(p + '1', 1 | (pn << 1), depth + 1)
        else:
            i = int(p, 2)
            return vs[i]

    return rec('', 0, 0)


def _L(vs):
    cls = type(vs[0])
    n = len(vs)

    def smallest_power2_larger(x):
        # returns log2(x) on x power of 2, else its adds the smallest d such that log2(x + d) is integer
        k = 0
        while 2 ** k < x:
            k += 1
        return k

    circumscribed = smallest_power2_larger(n)
    inscribed = circumscribed - 1

    rs = cls.nrand(circumscribed)

    def interval(p):
        lower = (int(p, 2) if p else 0)*2**(circumscribed - len(p))
        upper = min(n, lower + 2**(circumscribed - len(p)))
        return lower, upper

    def length(lu):
        d = lu[1] - lu[0]
        if d >= 1:
            return d
        else:
            return 0

    def rec(p):
        L0, L1 = length(interval(p + '0')), length(interval(p + '1'))

        if L0 and L1:
            # TODO could be replaced by fraction (created with AND, OR, and RAND)
            d = rs[len(p)] if L0 == L1 else cls.random(L1 / (L0 + L1))
            return d.select(rec(p + '1'), rec(p + '0'))
        elif L0:
            return rec(p + '0')
        elif L1:
            return rec(p + '1')
        else:
            i = int(p, 2)
            return vs[i]

    # generations = [[(i, i + 1) for i in range(n)]]
    # factors = []
    #
    # while len(generations[-1]) != 1:
    #     front = [e for e in generations[-1]]
    #     front.reverse()
    #     follow = []
    #     ffolow = []
    #
    #     while front:
    #         a = front.pop()
    #         try:
    #             b = front.pop()
    #             follow.append((a[0], b[1]))
    #             da = a[1] - a[0]
    #             db = b[1] - b[0]
    #             ffolow.append(f"{db}/{da + db}")
    #         except IndexError:
    #             follow.append(a)
    #
    #     generations.append(follow)
    #     factors.append(ffolow)
    #
    # for gen in generations:
    #     print(gen)
    #
    # for gen in factors:
    #     print(gen)

    size = 2*(2**circumscribed)
    arr = size*[None]
    lows = size*[None]
    highs = size*[None]

    for i in range(size - 1, 0, -1):
        if size - n <= i:
            arr[i] = vs[size - i - 1]
            lows[i] = size - i - 1
            highs[i] = size - i
        elif (2*i + 1) < size:
            if lows[2*i] is not None:
                da = highs[2*i] - lows[2*i]
                db = highs[2*i + 1] - lows[2*i + 1]

                lows[i] = lows[2*i + 1]
                highs[i] = highs[2*i]

                cond = rs[smallest_power2_larger(2*i + 1) - 2] if da == db else cls.random(db/(da + db))
                arr[i] = cond.select(arr[2*i + 1], arr[2*i])
            else:
                lows[i] = lows[2*i + 1]
                highs[i] = highs[2*i + 1]
                arr[i] = arr[2*i + 1]

# [None, (0, 13), (8, 13), (0, 8), (12, 13), (8, 12), (4, 8), (0, 4), None, (12, 13), (10, 12), (8, 10), (6, 8), (4, 6), (2, 4), (0, 2), None, None, None, (12, 13), (11, 12), (10, 11), (9, 10), (8, 9), (7, 8), (6, 7), (5, 6), (4, 5), (3, 4), (2, 3), (1, 2), (0, 1)]
# [None, (0, 2), (8, 10), (0, 2), (12, 13), (8, 10), (4, 6), (0, 2), None, (12, 13), (10, 12), (8, 10), (6, 8), (4, 6), (2, 4), (0, 2), None, None, None, (12, 13), (11, 12), (10, 11), (9, 10), (8, 9), (7, 8), (6, 7), (5, 6), (4, 5), (3, 4), (2, 3), (1, 2), (0, 1)]
#     print(lows)
#     print(highs)
    # print(arr[1])
    # print(arr[1].show_program())
    # quit()
    # return arr[1]
    return rec('')

# all_vars = [Var.shortname(i) for i in range(0, 64)]
# print(len(all_vars), all_vars)
# res = L(all_vars)
# print(res.show_program(impl="BHV."))
#
#
# print("digraph {")
# print("rankdir = BT;")
# res.graphviz(structural=False)
# print("}")


# quit()

R = BHV.representative

tvs = BHV.nrand(3)
print(R(tvs).distribution(tvs, metric=BHV.bit_error_rate))
print(L(tvs).distribution(tvs, metric=BHV.bit_error_rate))


def report(vs):
    mrs = np.array(R(vs).distribution(vs, metric=BHV.bit_error_rate))
    mrs_ = np.array(R(vs).distribution(vs, metric=BHV.bit_error_rate))
    lrs = np.array(L(vs).distribution(vs, metric=BHV.bit_error_rate))
    ds = mrs - lrs
    ds_ = mrs - mrs_
    avg_ds = np.mean(ds)
    min_ds = np.min(ds)
    max_ds = np.max(ds)
    avg_ads = np.mean(np.absolute(ds))

    avg_ds_ = np.mean(ds_)
    min_ds_ = np.min(ds_)
    max_ds_ = np.max(ds_)
    avg_ads_ = np.mean(np.absolute(ds_))

    return (min_ds, avg_ds, max_ds, avg_ads), (min_ds_, avg_ds_, max_ds_, avg_ads_)

def test(gen, on):
    control = []
    testing = []
    for i in on:
        vs = gen(i)
        c, t = report(vs)
        control.append(c)
        testing.append(t)
    return np.array(control).T, np.array(testing).T

def correlated_vs(i):
    rs = np.absolute(np.random.randn(i))
    rs /= rs.max()
    v = BHV.rand()
    return [v ^ (BHV.random(r) & vk) for r, vk in zip(rs, BHV.nrand(i))]

def shared_vs(i):
    pool = BHV.nrand(i)
    return choices(pool, k=i)

import matplotlib.pyplot as plt

RANGE = list(range(2, 300))

titles = ["uncorrelated", "correlated", "shared"]
data = [test(BHV.nrand, RANGE), test(correlated_vs, RANGE), test(shared_vs, RANGE)]

fig, axes = plt.subplots(1, len(titles), figsize=(16, 6), sharey="all")

for ax, title, data in zip(axes, titles, data):
    ax.plot(RANGE, data[0][0], color='orange', linewidth=1, linestyle='-')
    ax.plot(RANGE, data[0][1], color='orange', linewidth=2, linestyle='-', label='AVG testing')
    ax.plot(RANGE, data[0][2], color='orange', linewidth=1, linestyle='-')
    ax.plot(RANGE, data[0][3], color='orange', linewidth=2, linestyle='--', label='MAE testing')

    ax.plot(RANGE, data[1][0], color='green', linewidth=1, linestyle='-')
    ax.plot(RANGE, data[1][1], color='green', linewidth=2, linestyle='-', label='AVG control')
    ax.plot(RANGE, data[1][2], color='green', linewidth=1, linestyle='-')
    ax.plot(RANGE, data[1][3], color='green', linewidth=2, linestyle='--', label='MAE control')

    ax.set_ylim(bottom=-.04, top=.04)
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.yaxis.grid(True, which='major', linestyle=':', linewidth=0.25)

fig.suptitle('Representative Hypervector Operation', fontsize=16, y=.95)
fig.text(0.5, 0.02, '#inputs', ha='center', va='center', fontsize=12)
fig.text(0.01, 0.5, 'BER - E[BER]', ha='center', va='center', rotation='vertical', fontsize=12)

plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.savefig("representative.png", dpi=600)
