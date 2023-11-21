from bhv.abstract import AbstractBHV, BooleanAlgBHV, BaseBHV
from bhv.symbolic import Var, SymbolicBHV, Rand
from itertools import accumulate
from operator import or_, and_
from functools import cache


def logic_majority(vs):
    size = len(vs)
    half = size//2
    grid = [[None]*(half+1) for _ in range(half+1)]

    grid[half][half] = vs[size - 1]

    for i in range(half):
        v = vs[size - i - 2]
        grid[half - i - 1][half] = grid[half - i][half] & v
        grid[half][half - i - 1] = grid[half][half - i] | v

    for i in reversed(range(half)):
        for j in reversed(range(half)):
            grid[i][j] = vs[i + j].select(grid[i + 1][j], grid[i][j + 1])

    return grid[0][0]


print(logic_majority([Var.from_int(i) for i in range(9)]).show_program())
"""
def f(A, B, E, D, C):
    _0 = E | D
    _1 = _0 | C
    _2 = E & D
    _3 = C.select(_2, _0)
    _4 = B.select(_3, _1)
    _5 = _2 & C
    _6 = B.select(_5, _3)
    return A.select(_6, _4)
    
def f(A, B, C, D, I, H, G, F, E):
    _0 = I | H
    _1 = _0 | G
    _2 = _1 | F
    _3 = _2 | E
    _4 = I & H
    _5 = G.select(_4, _0)
    _6 = F.select(_5, _1)
    _7 = E.select(_6, _2)
    _8 = D.select(_7, _3)
    _9 = _4 & G
    _10 = F.select(_9, _5)
    _11 = E.select(_10, _6)
    _12 = D.select(_11, _7)
    _13 = C.select(_12, _8)
    _14 = _9 & F
    _15 = E.select(_14, _10)
    _16 = D.select(_15, _11)
    _17 = C.select(_16, _12)
    _18 = B.select(_17, _13)
    _19 = _14 & E
    _20 = D.select(_19, _15)
    _21 = C.select(_20, _16)
    _22 = B.select(_21, _17)
    return A.select(_22, _18)
"""