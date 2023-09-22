from bhv.np import NumPyPacked64BHV as BHV

from functools import reduce


if __name__ == '__main__':
    vs = BHV.nrand(5)

    m = BHV.majority(vs)
    mk = [BHV.majority(vs[:i] + vs[i+1:]) for i in range(5)]

    print(m.std_apart(BHV.majority(mk)))
    print([m.select_rand(~vs[i]).std_apart(k) for i, k in enumerate(mk)])
    print([m.std_apart(k) for i, k in enumerate(mk)])
    print([v.std_apart(m) for v in vs])

    # m_min_0 = mk[0]
    m_min_0 = m.select_rand(~vs[0])

    print([v.std_apart(m_min_0) for v in vs])

    # m_min_01 = mk[0] & mk[1]
    # m_min_01 = BHV.select_random(mk[0], mk[1], 1/2)
    m_min_01 = BHV.majority3(~vs[0], ~vs[1], m)

    print([v.std_apart(m_min_01) for v in vs])

    # m_min_012 = mk[0] & mk[1] & mk[2]
    # m_min_012 = BHV.majority([mk[0], mk[1], mk[2]])
    m_min_012 = BHV.majority([~vs[0], ~vs[1], ~vs[2], m])

    print([v.std_apart(m_min_012) for v in vs])

    m_min_0123 = BHV.majority([~vs[0], ~vs[1], ~vs[2], ~vs[3], m])

    print([v.std_apart(m_min_0123) for v in vs])

    m_min_01234 = BHV.majority([~vs[0], ~vs[1], ~vs[2], ~vs[3], ~vs[4], m])

    print([v.std_apart(m_min_01234) for v in vs])
