from .shared import *
from statistics import NormalDist


class AbstractBHV:
    # recall
    # information_entropy(p) = -p*log2(p) - (1 - p)*log2(1 - p)
    # so for some active fractions, it should be a lot cheaper than for others
    # hence a specialized method for 2^-n
    @classmethod
    def rand(cls) -> Self:
        raise NotImplementedError()

    @classmethod
    def nrand(cls, n) -> list[Self]:
        return [cls.rand() for _ in range(n)]

    @classmethod
    def rand2(cls, power=1) -> Self:
        r = cls.rand()
        return r if power == 1 else r & cls.rand2(power - 1)

    @classmethod
    def nrand2(cls, n, power=1) -> list[Self]:
        return [cls.rand2(power) for _ in range(n)]

    @classmethod
    def random(cls, active=0.5) -> Self:
        raise NotImplementedError()

    @classmethod
    def nrandom(cls, n, active=0.5) -> list[Self]:
        return [cls.random(active) for _ in range(n)]

    @classmethod
    def majority(cls, vs: list[Self]) -> Self:
        raise NotImplementedError()

    def __eq__(self, other: Self) -> bool:
        raise NotImplementedError()

    def __xor__(self, other: Self) -> Self:
        raise NotImplementedError()

    def __and__(self, other: Self) -> Self:
        raise NotImplementedError()

    def __or__(self, other: Self) -> Self:
        raise NotImplementedError()

    def __invert__(self) -> Self:
        raise NotImplementedError()

    def select(self, when1: Self, when0: Self) -> Self:
        return when0 ^ (self & (when0 ^ when1))

    def mix(self, other: Self, pick_l=0.5) -> Self:
        return self.random(pick_l).select(self, other)

    def active(self) -> int:
        raise NotImplementedError()

    def bias_rel(self, other: Self, rel: Self) -> float:
        rel_l = rel.select(self, self.ZERO).active()
        rel_r = rel.select(other, self.ZERO).active()
        return rel_l / (rel_l + rel_r)

    def active_fraction(self) -> float:
        return self.active()/DIMENSION

    def hamming(self, other: Self) -> int:
        return (self ^ other).active()

    def bit_error_rate(self, other: Self) -> float:
        return (self ^ other).active_fraction()

    def jaccard(self, other: Self) -> float:
        return 1. - float((self & other).active()) / float((self | other).active() + 1E-7)

    def cosine(self, other: Self) -> float:
        return 1 - float((self & other).active()) / float(self.active() * other.active() + 1E-7)**.5

    def zscore(self, other: Self) -> float:
        p = 0.5
        n = NormalDist(DIMENSION*p, (DIMENSION*p*(1 - p))**.5)
        return n.zscore(self.hamming(other))

    def pvalue(self, other: Self) -> float:
        p = 0.5
        n = NormalDist(DIMENSION*p, (DIMENSION*p*(1 - p))**.5)
        s = n.cdf(self.hamming(other))
        return 2*min(s, 1 - s)

    def sixsigma(self, other: Self) -> bool:
        return abs(self.zscore(other)) < 6

    @classmethod
    def _majority5_simplified(cls, x4: Self, x3: Self, x2: Self, x1: Self, x0: Self) -> Self:
        t1 = x1 | x0
        b1 = x1 & x0
        m2 = x2.select(t1, b1)
        t2 = x2 | t1
        b2 = x2 & b1
        m3h = x3.select(t2, m2)
        m3l = x3.select(m2, b2)
        m4 = x4.select(m3h, m3l)
        return m4

    @classmethod
    def _majority7_simplified(cls, x6: Self, x5: Self, x4: Self, x3: Self, x2: Self, x1: Self, x0: Self) -> Self:
        t1 = x1 | x0
        b1 = x1 & x0
        m2 = x2.select(t1, b1)
        t2 = x2 | t1
        b2 = x2 & b1
        m3h = x3.select(t2, m2)
        m3l = x3.select(m2, b2)
        m4 = x4.select(m3h, m3l)
        t3 = x3 | t2
        b3 = x3 & b2
        m4h = x4.select(t3, m3h)
        m4l = x4.select(m3l, b3)
        m5h = x5.select(m4h, m4)
        m5l = x5.select(m4, m4l)
        m6 = x6.select(m5h, m5l)
        return m6

    @classmethod
    def _majority9_simplified(cls, x8: Self, x7: Self, x6: Self, x5: Self, x4: Self, x3: Self, x2: Self, x1: Self, x0: Self) -> Self:
        t1 = x0 | x1
        b1 = x0 & x1
        m2 = x2.select(t1, b1)
        t2 = x2 | t1
        b2 = x2 & b1
        m3h = x3.select(t2, m2)
        m3l = x3.select(m2, b2)
        t3 = x3 | t2
        b3 = x3 & b2
        m4h = x4.select(t3, m3h)
        m4 = x4.select(m3h, m3l)
        m4l = x4.select(m3l, b3)
        t4 = x4 | t3
        b4 = x4 & b3
        m5hh = x5.select(t4, m4h)
        m5h = x5.select(m4h, m4)
        m5l = x5.select(m4, m4l)
        m5ll = x5.select(m4l, b4)
        m6h = x6.select(m5hh, m5h)
        m6 = x6.select(m5h, m5l)
        m6l = x6.select(m5l, m5ll)
        m7h = x7.select(m6h, m6)
        m7l = x7.select(m6, m6l)
        m8 = x8.select(m7h, m7l)
        return m8

    ZERO: Self
    ONE: Self
