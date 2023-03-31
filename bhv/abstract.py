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

    ZERO: Self
    ONE: Self
