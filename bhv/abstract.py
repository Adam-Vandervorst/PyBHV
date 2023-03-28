from .shared import *


class AbstractBHV:
    # recall
    # information_entropy(p) = -p*log2(p) - (1 - p)*log2(1 - p)
    # so for some active fractions, it should be a lot cheaper than for others
    # hence a specialized method for 2^-n
    @classmethod
    def rand(cls) -> Self:
        raise NotImplementedError()

    @classmethod
    def nrand(cls, n) -> Self:
        return [cls.rand() for _ in range(n)]

    @classmethod
    def rand2(cls, power=1) -> Self:
        r = cls.rand()
        return r if power == 1 else r & cls.rand2(power - 1)

    @classmethod
    def nrand2(cls, n, power=1) -> Self:
        return [cls.rand2(power) for _ in range(n)]

    @classmethod
    def random(cls, active=0.5) -> Self:
        raise NotImplementedError()

    @classmethod
    def nrandom(cls, n, active=0.5) -> list[Self]:
        return [cls.random(active) for _ in range(n)]

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

    def bit_error_rate(self, other: Self) -> float:
        return (self ^ other).active_fraction()

    def jaccard(self, other: Self) -> float:
        return 1. - float((self & other).active()) / float((self | other).active())

    def cosine(self, other: Self) -> float:
        return 1 - float((self & other).active()) / float(self.active() * other.active())**.5

    ZERO: Self
    ONE: Self
