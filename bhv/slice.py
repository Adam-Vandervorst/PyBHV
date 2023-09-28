from .abstract import *
import random


class Slice(AbstractBHV):
    def __init__(self, b):
        self.b = b

    def __repr__(self):
        return f"Slice({self.b})"

    @classmethod
    def rand(cls) -> Self:
        return cls.random(.5)

    @classmethod
    def random(cls, active: float) -> Self:
        assert 0. <= active <= 1.
        return Slice(random.random() > active)

    def permute(self, permutation_id: int) -> Self:
        raise NotImplementedError()

    def swap_halves(self) -> Self:
        raise NotImplementedError()

    def rehash(self) -> Self:
        raise NotImplementedError()

    def __eq__(self, other: Self) -> bool:
        return self.b == other.b

    def __xor__(self, other: Self) -> Self:
        return Slice(self.b ^ other.b)

    def __and__(self, other: Self) -> Self:
        return Slice(self.b & other.b)

    def __or__(self, other: Self) -> Self:
        return Slice(self.b | other.b)

    def __invert__(self) -> Self:
        return Slice(not self.b)

    @classmethod
    def _logic_majority(cls, vs: list[Self]) -> Self:
        return Slice(sum(v.b for v in vs) > len(vs)//2)

    def active(self) -> int:
        return int(self.b)

    @classmethod
    def threshold(cls, vs: list[Self], t: int) -> Self:
        return Slice(sum(v.b for v in vs) > t)
Slice.ONE = Slice(True)
Slice.ZERO = Slice(False)
