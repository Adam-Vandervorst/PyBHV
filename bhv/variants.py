from typing import Generic, TypeVar, Type
from .abstract import *

HV = TypeVar("HV", bound=BooleanAlgBHV, covariant=True)


class LinearBHV(Generic[HV]):
    @staticmethod
    def spawn(cls: 'Type[HV]', one: 'HV.ONE', zero: 'HV.ZERO') -> (HV, HV):
        # I O  ->  P N
        # 1 0  ->  1 0  |  0 1
        r = cls.rand()
        return (r, ~r)

    @staticmethod
    def xor(self: HV, other: HV) -> (HV, HV, HV):
        # L R  ->  X L R
        # 0 0  ->  0 0 0
        # 0 1  ->  1 0 0
        # 1 0  ->  1 0 0
        # 1 1  ->  0 1 1
        b = self & other
        return (self ^ other, b, b)

    @staticmethod
    def thresholds3(cls, x: Self, y: Self, z: Self) -> (Self, Self, Self):
        # x y z  ->  + M -
        # x y z  ->  + M -
        # 0 0 0  ->  0 0 0

        # 0 0 1  ->  0 0 1
        # 0 1 0  ->  0 0 1
        # 1 0 0  ->  0 0 1

        # 0 1 1  ->  0 1 1
        # 1 0 1  ->  0 1 1
        # 1 1 0  ->  0 1 1

        # 1 1 1  ->  1 1 1
        return (x & y & z, cls.majority([x, y, z]), x | y | z)

    @staticmethod
    def and_or(self: Self, other: Self) -> (Self, Self):
        # x y  ->  & |
        # 0 0  ->  0 0
        # 0 1  ->  0 1
        # 1 0  ->  0 1
        # 1 1  ->  1 1
        return (self & other, self | other)

    @staticmethod
    def switch(self: Self, left: Self, right: Self) -> (Self, Self, Self):
        # C L R  ->  C P N
        # 0 0 0  ->  0 0 0

        # 0 0 1  ->  0 0 1
        # 0 1 0  ->  0 1 0
        # 0 1 1  ->  0 1 1

        # 1 0 1  ->  0 1 0
        # 1 1 0  ->  0 0 1
        # 1 1 1  ->  0 1 1

        # 1 1 1  ->  1 1 1

        pos = self.select(left, right)
        neg = self.select(right, left)
        return (self, pos, neg)

    @staticmethod
    def invert(self: Self, one: 'HV.ONE', zero: 'HV.ZERO') -> (Self, Self, Self):
        # P I O  ->  N A B
        # 0 1 0  ->  1 0 0
        # 1 1 0  ->  0 1 1
        (self_1, self_2, inverted) = LinearBHV.switch(self, one, zero)
        return (inverted, self_1, self_2)

    @staticmethod
    def permute(self, permutation_id: int) -> Self:
        return self.permute(permutation_id)


class MarbleBHV:
    @staticmethod
    def spawn_c(cls) -> (Self, Self, int):
        # I O  ->  P N
        # 1 0  ->  1 0  |  0 1
        r = cls.rand()
        return (r, ~r, -DIMENSION)

    @staticmethod
    def xor_c(self: Self, other: Self) -> (Self, int):
        # L R  ->  X
        # 0 0  ->  0
        # 0 1  ->  1
        # 1 0  ->  1
        # 1 1  ->  0
        b = self.overlap(other)
        return (self ^ other, 2*b)

    @staticmethod
    def majority3_c(cls, x: Self, y: Self, z: Self) -> (Self, int):
        # x y z  ->  M
        # 0 0 0  ->  0

        # 0 0 1  ->  0
        # 0 1 0  ->  0
        # 1 0 0  ->  0

        # 0 1 1  ->  1
        # 1 0 1  ->  1
        # 1 1 0  ->  1

        # 1 1 1  ->  1
        return (cls.majority([x, y, z]), (x & y & z).active() + (x | y | z).active())

    @staticmethod
    def and_or_c(self: Self, other: Self) -> (Self, Self, int):
        # x y  ->  & |
        # 0 0  ->  0 0
        # 0 1  ->  0 1
        # 1 0  ->  0 1
        # 1 1  ->  1 1
        return (self & other, self | other, 0)

    @staticmethod
    def switch_c(self: Self, left: Self, right: Self) -> (Self, Self, Self, int):
        # C L R  ->  C P N
        # 0 0 0  ->  0 0 0

        # 0 0 1  ->  0 0 1
        # 0 1 0  ->  0 1 0
        # 0 1 1  ->  0 1 1

        # 1 0 1  ->  0 1 0
        # 1 1 0  ->  0 0 1
        # 1 1 1  ->  0 1 1

        # 1 1 1  ->  1 1 1

        pos = self.select(left, right)
        neg = self.select(right, left)
        return (self, pos, neg, 0)

    @staticmethod
    def invert_c(self: Self) -> (Self, int):
        # P I O  ->  N A B
        # 0 1 0  ->  1 0 0
        # 1 1 0  ->  0 1 1
        return (~self, 2*(self.active() - DIMENSION/2))

    @staticmethod
    def permute_c(self, permutation_id: int) -> (Self, int):
        return (self.permute(permutation_id), 0)

