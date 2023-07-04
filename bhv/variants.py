from .abstract import *


class LinearBHV(BooleanAlgBHV):
    @classmethod
    def spawn(cls, one: BooleanAlgBHV.ONE, zero: BooleanAlgBHV.ZERO) -> (Self, Self):
        # I O  ->  P N
        # 1 0  ->  1 0  |  0 1
        r = cls.rand()
        return (r, ~r)

    def xor(self: Self, other: Self) -> (Self, Self, Self):
        # L R  ->  X L R
        # 0 0  ->  0 0 0
        # 0 1  ->  1 0 0
        # 1 0  ->  1 0 0
        # 1 1  ->  0 1 1
        b = self & other
        return (self ^ other, b, b)

    @classmethod
    def thresholds3(cls, x: Self, y: Self, z: Self) -> (Self, Self, Self):
        # x y z  ->  + M -
        # 0 0 0  ->  0 0 0
        # 0 0 1  ->  0 0 1
        # 0 1 1  ->  0 1 1
        # 1 1 1  ->  1 1 1
        return (x & y & z, cls.majority([x, y, z]), x | y | z)

    def and_or(self: Self, other: Self) -> (Self, Self):
        # x y  ->  & |
        # 0 0  ->  0 0
        # 0 1  ->  0 1
        # 1 0  ->  0 1
        # 1 1  ->  1 1
        return (self & other, self | other)

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

    def invert(self: Self, one: BooleanAlgBHV.ONE, zero: BooleanAlgBHV.ZERO) -> (Self, Self, Self):
        # P I O  ->  N A B
        # 0 1 0  ->  1 0 0
        # 1 1 0  ->  0 1 1
        return self.switch(one, zero)

    # def permute(self, permutation_id: int) -> Self


class MarbleBHV(BooleanAlgBHV):
    @classmethod
    def spawn_c(cls) -> (Self, Self, int):
        # I O  ->  P N
        # 1 0  ->  1 0  |  0 1
        r = cls.rand()
        return (r, ~r, -DIMENSION)

    def xor_c(self: Self, other: Self) -> (Self, int):
        # L R  ->  X
        # 0 0  ->  0
        # 0 1  ->  1
        # 1 0  ->  1
        # 1 1  ->  0
        b = self.hamming(other)
        return (self ^ other, 2*b)

    @classmethod
    def majority3_c(cls, x: Self, y: Self, z: Self) -> (Self, int):
        # x y z  ->  + M -
        # 0 0 0  ->  0 0 0
        # 0 0 1  ->  0 0 1
        # 0 1 1  ->  0 1 1
        # 1 1 1  ->  1 1 1
        return (cls.majority([x, y, z]), (x & y & z).active() + (x | y | z).active())

    def and_or_c(self: Self, other: Self) -> (Self, Self, int):
        # x y  ->  & |
        # 0 0  ->  0 0
        # 0 1  ->  0 1
        # 1 0  ->  0 1
        # 1 1  ->  1 1
        return (self & other, self | other, 0)

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

    def invert_c(self: Self) -> (Self, int):
        # P I O  ->  N A B
        # 0 1 0  ->  1 0 0
        # 1 1 0  ->  0 1 1
        return (~self, 2*(self.active() - DIMENSION/2))

    def permute_c(self, permutation_id: int) -> (Self, int):
        return (self.permute(permutation_id), 0)

