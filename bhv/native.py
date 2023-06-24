from bhv.abstract import AbstractBHV, DIMENSION
from bhv.cnative import CNativePackedBHV, _DIMENSION

assert DIMENSION == _DIMENSION()


class NativePackedBHV(AbstractBHV):
    def __init__(self, native_bhv):
        self.ins = native_bhv

    @classmethod
    def rand(cls):
        return NativePackedBHV(CNativePackedBHV.rand())

    @classmethod
    def random(cls, p):
        return NativePackedBHV(CNativePackedBHV.random(p))

    def __or__(self, other):
        return NativePackedBHV(self.ins | other.ins)

    def __and__(self, other):
        return NativePackedBHV(self.ins & other.ins)

    def __xor__(self, other):
        return NativePackedBHV(self.ins ^ other.ins)

    def __invert__(self):
        return NativePackedBHV(~self.ins)

    def select(self, when1, when0):
        return NativePackedBHV(self.ins.select(when1.ins, when0.ins))

    @classmethod
    def majority(cls, xs):
        return NativePackedBHV(CNativePackedBHV.majority([x.ins for x in xs]))

    def active(self):
        return self.ins.active()

    def hamming(self, other):
        return self.ins.hamming(other.ins)

    def permute(self, permutation_id: int):
        return NativePackedBHV(self.ins.permute(permutation_id))

    def rehash(self):
        return NativePackedBHV(self.ins.rehash())

    def swap_halves(self):
        return NativePackedBHV(self.ins.swap_halves())

    def __eq__(self, other):
        return self.ins == other.ins

    @classmethod
    def from_bytes(cls, bs):
        return NativePackedBHV(CNativePackedBHV.from_bytes(bs))

    def to_bytes(self):
        return self.ins.to_bytes()

NativePackedBHV.ZERO = NativePackedBHV(CNativePackedBHV.ZERO)
NativePackedBHV.ONE = NativePackedBHV(CNativePackedBHV.ONE)
NativePackedBHV.HALF = NativePackedBHV(CNativePackedBHV.HALF)
NativePackedBHV._FEISTAL_SUBKEYS = NativePackedBHV.nrand2(NativePackedBHV._FEISTAL_ROUNDS, 4)
