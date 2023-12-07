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

    def ternary(self, y, z, op):
        return NativePackedBHV(self.ins.ternary(y.ins, z.ins, op))

    @classmethod
    def majority(cls, xs):
        return NativePackedBHV(CNativePackedBHV.majority([x.ins for x in xs]))

    @classmethod
    def parity(cls, xs):
        return NativePackedBHV(CNativePackedBHV.parity([x.ins for x in xs]))

    @classmethod
    def threshold(cls, xs, t):
        return NativePackedBHV(CNativePackedBHV.threshold([x.ins for x in xs], t))

    @classmethod
    def weighted_threshold(cls, xs, ws, t):
        return NativePackedBHV(CNativePackedBHV.weighted_threshold([x.ins for x in xs], list(map(float, ws)), float(t)))

    @classmethod
    def representative(cls, xs):
        return NativePackedBHV(CNativePackedBHV.representative([x.ins for x in xs]))

    def active(self):
        return self.ins.active()

    def hamming(self, other):
        return self.ins.hamming(other.ins)

    def closest(self, xs):
        return self.ins.closest([x.ins for x in xs])

    def top(self, xs, k):
        return self.ins.top([x.ins for x in xs], k)

    def within(self, xs, d):
        return self.ins.within([x.ins for x in xs], d)

    def permute_words(self, permutation_id: int):
        return NativePackedBHV(self.ins.permute_words(permutation_id))

    def permute_byte_bits(self, permutation_id: int):
        return NativePackedBHV(self.ins.permute_byte_bits(permutation_id))

    def roll_words(self, d: int):
        return NativePackedBHV(self.ins.roll_words(d))

    def roll_word_bits(self, d: int):
        return NativePackedBHV(self.ins.roll_word_bits(d))

    def roll_bits(self, d: int):
        return NativePackedBHV(self.ins.roll_bits(d))

    def _permute_composite(self, permutation_id: 'tuple'):
        v = self
        for e in permutation_id:
            v = v.permute(e)
        return v

    def permute(self, permutation_id: 'int | tuple'):
        if isinstance(permutation_id, tuple):
            return self._permute_composite(permutation_id)
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

    def __getstate__(self):
        return self.to_bytes()

    def __setstate__(self, state):
        self.ins = CNativePackedBHV.from_bytes(state)

NativePackedBHV.ZERO = NativePackedBHV(CNativePackedBHV.ZERO)
NativePackedBHV.ONE = NativePackedBHV(CNativePackedBHV.ONE)
NativePackedBHV.HALF = NativePackedBHV(CNativePackedBHV.HALF)
NativePackedBHV._FEISTAL_SUBKEYS = NativePackedBHV.nrand2(NativePackedBHV._FEISTAL_ROUNDS, 4)
