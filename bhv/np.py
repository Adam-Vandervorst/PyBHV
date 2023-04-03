from .abstract import *
from typing import Tuple
import numpy as np


class NumPyBoolBHV(AbstractBHV):
    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def rand(cls) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.random.randint(0, high=2, size=DIMENSION, dtype=np.bool_))

    @classmethod
    def random(cls, active=0.5) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.random.binomial(1, active, DIMENSION))

    def select(self, when1: 'NumPyBoolBHV', when0: 'NumPyBoolBHV') -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.where(self.data, when1.data, when0.data))

    def roll_bits(self, n: int) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.roll(self.data, n))

    def permute(self, permutation_id: int) -> 'NumPyBoolBHV':
        assert abs(permutation_id) < DIMENSION, "only supports DIMENSION permutations with roll_bits implementation"
        return self.roll_bits(permutation_id)

    @classmethod
    def majority(cls, vs: list['NumPyBoolBHV']) -> 'NumPyBoolBHV':
        data = [v.data for v in vs]
        extra = [cls.rand().data] if len(vs) % 2 == 0 else []

        tensor = np.stack(data + extra)
        counts = tensor.sum(axis=-2, dtype=np.uint8)

        threshold = (len(vs) + len(extra))//2

        return NumPyBoolBHV(np.greater(counts, threshold))

    def __eq__(self, other: 'NumPyBoolBHV') -> bool:
        return np.array_equal(self.data, other.data)

    def __xor__(self, other: 'NumPyBoolBHV') -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.bitwise_xor(self.data, other.data))

    def __and__(self, other: 'NumPyBoolBHV') -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.bitwise_and(self.data, other.data))

    def __or__(self, other: 'NumPyBoolBHV') -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.bitwise_or(self.data, other.data))

    def __invert__(self) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.bitwise_not(self.data))

    def active(self) -> int:
        return int(np.sum(self.data))

    def pack8(self) -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(np.packbits(self.data))

    def pack64(self) -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(np.packbits(self.data).view(dtype=np.uint64))

NumPyBoolBHV.ZERO = NumPyBoolBHV(np.zeros(DIMENSION, dtype=np.bool_))
NumPyBoolBHV.ONE = NumPyBoolBHV(np.ones(DIMENSION, dtype=np.bool_))


class NumPyPacked8BHV(AbstractBHV):
    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def rand(cls) -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(np.random.randint(0, 255, DIMENSION//8, dtype=np.uint8))

    @classmethod
    def random(cls, active=0.5) -> 'NumPyPacked8BHV':
        return NumPyBoolBHV.random(active).pack8()

    def roll_bytes(self, n: int) -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(np.roll(self.data, n))

    def permute(self, permutation_id: int) -> 'NumPyPacked8BHV':
        assert abs(permutation_id) < DIMENSION//8, "only supports DIMENSION/8 permutations with roll_bytes implementation"
        return self.roll_bytes(permutation_id)

    def __eq__(self, other: 'NumPyPacked8BHV') -> bool:
        return np.array_equal(self.data, other.data)

    def __xor__(self, other: 'NumPyPacked8BHV') -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(np.bitwise_xor(self.data, other.data))

    def __and__(self, other: 'NumPyPacked8BHV') -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(np.bitwise_and(self.data, other.data))

    def __or__(self, other: 'NumPyPacked8BHV') -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(np.bitwise_or(self.data, other.data))

    def __invert__(self) -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(np.bitwise_not(self.data))

    def active(self) -> int:
        return sum(x.bit_count() for x in self.data)

    def unpack(self) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.unpackbits(self.data))

NumPyPacked8BHV.ZERO = NumPyPacked8BHV(np.zeros(DIMENSION//8, dtype=np.uint8))
NumPyPacked8BHV.ONE = NumPyPacked8BHV(np.full(DIMENSION//8, fill_value=255, dtype=np.uint8))


RAND = np.random.SFC64()
class NumPyPacked64BHV(AbstractBHV):
    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def rand(cls) -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(RAND.random_raw(DIMENSION//64))

    @classmethod
    def random(cls, active=0.5) -> 'NumPyPacked64BHV':
        return NumPyBoolBHV.random(active).pack64()

    @classmethod
    def _majority_via_unpacked(cls, vs: list['NumPyPacked64BHV']) -> 'NumPyPacked64BHV':
        return NumPyBoolBHV.majority([v.unpack() for v in vs]).pack64()

    def roll_words(self, n: int) -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(np.roll(self.data, n))

    def permute(self, permutation_id: int) -> 'NumPyPacked64BHV':
        assert abs(permutation_id) < DIMENSION//64, "only supports DIMENSION/64 permutations with roll_words implementation"
        return self.roll_words(permutation_id)

    def __eq__(self, other: 'NumPyPacked64BHV') -> bool:
        return np.array_equal(self.data, other.data)

    def __xor__(self, other: 'NumPyPacked64BHV') -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(np.bitwise_xor(self.data, other.data))

    def __and__(self, other: 'NumPyPacked64BHV') -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(np.bitwise_and(self.data, other.data))

    def __or__(self, other: 'NumPyPacked64BHV') -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(np.bitwise_or(self.data, other.data))

    def __invert__(self) -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(np.bitwise_not(self.data))

    def active(self) -> int:
        return sum(x.bit_count() for x in self.data)

    def unpack(self) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.unpackbits(self.data.view(np.uint8)))

NumPyPacked64BHV.ZERO = NumPyPacked64BHV(np.zeros(DIMENSION//64, dtype=np.uint64))
NumPyPacked64BHV.ONE = NumPyPacked64BHV(np.full(DIMENSION//64, fill_value=-1, dtype=np.uint64))
