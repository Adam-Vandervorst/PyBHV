from .abstract import *
import numpy as np


class NumPyBoolHV(AbstractHV):
    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def rand(cls) -> 'NumPyBoolHV':
        return NumPyBoolHV(np.random.randint(0, high=2, size=DIMENSION, dtype=np.bool_))

    @classmethod
    def random(cls, active=0.5) -> 'NumPyBoolHV':
        return NumPyBoolHV(np.random.binomial(1, active, DIMENSION))

    def select(self, when1: 'NumPyBoolHV', when0: 'NumPyBoolHV') -> 'NumPyBoolHV':
        return NumPyBoolHV(np.where(self.data, when1.data, when0.data))

    def __xor__(self, other: 'NumPyBoolHV') -> 'NumPyBoolHV':
        return NumPyBoolHV(np.bitwise_xor(self.data, other.data))

    def __and__(self, other: 'NumPyBoolHV') -> 'NumPyBoolHV':
        return NumPyBoolHV(np.bitwise_and(self.data, other.data))

    def __or__(self, other: 'NumPyBoolHV') -> 'NumPyBoolHV':
        return NumPyBoolHV(np.bitwise_or(self.data, other.data))

    def __invert__(self, other: 'NumPyBoolHV') -> 'NumPyBoolHV':
        return NumPyBoolHV(np.bitwise_not(self.data, other.data))

    def active(self) -> int:
        return int(np.sum(self.data))

    def pack8(self) -> 'NumPyPacked8HV':
        return NumPyPacked8HV(np.packbits(self.data))

    def pack64(self) -> 'NumPyPacked64HV':
        return NumPyPacked64HV(np.packbits(self.data).view(dtype=np.uint64))

NumPyBoolHV.ZERO = NumPyBoolHV(np.zeros(DIMENSION, dtype=np.bool_))
NumPyBoolHV.ONE = NumPyBoolHV(np.ones(DIMENSION, dtype=np.bool_))


class NumPyPacked8HV(AbstractHV):
    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def rand(cls) -> 'NumPyPacked8HV':
        return NumPyPacked8HV(np.random.randint(0, 255, DIMENSION//8, dtype=np.uint8))

    @classmethod
    def random(cls, active=0.5) -> 'NumPyPacked8HV':
        return NumPyBoolHV.random(active).pack8()

    def __xor__(self, other: 'NumPyPacked8HV') -> 'NumPyPacked8HV':
        return NumPyPacked8HV(np.bitwise_xor(self.data, other.data))

    def __and__(self, other: 'NumPyPacked8HV') -> 'NumPyPacked8HV':
        return NumPyPacked8HV(np.bitwise_and(self.data, other.data))

    def __or__(self, other: 'NumPyPacked8HV') -> 'NumPyPacked8HV':
        return NumPyPacked8HV(np.bitwise_or(self.data, other.data))

    def __invert__(self) -> 'NumPyPacked8HV':
        return NumPyPacked8HV(np.bitwise_not(self.data))

    def active(self) -> int:
        return sum(x.bit_count() for x in self.data)

    def unpack(self) -> 'NumPyBoolHV':
        return NumPyBoolHV(np.unpackbits(self.data))

NumPyPacked8HV.ZERO = NumPyPacked8HV(np.zeros(DIMENSION//8, dtype=np.uint8))
NumPyPacked8HV.ONE = NumPyPacked8HV(np.full(DIMENSION//8, fill_value=255, dtype=np.uint8))


RAND = np.random.SFC64()
class NumPyPacked64HV(AbstractHV):
    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def rand(cls) -> 'NumPyPacked64HV':
        return NumPyPacked64HV(RAND.random_raw(DIMENSION//64))

    @classmethod
    def random(cls, active=0.5) -> 'NumPyPacked64HV':
        return NumPyBoolHV.random(active).pack64()

    def __xor__(self, other: 'NumPyPacked64HV') -> 'NumPyPacked64HV':
        return NumPyPacked64HV(np.bitwise_xor(self.data, other.data))

    def __and__(self, other: 'NumPyPacked64HV') -> 'NumPyPacked64HV':
        return NumPyPacked64HV(np.bitwise_and(self.data, other.data))

    def __or__(self, other: 'NumPyPacked64HV') -> 'NumPyPacked64HV':
        return NumPyPacked64HV(np.bitwise_or(self.data, other.data))

    def __invert__(self) -> 'NumPyPacked64HV':
        return NumPyPacked64HV(np.bitwise_not(self.data))

    def active(self) -> int:
        return sum(x.bit_count() for x in self.data)

    def unpack(self) -> 'NumPyBoolHV':
        return NumPyBoolHV(np.unpackbits(self.data.view(np.uint8)))

NumPyPacked64HV.ZERO = NumPyPacked64HV(np.zeros(DIMENSION//64, dtype=np.uint64))
NumPyPacked64HV.ONE = NumPyPacked64HV(np.full(DIMENSION//64, fill_value=-1, dtype=np.uint64))
