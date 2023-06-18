from .abstract import *
import numpy as np
from sys import byteorder, version_info


class NumPyBoolPermutation(MemoizedPermutation):
    _permutations: 'dict[int | tuple[int, ...], Self]' = {}
    rng = np.random.default_rng()

    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def random(cls) -> 'NumPyBoolPermutation':
        return NumPyBoolPermutation(cls.rng.permutation(DIMENSION))

    def __mul__(self, other: 'NumPyBoolPermutation') -> 'NumPyBoolPermutation':
        return NumPyBoolPermutation(self.data[other.data])

    def __invert__(self) -> 'NumPyBoolPermutation':
        inv_permutation = np.empty_like(self.data)
        inv_permutation[self.data] = np.arange(DIMENSION)
        return NumPyBoolPermutation(inv_permutation)

    def __call__(self, hv: 'NumPyBoolBHV') -> 'NumPyBoolBHV':
        return hv.permute_bits(self)


class NumPyBoolBHV(AbstractBHV):
    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def rand(cls) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.random.randint(0, high=2, size=DIMENSION, dtype=np.bool_))

    @classmethod
    def random(cls, active: float) -> 'NumPyBoolBHV':
        assert 0. <= active <= 1.
        return NumPyBoolBHV(np.random.binomial(1, active, DIMENSION))

    def select(self, when1: 'NumPyBoolBHV', when0: 'NumPyBoolBHV') -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.where(self.data, when1.data, when0.data))

    def swap_halves(self) -> 'NumPyBoolBHV':
        return self.roll_bits(DIMENSION//2)

    def rehash(self) -> 'NumPyBoolBHV':
        return self.pack8().rehash().unpack()

    def roll_bits(self, n: int) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.roll(self.data, n))

    def permute_bits(self, permutation: 'NumPyBoolPermutation') -> 'NumPyBoolBHV':
        return NumPyBoolBHV(self.data[permutation.data])

    def permute(self, permutation_id: 'int | tuple[int, ...]') -> 'NumPyBoolBHV':
        return self.permute_bits(NumPyBoolPermutation.get(permutation_id))

    @classmethod
    def majority(cls, vs: list['NumPyBoolBHV']) -> 'NumPyBoolBHV':
        data = [v.data for v in vs]
        extra = [cls.rand().data] if len(vs) % 2 == 0 else []

        tensor = np.stack(data + extra)
        counts = tensor.sum(axis=-2, dtype=np.uint8 if len(vs) < 256 else np.uint32)

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

    def to_bytes(self):
        return self.pack8().to_bytes()

    @classmethod
    def from_bytes(cls, bs):
        return NumPyPacked8BHV.from_bytes(bs).unpack()

    def bits(self):
        return iter(self.data.astype(np.uint8))

NumPyBoolBHV.ZERO = NumPyBoolBHV(np.zeros(DIMENSION, dtype=np.bool_))
NumPyBoolBHV.ONE = NumPyBoolBHV(np.ones(DIMENSION, dtype=np.bool_))
NumPyBoolBHV._FEISTAL_SUBKEYS = NumPyBoolBHV.nrand2(NumPyBoolBHV._FEISTAL_ROUNDS, 4)
_halfb = np.zeros(DIMENSION, dtype=np.bool_)
_halfb[:DIMENSION//2] = np.bool_(True)
NumPyBoolBHV.HALF = NumPyBoolBHV(_halfb)


class NumPyBytePermutation(MemoizedPermutation):
    _permutations: 'dict[int | tuple[int, ...], Self]' = {}
    rng = np.random.default_rng()

    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def random(cls) -> 'NumPyBytePermutation':
        return NumPyBytePermutation(cls.rng.permutation(DIMENSION//8))

    def __mul__(self, other: 'NumPyBytePermutation') -> 'NumPyBytePermutation':
        return NumPyBytePermutation(self.data[other.data])

    def __invert__(self) -> 'NumPyBytePermutation':
        inv_permutation = np.empty_like(self.data)
        inv_permutation[self.data] = np.arange(DIMENSION//8)
        return NumPyBytePermutation(inv_permutation)

    def __call__(self, hv: 'NumPyPacked8BHV') -> 'NumPyPacked8BHV':
        return hv.permute_bytes(self)


class NumPyPacked8BHV(AbstractBHV):
    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def rand(cls) -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(np.random.randint(0, 255, DIMENSION//8, dtype=np.uint8))

    @classmethod
    def random(cls, active: float) -> 'NumPyPacked8BHV':
        assert 0. <= active <= 1.
        return NumPyBoolBHV.random(active).pack8()

    def roll_bytes(self, n: int) -> 'NumPyPacked8BHV':
        assert abs(n) < DIMENSION//8, "only supports DIMENSION/8 rolls"
        return NumPyPacked8BHV(np.roll(self.data, n))

    def swap_halves(self) -> 'NumPyPacked8BHV':
        return self.roll_bytes(DIMENSION//16)

    def permute_bytes(self, permutation: 'NumPyBytePermutation') -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(self.data[permutation.data])

    def permute(self, permutation_id: 'int | tuple[int, ...]') -> 'NumPyPacked8BHV':
        return self.permute_bytes(NumPyBytePermutation.get(permutation_id))

    def rehash(self) -> 'NumPyPacked8BHV':
        byte_data = self.to_bytes()
        rehashed_byte_data = hashlib.shake_256(byte_data).digest(DIMENSION//8)
        return NumPyPacked8BHV.from_bytes(rehashed_byte_data)

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

    if version_info[2] >= 10:
        def active(self) -> int:
            return int.from_bytes(self.data.tobytes(), byteorder).bit_count()
    else:
        lookup = np.array([bin(i).count("1") for i in range(256)])

        def active(self) -> int:
            return self.lookup[self.data].sum()

    def unpack(self) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.unpackbits(self.data))

    def repack64(self) -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(self.data.view(dtype=np.uint64))

    def to_bytes(self):
        return self.data.tobytes()

    @classmethod
    def from_bytes(cls, bs):
        return cls(np.frombuffer(bs, dtype=np.uint8))


NumPyPacked8BHV.ZERO = NumPyPacked8BHV(np.zeros(DIMENSION//8, dtype=np.uint8))
NumPyPacked8BHV.ONE = NumPyPacked8BHV(np.full(DIMENSION//8, fill_value=255, dtype=np.uint8))
NumPyPacked8BHV._FEISTAL_SUBKEYS = NumPyPacked8BHV.nrand2(NumPyPacked8BHV._FEISTAL_ROUNDS, 4)
_half8 = np.zeros(DIMENSION//8, dtype=np.uint8)
_half8[:DIMENSION//16] = np.uint8(255)
NumPyPacked8BHV.HALF = NumPyPacked8BHV(_half8)


class NumPyWordPermutation(MemoizedPermutation):
    _permutations: 'dict[int | tuple[int, ...], Self]' = {}
    rng = np.random.default_rng()

    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def random(cls) -> 'NumPyWordPermutation':
        return NumPyWordPermutation(cls.rng.permutation(DIMENSION//64))

    def __mul__(self, other: 'NumPyWordPermutation') -> 'NumPyWordPermutation':
        return NumPyWordPermutation(self.data[other.data])

    def __invert__(self) -> 'NumPyWordPermutation':
        inv_permutation = np.empty_like(self.data)
        inv_permutation[self.data] = np.arange(DIMENSION//64)
        return NumPyWordPermutation(inv_permutation)

    def __call__(self, hv: 'NumPyPacked64BHV') -> 'NumPyPacked64BHV':
        return hv.permute_words(self)


class NumPyPacked64BHV(AbstractBHV):
    rng = np.random.SFC64()

    def __init__(self, array: np.ndarray):
        self.data: np.ndarray = array

    @classmethod
    def rand(cls) -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(cls.rng.random_raw(DIMENSION//64))

    @classmethod
    def random(cls, active: float) -> 'NumPyPacked64BHV':
        assert 0. <= active <= 1.
        return NumPyBoolBHV.random(active).pack64()

    @classmethod
    def majority(cls, vs: list['NumPyPacked64BHV']) -> 'NumPyPacked64BHV':
        if len(vs) <= 9:
            return cls._majority_via_custom(vs)
        else:
            return cls._majority_via_unpacked(vs)

    @classmethod
    def _majority_via_unpacked(cls, vs: list['NumPyPacked64BHV']) -> 'NumPyPacked64BHV':
        return NumPyBoolBHV.majority([v.unpack() for v in vs]).pack64()

    def swap_halves(self) -> 'NumPyPacked64BHV':
        return self.roll_words(DIMENSION//128)

    def rehash(self) -> 'NumPyPacked64BHV':
        return self.repack8().rehash().repack64()

    def roll_words(self, n: int) -> 'NumPyPacked64BHV':
        assert abs(n) < DIMENSION//64, "only supports DIMENSION/64 rolls"
        return NumPyPacked64BHV(np.roll(self.data, n))

    def roll_word_bits(self, n: int) -> 'NumPyPacked64BHV':
        assert abs(n) < 64, "only supports 64 rolls"
        if n == 0:
            return NumPyPacked64BHV(self.data)
        elif n > 0:
            return np.bitwise_or(np.right_shift(self.data, n), np.left_shift(self.data, (64 - n)))
        else:
            return np.bitwise_or(np.left_shift(self.data, n), np.right_shift(self.data, (64 - n)))

    # roll_words and roll_word_bits could be combined for more options allowing positive and negative combinations
    # ((1 2 3 4) (a b c d) (α β γ δ))
    # rolled by 1, -2 for example results in
    # ((γ δ α β) (3 4 1 2) (c d a b))

    def permute_words(self, permutation: 'NumPyWordPermutation') -> 'NumPyPacked64BHV':
        return NumPyPacked64BHV(self.data[permutation.data])

    def permute(self, permutation_id: 'int | tuple[int, ...]') -> 'NumPyPacked64BHV':
        return self.permute_words(NumPyWordPermutation.get(permutation_id))

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

    if version_info[2] >= 10:
        def active(self) -> int:
            return int.from_bytes(self.data.tobytes(), byteorder).bit_count()
    else:
        def active(self) -> int:
            return self.repack8().active()

    def unpack(self) -> 'NumPyBoolBHV':
        return NumPyBoolBHV(np.unpackbits(self.data.view(np.uint8)))

    def repack8(self) -> 'NumPyPacked8BHV':
        return NumPyPacked8BHV(self.data.view(dtype=np.uint8))

    def to_bytes(self):
        return self.repack8().to_bytes()

    @classmethod
    def from_bytes(cls, bs):
        return NumPyPacked8BHV.from_bytes(bs).repack64()

NumPyPacked64BHV.ZERO = NumPyPacked64BHV(np.zeros(DIMENSION//64, dtype=np.uint64))
NumPyPacked64BHV.ONE = NumPyPacked64BHV(np.full(DIMENSION//64, fill_value=-1, dtype=np.uint64))
NumPyPacked64BHV._FEISTAL_SUBKEYS = NumPyPacked64BHV.nrand2(NumPyPacked64BHV._FEISTAL_ROUNDS, 4)
_half64 = np.zeros(DIMENSION//64, dtype=np.uint64)
_half64[:DIMENSION//128] = np.uint64(-1)
NumPyPacked64BHV.HALF = NumPyPacked64BHV(_half64)
