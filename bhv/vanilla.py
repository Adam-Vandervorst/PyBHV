from .abstract import *
import random
from sys import byteorder, version_info


class VanillaPermutation(MemoizedPermutation):
    _permutations: 'dict[int | tuple[int, ...], Self]' = {}

    def __init__(self, indices):
        self.data = indices

    @classmethod
    def random(cls) -> 'VanillaPermutation':
        p = list(range(DIMENSION//8))
        random.shuffle(p)
        return VanillaPermutation(p)

    def __mul__(self, other: 'VanillaPermutation') -> 'VanillaPermutation':
        return VanillaPermutation([self.data[other.data[i]] for i in range(DIMENSION//8)])

    def __invert__(self) -> 'VanillaPermutation':
        p = [None]*(DIMENSION//8)
        for i, j in enumerate(self.data):
            p[j] = i
        return VanillaPermutation(p)

    def __call__(self, hv: 'VanillaBHV') -> 'VanillaBHV':
        return hv.permute_bytes(self)


class VanillaBHV(AbstractBHV):
    def __init__(self, data: bytes):
        self.data: bytes = data

    @classmethod
    def rand(cls) -> 'VanillaBHV':
        return VanillaBHV(random.randbytes(DIMENSION//8))

    @classmethod
    def random(cls, active: float) -> 'VanillaBHV':
        assert 0. <= active <= 1.
        return VanillaBHV(bytes([
            (random.random() < active) | (random.random() < active) << 1 | (random.random() < active) << 2 | (random.random() < active) << 3 |
            (random.random() < active) << 4 | (random.random() < active) << 5 | (random.random() < active) << 6 | (random.random() < active) << 7
            for _ in range(DIMENSION//8)]))

    def roll_bytes(self, n: int) -> 'VanillaBHV':
        assert abs(n) < DIMENSION//8, "only supports DIMENSION/8 rolls"
        return VanillaBHV(self.data[n:] + self.data[:n])

    def swap_halves(self) -> 'VanillaBHV':
        return self.roll_bytes(DIMENSION//16)

    def permute_bytes(self, permutation: 'VanillaPermutation') -> 'VanillaBHV':
        return VanillaBHV(bytes([self.data[i] for i in permutation.data]))

    def permute(self, permutation_id: 'int | tuple[int, ...]') -> 'VanillaBHV':
        return self.permute_bytes(VanillaPermutation.get(permutation_id))

    def rehash(self) -> 'VanillaBHV':
        return VanillaBHV(hashlib.shake_256(self.data).digest(DIMENSION//8))

    def __eq__(self, other: 'VanillaBHV') -> bool:
        return self.data == other.data

    def __xor__(self, other: 'VanillaBHV') -> 'VanillaBHV':
        return self.from_int(self.to_int() ^ other.to_int())

    def __and__(self, other: 'VanillaBHV') -> 'VanillaBHV':
        return self.from_int(self.to_int() & other.to_int())

    def __or__(self, other: 'VanillaBHV') -> 'VanillaBHV':
        return self.from_int(self.to_int() | other.to_int())

    if version_info[2] >= 10:
        def active(self) -> int:
            return self.to_int().bit_count()
    else:
        def active(self) -> int:
            return bin(self.to_int()).count("1")

    def to_int(self) -> int:
        return int.from_bytes(self.data, byteorder, signed=False)

    @classmethod
    def from_int(cls, i: int):
        return VanillaBHV(i.to_bytes(DIMENSION//8, byteorder, signed=False))

    def to_bytes(self):
        return self.data

    @classmethod
    def from_bytes(cls, bs):
        return cls(bs)

VanillaBHV.ZERO = VanillaBHV(bytes([0 for _ in range(DIMENSION//8)]))
VanillaBHV.ONE = VanillaBHV(bytes([0xff for _ in range(DIMENSION//8)]))
VanillaBHV._FEISTAL_SUBKEYS = VanillaBHV.nrand2(VanillaBHV._FEISTAL_ROUNDS, 4)
VanillaBHV.HALF = VanillaBHV(bytes([0 for _ in range(DIMENSION//16)] + [0xff for _ in range(DIMENSION//16)]))
