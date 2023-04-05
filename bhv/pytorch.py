from .abstract import *
import torch


# GPT-4 wrote this code, do not use in production
def pack_bool_to_long(bool_tensor):
    assert bool_tensor.dtype == torch.bool, "Input tensor must be of dtype torch.bool"
    assert bool_tensor.numel() % 64 == 0, "Input tensor must have a number of elements divisible by 64"

    bool_tensor = bool_tensor.view(-1, 64)
    packed_tensor = torch.zeros(bool_tensor.shape[0], dtype=torch.int64)

    for i in range(64):
        packed_tensor |= (bool_tensor[:, i].to(torch.int64) << i)

    return packed_tensor


# GPT-4 wrote this code, do not use in production
def unpack_long_to_bool(packed_tensor):
    assert packed_tensor.dtype == torch.int64, "Input tensor must be of dtype torch.int64"

    bool_tensor = torch.zeros(packed_tensor.numel() * 64, dtype=torch.bool)

    for i in range(64):
        bool_tensor[i::64] = ((packed_tensor >> i) & 1).bool()

    return bool_tensor


class TorchBoolPermutation(MemoizedPermutation):
    _permutations: 'dict[int | tuple[int, ...], Self]' = {}

    def __init__(self, tensor: torch.IntTensor):
        self.data: torch.BoolTensor = tensor

    @classmethod
    def random(cls) -> 'TorchBoolPermutation':
        return TorchBoolPermutation(torch.randperm(DIMENSION))

    def __mul__(self, other: 'TorchBoolPermutation') -> 'TorchBoolPermutation':
        return TorchBoolPermutation(self.data[other.data])

    def __invert__(self) -> 'TorchBoolPermutation':
        inv_permutation = torch.empty_like(self.data)
        inv_permutation[self.data] = torch.arange(DIMENSION)
        return TorchBoolPermutation(inv_permutation)

    def __call__(self, hv: 'TorchBoolBHV') -> 'TorchBoolBHV':
        return hv.permute_bits(self)


class TorchBoolBHV(AbstractBHV):
    def __init__(self, tensor: torch.BoolTensor):
        self.data: torch.BoolTensor = tensor

    @classmethod
    def rand(cls) -> 'TorchBoolBHV':
        return TorchBoolBHV(torch.empty(DIMENSION, dtype=torch.bool).random_())

    @classmethod
    def random(cls, active=0.5) -> 'TorchBoolBHV':
        return TorchBoolBHV(torch.empty(DIMENSION, dtype=torch.bool).bernoulli_(active))

    def select(self, when1: 'TorchBoolBHV', when0: 'TorchBoolBHV') -> 'TorchBoolBHV':
        return TorchBoolBHV(torch.where(self.data, when1.data, when0.data))

    @classmethod
    def majority(cls, vs: list['TorchBoolBHV']) -> 'TorchBoolBHV':
        data = [v.data for v in vs]
        extra = [cls.rand().data] if len(vs) % 2 == 0 else []

        tensor = torch.stack(data + extra)
        counts = tensor.sum(dim=-2, dtype=torch.int8)

        threshold = (len(vs) + len(extra)) // 2

        return TorchBoolBHV(torch.greater(counts,  threshold).to(torch.bool))

    def roll_bits(self, n: int) -> 'TorchBoolBHV':
        return TorchBoolBHV(torch.roll(self.data, n))

    def permute_bits(self, permutation: TorchBoolPermutation) -> 'TorchBoolBHV':
        return TorchBoolBHV(self.data[permutation.data])

    def permute(self, permutation_id: 'int | tuple[int, ...]') -> 'TorchBoolBHV':
        return self.permute_bits(TorchBoolPermutation.get(permutation_id))

    def __eq__(self, other: 'TorchBoolBHV') -> bool:
        return torch.equal(self.data, other.data)

    def __xor__(self, other: 'TorchBoolBHV') -> 'TorchBoolBHV':
        return TorchBoolBHV(torch.bitwise_xor(self.data, other.data))

    def __and__(self, other: 'TorchBoolBHV') -> 'TorchBoolBHV':
        return TorchBoolBHV(torch.bitwise_and(self.data, other.data))

    def __or__(self, other: 'TorchBoolBHV') -> 'TorchBoolBHV':
        return TorchBoolBHV(torch.bitwise_or(self.data, other.data))

    def __invert__(self) -> 'TorchBoolBHV':
        return TorchBoolBHV(torch.bitwise_not(self.data))

    def active(self) -> int:
        return torch.sum(self.data).item()

    def pack(self) -> 'TorchPackedBHV':
        return TorchPackedBHV(pack_bool_to_long(self.data))

TorchBoolBHV.ZERO = TorchBoolBHV(torch.zeros(DIMENSION, dtype=torch.bool))
TorchBoolBHV.ONE = TorchBoolBHV(torch.ones(DIMENSION, dtype=torch.bool))


class TorchWordPermutation(MemoizedPermutation):
    _permutations: 'dict[int | tuple[int, ...], Self]' = {}

    def __init__(self, tensor: torch.IntTensor):
        self.data: torch.BoolTensor = tensor

    @classmethod
    def random(cls) -> 'TorchWordPermutation':
        return TorchWordPermutation(torch.randperm(DIMENSION//64))

    def __mul__(self, other: 'TorchWordPermutation') -> 'TorchWordPermutation':
        return TorchWordPermutation(self.data[other.data])

    def __invert__(self) -> 'TorchWordPermutation':
        inv_permutation = torch.empty_like(self.data)
        inv_permutation[self.data] = torch.arange(DIMENSION//64)
        return TorchWordPermutation(inv_permutation)

    def __call__(self, hv: 'TorchPackedBHV') -> 'TorchPackedBHV':
        return hv.permute_words(self)


class TorchPackedBHV(AbstractBHV):
    def __init__(self, tensor: torch.LongTensor):
        assert DIMENSION % 64 == 0
        self.data: torch.LongTensor = tensor

    @classmethod
    def rand(cls) -> Self:
        return TorchPackedBHV(torch.randint(-9223372036854775808, 9223372036854775807, size=(DIMENSION//64,), dtype=torch.long))

    @classmethod
    def random(cls, active=0.5) -> Self:
        return TorchBoolBHV.random(active).pack()

    def roll_words(self, n: int) -> 'TorchPackedBHV':
        return TorchPackedBHV(torch.roll(self.data, n))

    def permute_words(self, permutation: TorchWordPermutation) -> 'TorchPackedBHV':
        return TorchPackedBHV(self.data[permutation.data])

    def permute(self, permutation_id: 'int | tuple[int, ...]') -> 'TorchPackedBHV':
        return self.permute_words(TorchWordPermutation.get(permutation_id))

    def __eq__(self, other: 'TorchBoolBHV') -> bool:
        return torch.equal(self.data, other.data)

    def __xor__(self, other: 'TorchPackedBHV') -> 'TorchPackedBHV':
        return TorchPackedBHV(torch.bitwise_xor(self.data, other.data))

    def __and__(self, other: 'TorchPackedBHV') -> 'TorchPackedBHV':
        return TorchPackedBHV(torch.bitwise_and(self.data, other.data))

    def __or__(self, other: 'TorchPackedBHV') -> 'TorchPackedBHV':
        return TorchPackedBHV(torch.bitwise_or(self.data, other.data))

    def __invert__(self) -> 'TorchPackedBHV':
        return TorchPackedBHV(torch.bitwise_not(self.data))

    def active(self) -> int:
        # currently no efficient implementation available for this https://github.com/pytorch/pytorch/issues/36380
        x = self.data
        x = x - ((x >> 1) & 0x5555555555555555)
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
        x = (x * 0x0101010101010101) >> 56
        return torch.sum(x).item()

    def unpack(self) -> 'TorchBoolBHV':
        return TorchBoolBHV(unpack_long_to_bool(self.data))

TorchPackedBHV.ZERO = TorchPackedBHV(torch.zeros(DIMENSION//64, dtype=torch.long))
TorchPackedBHV.ONE = TorchPackedBHV(torch.full((DIMENSION//64,), fill_value=-1, dtype=torch.long))  # -1 is all ones in torch's encoding
