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


class TorchBoolHV(AbstractHV):
    def __init__(self, tensor: torch.BoolTensor):
        self.data: torch.BoolTensor = tensor

    @classmethod
    def rand(cls) -> 'TorchBoolHV':
        return TorchBoolHV(torch.empty(DIMENSION, dtype=torch.bool).random_())

    @classmethod
    def random(cls, active=0.5) -> 'TorchBoolHV':
        return TorchBoolHV(torch.empty(DIMENSION, dtype=torch.bool).bernoulli_(active))

    def select(self, when1: 'TorchBoolHV', when0: 'TorchBoolHV') -> 'TorchBoolHV':
        return TorchBoolHV(torch.where(self.data, when1.data, when0.data))

    def __xor__(self, other: 'TorchBoolHV') -> 'TorchBoolHV':
        return TorchBoolHV(torch.bitwise_xor(self.data, other.data))

    def __and__(self, other: 'TorchBoolHV') -> 'TorchBoolHV':
        return TorchBoolHV(torch.bitwise_and(self.data, other.data))

    def __or__(self, other: 'TorchBoolHV') -> 'TorchBoolHV':
        return TorchBoolHV(torch.bitwise_or(self.data, other.data))

    def __invert__(self) -> 'TorchBoolHV':
        return TorchBoolHV(torch.bitwise_not(self.data))

    def active(self) -> int:
        return torch.sum(self.data).item()

    def pack(self) -> 'TorchPackedHV':
        return TorchPackedHV(pack_bool_to_long(self.data))

TorchBoolHV.ZERO = TorchBoolHV(torch.zeros(DIMENSION, dtype=torch.bool))
TorchBoolHV.ONE = TorchBoolHV(torch.ones(DIMENSION, dtype=torch.bool))


class TorchPackedHV(AbstractHV):
    def __init__(self, tensor: torch.LongTensor):
        assert DIMENSION % 64 == 0
        self.data: torch.LongTensor = tensor

    @classmethod
    def rand(cls) -> Self:
        return TorchPackedHV(torch.randint(-9223372036854775808, 9223372036854775807, size=(DIMENSION//64,), dtype=torch.long))

    @classmethod
    def random(cls, active=0.5) -> Self:
        return TorchBoolHV.random(active).pack()

    def __xor__(self, other: 'TorchPackedHV') -> 'TorchPackedHV':
        return TorchPackedHV(torch.bitwise_xor(self.data, other.data))

    def __and__(self, other: 'TorchPackedHV') -> 'TorchPackedHV':
        return TorchPackedHV(torch.bitwise_and(self.data, other.data))

    def __or__(self, other: 'TorchPackedHV') -> 'TorchPackedHV':
        return TorchPackedHV(torch.bitwise_or(self.data, other.data))

    def __invert__(self) -> 'TorchPackedHV':
        return TorchPackedHV(torch.bitwise_not(self.data))

    def active(self) -> int:
        # currently no efficient implementation available for this https://github.com/pytorch/pytorch/issues/36380
        x = self.data
        x = x - ((x >> 1) & 0x5555555555555555)
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
        x = (x * 0x0101010101010101) >> 56
        return torch.sum(x).item()

    def unpack(self) -> 'TorchBoolHV':
        return TorchBoolHV(unpack_long_to_bool(self.data))

TorchPackedHV.ZERO = TorchPackedHV(torch.zeros(DIMENSION//64, dtype=torch.long))
TorchPackedHV.ONE = TorchPackedHV(torch.full((DIMENSION//64,), fill_value=-1, dtype=torch.long))  # -1 is all ones in torch's encoding
