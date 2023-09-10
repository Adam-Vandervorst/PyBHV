from random import random
from bhv.abstract import AbstractBHV
from typing import Generic, TypeVar, Type, Optional, Iterable

T = TypeVar('T')


class Embedding(Generic[T]):
    def forward(self, x: T) -> AbstractBHV:
        raise NotImplementedError()

    def back(self, x: AbstractBHV) -> Optional[T]:
        raise NotImplementedError()


class Random(Embedding[T]):
    def __init__(self, hvt: Type[AbstractBHV]):
        self.hvt = hvt
        self.hvs = {}

    def forward(self, x: T) -> AbstractBHV:
        if x in self.hvs:
            return self.hvs[x]
        else:
            hv = self.hvt.rand()
            self.hvs[x] = hv
            return hv

    def back(self, input_hv: AbstractBHV, threshold=.1) -> Optional[T]:
        best_x = None
        best_distance = 1.
        for x, hv in self.hvs.items():
            distance = input_hv.bit_error_rate(hv)
            if distance < best_distance:
                best_distance = distance
                best_x = x

        if best_distance < threshold:
            return best_x


class InterpolateBetween(Embedding[float]):
    def __init__(self, hvt: Type[AbstractBHV], begin: AbstractBHV = None, end: AbstractBHV = None):
        self.hvt = hvt
        self.begin = hvt.rand() if begin is None else begin
        self.end = hvt.rand() if end is None else end

    def forward(self, x: float) -> AbstractBHV:
        return self.end.select_random(self.begin, x)

    def back(self, input_hv: AbstractBHV, threshold=.1) -> Optional[float]:
        beginh = self.begin.bit_error_rate(input_hv)
        endh = self.end.bit_error_rate(input_hv)
        totalh = endh + beginh
        if abs(totalh - .5) < threshold:
            return beginh/totalh


class Collapse(Embedding[Iterable[float]]):
    def __init__(self, hvt: Type[AbstractBHV]):
        self.hvt = hvt

    def forward(self, x: Iterable[float]) -> AbstractBHV:
        return self.hvt.from_bitstream(random() < v for v in x)

    def back(self, input_hv: AbstractBHV, soft=.1) -> Optional[Iterable[float]]:
        i = 1. - soft
        o = soft
        return (i if b else o for b in input_hv.bits())
