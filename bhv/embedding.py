from bhv.abstract import AbstractBHV
from typing import Generic, TypeVar, Type

T = TypeVar('T')


class Translation(Generic[T]):
    def forward(self, x: T) -> AbstractBHV:
        raise NotImplementedError()

    def back(self, x: AbstractBHV) -> T:
        raise NotImplementedError()


class Random(Translation):
    def __init__(self, hvt: Type[AbstractBHV]):
        self.hv = hvt
        self.hvs = {}

    def forward(self, x: T) -> AbstractBHV:
        if x in self.hvs:
            return self.hvs[x]
        else:
            hv = self.hv.rand()
            self.hvs[x] = hv
            return hv

    def back(self, input_hv: AbstractBHV, threshold=.1) -> T:
        best_x = None
        best_score = 1.
        for x, hv in self.hvs.items():
            score = input_hv.bit_error_rate(hv)
            if score < best_score and score < threshold:
                best_score = score
                best_x = x
        return best_x

