from random import random, uniform
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


class Intervals:
    @staticmethod
    def _perfect_overlap(low: float, high: float, divisions: int, overlap: int, ends: bool):
        """
        Generate a list of intervals dividing a given range [low, high] into 'divisions' with a specified 'overlap'.

        Example:
        >>> Intervals._perfect_overlap(0.0, 1.0, divisions=5, overlap=2, ends=True)
        [(0.0, 0.1), (0.0, 0.2), (0.1, 0.3), (0.2, 0.4), (0.3, 0.5), (0.4, 0.6),
         (0.5, 0.7), (0.6, 0.8), (0.7, 0.9), (0.8, 1.0), (0.9, 1.0)]

        >>> Intervals._perfect_overlap(0.0, 1.0, divisions=5, overlap=2, ends=False)
        [(0.0, 0.2), (0.1, 0.3), (0.2, 0.4), (0.3, 0.5), (0.4, 0.6),
         (0.5, 0.7), (0.6, 0.8), (0.7, 0.9), (0.8, 1.0)]
        """
        step = (high - low) / divisions
        offset = step / overlap
        intervals = []

        if ends:
            intervals.append((low, low + offset))

        current_low = low
        while current_low + step <= high:
            current_high = current_low + step
            intervals.append((current_low, current_high))
            current_low += offset

        if ends and current_low < high:
            intervals.append((current_low, high))

        return intervals

    @classmethod
    def perfect(cls, hvt, low: float = 0., high: float = 1., divisions: int = 5, overlap: int = 2, ends: bool = False):
        return cls(hvt, cls._perfect_overlap(low, high, divisions, overlap, ends))

    def __init__(self, hvt, intervals):
        self.hvt = hvt
        self.intervals = intervals
        ls, hs = zip(*intervals)
        self.span = (min(ls), max(hs))
        self.hvs = [hvt.rand() for _ in intervals]

    def forward(self, x: float):
        matching = [hv for hv, (l, h) in zip(self.hvs, self.intervals) if l <= x <= h]
        return self.hvt.majority(matching)

    def back(self, input_hv, threshold=4):
        L, H = self.span
        for hv, (l, h) in zip(self.hvs, self.intervals):
            if not hv.unrelated(input_hv, threshold):
                L = max(L, l)
                H = min(H, h)
                if L > H:
                    return
        return uniform(L, H)
