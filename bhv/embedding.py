from random import random, uniform
from bhv.abstract import AbstractBHV
from typing import Generic, TypeVar, Type, Optional, Iterable, Callable

T = TypeVar('T')
S = TypeVar('S')


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

    def back(self, input_hv: AbstractBHV, threshold=6) -> Optional[T]:
        best_x = None
        best_distance = self.hvt.EXPECTED_RAND_APART
        for x, hv in self.hvs.items():
            distance = input_hv.std_apart(hv)
            if distance < best_distance:
                best_distance = distance
                best_x = x

        if best_distance < self.hvt.EXPECTED_RAND_APART - threshold:
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


class Intervals(Embedding[float]):
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

    def forward(self, x: float) -> AbstractBHV:
        matching = [hv for hv, (l, h) in zip(self.hvs, self.intervals) if l <= x <= h]
        return self.hvt.majority(matching)

    def back(self, input_hv: AbstractBHV, threshold=4) -> Optional[float]:
        L, H = self.span
        for hv, (l, h) in zip(self.hvs, self.intervals):
            if not hv.unrelated(input_hv, threshold):
                L = max(L, l)
                H = min(H, h)
                if L > H:
                    return
        return uniform(L, H)


class BeanMachine(Embedding[float]):
    def __init__(self, hvt, ncolumns, kernel, border: '"wrap" | "cut" | "bunch"' = "cut"):
        assert len(kernel) <= ncolumns
        self.hvt = hvt
        self.ncolumns = ncolumns
        self.kernel = kernel
        self.border = border
        self.hvs = [hvt.rand() for _ in range(ncolumns)]

    @staticmethod
    def _discrete_sample(data: list[S], point: float, kernel: list[int], border: '"wrap" | "cut" | "bunch"') -> list[S]:
        """
        Samples data points from a list based on a center point and a kernel.

        Parameters:
        - data: List of data points to sample from.
        - point: Center point as a float between 0 and 1.
        - kernel: List of integers representing the multipliers for each data point around the center.
        - border: How to handle edge cases. Can be "wrap", "cut", or "bunch".

        Returns:
        - List[X]: List of sampled data points.

        Examples (doctest):
        >>> BeanMachine._discrete_sample(list('abcde'), 0.5, [1, 2, 1])
        ['b', 'c', 'c', 'd']

        >>> BeanMachine._discrete_sample(list('abc'), 0.0, [1, 1, 1], border="wrap")
        ['c', 'a', 'b']

        >>> BeanMachine._discrete_sample(list('abc'), 0.0, [1, 1, 1], border="cut")
        ['a', 'b']

        >>> BeanMachine._discrete_sample(list('abc'), 0.0, [1, 1, 1], border="bunch")
        ['a', 'a', 'b']
        """
        n = len(data)
        center_idx = int(point * n)

        # Calculate half-length of the kernel and whether it's even or odd
        k_len = len(kernel)
        half_k_len = k_len // 2
        is_even_kernel = k_len % 2 == 0

        result = []
        for i in range(-half_k_len, half_k_len + 1):
            if is_even_kernel and i == half_k_len:
                break  # Skip the last element for even-length kernels

            idx = center_idx + i

            if idx < 0:
                if border == "wrap":
                    idx = n + idx
                elif border == "cut":
                    continue
                else:  # "bunch"
                    idx = 0
            elif idx >= n:
                if border == "wrap":
                    idx = idx - n
                elif border == "cut":
                    continue
                else:  # "bunch"
                    idx = n - 1

            # Add the data point as many times as specified by the kernel
            result.extend([data[idx]] * kernel[i + half_k_len])

        return result

    def forward(self, x: float) -> AbstractBHV:
        return self.hvt.majority(self._discrete_sample(self.hvs, x, self.kernel, self.border))

    @staticmethod
    def _position_infer(ds: list[float], kernel: list[int], border: '"wrap" | "cut" | "bunch"') -> int:
        """
        Infers the center point in the data list that most closely matches the given data point `x` based on a kernel.

        Parameters:
        - data: List of data points to sample from.
        - x: The data point to find in the data list.
        - kernel: List of integers representing the multipliers for each data point around the center.
        - distance_metric: A function to compute the distance between two data points. Defaults to the example `d`.
        - border: How to handle edge cases. Can be "wrap", "cut", or "bunch".

        Returns:
        - int: Inferred index of kernel optimal kernel application.

        Examples:
        >>> BeanMachine._position_infer([0, 3, 2, 4, 1, 5, 2, 6, 3, 7], 1.5, [1, 3, 1], border="cut")
        0  # Distance: 5.5

        >>> BeanMachine._position_infer([6, 4, 2, 0, 2, 4, 6], 5.5, [1, 3, 1], border="wrap")
        6  # Distance: 3.5

        >>> BeanMachine._position_infer([0, 2, 4, 6, 4, 2, 0], 5.5, [1, 3, 1], border="bunch")
        6  # Distance: 4.5
        """

        # [Function body remains the same]
        min_distance = float('inf')
        best_center_idx = 0
        n = len(ds)
        k_len = len(kernel)
        half_k_len = k_len // 2

        for i in range(n):
            weighted_distance = 0.0
            for j in range(-half_k_len, half_k_len + 1):
                idx = i + j
                if idx < 0:
                    if border == "wrap":
                        idx = n + idx
                    elif border == "cut":
                        continue
                    else:
                        idx = 0
                elif idx >= n:
                    if border == "wrap":
                        idx = idx - n
                    elif border == "cut":
                        continue
                    else:
                        idx = n - 1
                weighted_distance += ds[idx] * kernel[j + half_k_len]
            if weighted_distance < min_distance:
                min_distance = weighted_distance
                best_center_idx = i
        return best_center_idx

    def back(self, input_hv: AbstractBHV, threshold=4) -> float:
        ds = [1. - hv.bit_error_rate(input_hv) for hv in self.hvs]
        return self._position_infer(ds, self.kernel, self.border)/(self.ncolumns - 1)


class Periodic(Embedding[float]):
    @classmethod
    def simple(cls, hvt: Type[AbstractBHV], divisions: int, periods: list[float]):
        # e.g. Periodic.simple(BHV, 4, [1, 1, 1, 1/2])
        cells = []

        for period in periods:
            roffset = uniform(0, period)
            sensitive = period/divisions
            for i in range(divisions):
                offset = (i*sensitive + roffset) % period
                cells.append((period, sensitive, offset))

        return cls(hvt, cells)

    @classmethod
    def random(cls, hvt: Type[AbstractBHV], n: int, p: Callable[[], float], s: Callable[[], float], o: Callable[[], float]):
        # e.g. Periodic.random(BHV, 25, lambda: uniform(.5, 1.), lambda: uniform(0, 1.), lambda: triangular(.1, .5, .2))
        cells = [(p(), s(), o()) for _ in range(n)]

        return cls(hvt, cells)

    def __init__(self, hvt: Type[AbstractBHV], cells: list[tuple[float, float, float]]):
        self.hvt = hvt
        self.cells = cells
        self.cell_hvs = hvt.nrand(len(cells))

    def forward(self, x: float) -> AbstractBHV:
        return self.hvt.majority([hv for (period, sensitive, offset), hv in zip(self.cells, self.cell_hvs) if
                                  (abs(x - offset) % period) <= sensitive])

    def back(self, input_hv: AbstractBHV, threshold=.1) -> Optional[float]:
        raise NotImplementedError()


class Collapse(Embedding[Iterable[float]]):
    def __init__(self, hvt: Type[AbstractBHV]):
        self.hvt = hvt

    def forward(self, x: Iterable[float]) -> AbstractBHV:
        return self.hvt.from_bitstream(random() < v for v in x)

    def back(self, input_hv: AbstractBHV, soft=.1) -> Optional[Iterable[float]]:
        i = 1. - soft
        o = soft
        return (i if b else o for b in input_hv.bits())
