from bhv.abstract import AbstractBHV
from typing import Generic, TypeVar, Type, Optional

T = TypeVar('T')


class Embedding(Generic[T]):
    def forward(self, x: T) -> AbstractBHV:
        raise NotImplementedError()

    def back(self, x: AbstractBHV) -> Optional[T]:
        raise NotImplementedError()


class Random(Embedding):
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
        best_score = 1.
        for x, hv in self.hvs.items():
            score = input_hv.bit_error_rate(hv)
            if score < best_score and score < threshold:
                best_score = score
                best_x = x
        return best_x


class InterpolateBetween(Embedding):
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



def make_mask(on, n):
    ar = torch.zeros(n, dtype=torch.bool)
    ar[:on] = True
    perm = torch.randperm(n)
    return ar[perm]

def binarize_3(x, n):
    on = int(float(x)*(n+1))
    if x == 1.0:
        return torch.ones(n, dtype=torch.bool)
    ar = torch.zeros(n, dtype=torch.bool)
    ar[:on] = True
    return ar

def filln(xs, n):
    ar = torch.empty(len(xs)*n, dtype=torch.bool)
    for i, x in enumerate(xs):
        ar[i*n:(i + 1)*n] = binarize_3(x, n)
    return ar
