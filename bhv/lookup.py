from .abstract import AbstractBHV
from typing import TypeVar, Generic, Iterable, Iterator, Optional, Mapping
from math import comb


K = TypeVar("K")


class Store(Generic[K]):
    @classmethod
    def auto_associative(cls, vs: Iterable[AbstractBHV]) -> 'Store[int]':
        return cls(dict(enumerate(vs)))

    def __init__(self, hvs: Mapping[K, AbstractBHV]):
        self.hvs = hvs

    def related(self, v: AbstractBHV, threshold=6) -> Iterator[K]:
        raise NotImplementedError()

    def closest(self, v: AbstractBHV) -> Optional[K]:
        raise NotImplementedError()


class StoreList(Store):
    def related(self, v: AbstractBHV, threshold=6) -> Iterator[K]:
        for k, v_ in self.hvs.items():
            if v_.std_apart(v) <= threshold:
                yield k

    def closest(self, v: AbstractBHV) -> Optional[K]:
        closest = None
        shortest_distance = float('inf')

        for k, v_ in self.hvs.items():
            d = v_.std_apart(v)
            if d < shortest_distance:
                closest = k
                shortest_distance = d

        return closest


class StoreRejectList(Store):
    def __init__(self, hvs: dict[K, AbstractBHV]):
        super().__init__(hvs)
        vs = list(hvs.values())
        representative = vs[0]
        self.bundle = representative.majority(vs)
        self.bundle_size = len(vs)
        self.included_std = AbstractBHV.frac_to_std(AbstractBHV.maj_ber(self.bundle_size))
    # expected number of bits flipped compared to the majority of N random hvs
    # E[bit_error_rate(v, MAJ(v, v_0, ..v_n))]
    # E[v_q != MAJ(v, v_0, ..v_n)_q] WLOG
    # E[1_q != MAJ(1, v_0, ..v_n)_q]
    # E[1 != MAJ(1, v_0, ..v_n)_q]
    # PoiBin(1, af(v_0), ..af(v_n)).cdf(n//2)
    # further assuming af(v_0) == af(v_k) == 0.5,
    # for n=1,3,5,...
    # E[v_q != MAJ(v, v_0, ..v_n)_q]
    # E[MAJ(v, v_0, ..v_n)_q]
    # E[1 | SUM(v_k) > n/2
    #   0 | SUM(v_k) < n/2    # doesn't contribute
    #   v | SUM(v_k) = n/2]   # using af(v) == 0.5
    # in code
    # sum(.5 if sum(bc) == n//2 else (sum(bc) > n//2) for bc in bitconfigs(n))
    # E[SUM(v_k) > n/2] + E[V]
    # (comb(n - 1, (n - 1)//2))/2**n
    # n! / ((n/2)! * (n - n/2)!)

    def related_reject(self, v: AbstractBHV, threshold=6, reject_safety=3) -> Iterator[K]:
        if self.bundle.unrelated(v, self.included_std - reject_safety):
            return
        for k, v_ in self.hvs.items():
            if v_.std_apart(v) <= threshold:
                yield k

    def related(self, v: AbstractBHV, threshold=6) -> Iterator[K]:
        for k, v_ in self.hvs.items():
            if v_.std_apart(v) <= threshold:
                yield k

    def find_closest(self, v: AbstractBHV) -> Optional[K]:
        closest = None
        shortest_distance = float('inf')

        for k, v_ in self.hvs.items():
            d = v_.std_apart(v)
            if d < shortest_distance:
                closest = k
                shortest_distance = d

        return closest


class StoreChunks(Store):
    def __init__(self, hvs: dict[K, AbstractBHV], chunk_size=50):
        super().__init__(hvs)
        ks = list(hvs.keys())
        vs = list(hvs.values())
        representative = vs[0]
        self.chunks = [(representative.majority(vs[i*chunk_size:(i+1)*chunk_size]), dict(zip(ks[i*chunk_size:(i+1)*chunk_size], vs[i*chunk_size:(i+1)*chunk_size])))
                       for i in range(len(vs)//chunk_size+1)]
        self.total_size = len(vs)
        self.chunk_size = chunk_size
        self.included_std = AbstractBHV.frac_to_std(AbstractBHV.maj_ber(chunk_size))

    def related(self, v: AbstractBHV, threshold=6) -> Iterator[K]:
        for maj, chunk in self.chunks:
            if not maj.unrelated(v, self.included_std - 3):
                for k, v_ in chunk.items():
                    if v_.related(v, threshold):
                        yield k

    def find_closest(self, v: AbstractBHV) -> Optional[K]:
        closest = None
        shortest_distance = float('inf')

        for k, v_ in self.hvs.items():
            d = v_.std_apart(v)
            if d < shortest_distance:
                closest = k
                shortest_distance = d

        return closest
