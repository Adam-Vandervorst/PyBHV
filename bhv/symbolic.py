from dataclasses import dataclass, field, fields
from .abstract import *
from .shared import stable_hashcode

done = set()
class SymbolicBHV(AbstractBHV):
    def nodename(self):
        return f"{type(self).__name__.upper()}"

    def nodeid(self):
        return f"{type(self).__name__}{getattr(self, 'name', stable_hashcode(self).replace('-', ''))}"

    def graphviz(self):
        noden = self.nodeid()
        if noden in done:
            return
        done.add(noden)
        nodecs = [(getattr(self, f.name), f.name) for f in fields(self) if issubclass(f.type, SymbolicBHV)]
        print(f"{noden} [label=\"{self.nodename()}\"];")
        for c, label in nodecs:
            print(f"{noden} -> {c.nodeid()};")
            c.graphviz()


    @classmethod
    def rand(cls) -> Self:
        return Rand()

    @classmethod
    def rand2(cls, power=1) -> Self:
        return Rand2(power)

    @classmethod
    def random(cls, active=0.5) -> Self:
        return Random(active)

    @classmethod
    def majority(cls, vs: list[Self]) -> Self:
        return Majority(vs)

    def __eq__(self, other: Self) -> bool:
        return Eq(self, other)

    def __xor__(self, other: Self) -> Self:
        return Xor(self, other)

    def __and__(self, other: Self) -> Self:
        return And(self, other)

    def __or__(self, other: Self) -> Self:
        return Or(self, other)

    def __invert__(self) -> Self:
        return Invert(self)

    def select(self, when1: Self, when0: Self) -> Self:
        return Select(self, when0, when1)

    def mix(self, other: Self, pick_l=0.5) -> Self:
        return Mix(pick_l, self, other)

    def active(self) -> int:
        return Active(self)

    def bias_rel(self, other: Self, rel: Self) -> float:
        return BiasRel(rel, self, other)

    # def active_fraction(self) -> float:
    #     return self.active()/DIMENSION
    #
    # def hamming(self, other: Self) -> int:
    #     return (self ^ other).active()
    #
    # def bit_error_rate(self, other: Self) -> float:
    #     return (self ^ other).active_fraction()
    #
    # def jaccard(self, other: Self) -> float:
    #     return 1. - float((self & other).active()) / float((self | other).active() + 1E-7)
    #
    # def cosine(self, other: Self) -> float:
    #     return 1 - float((self & other).active()) / float(self.active() * other.active() + 1E-7)**.5
    #
    # def zscore(self, other: Self) -> float:
    #     p = 0.5
    #     n = NormalDist(DIMENSION*p, (DIMENSION*p*(1 - p))**.5)
    #     return n.zscore(self.hamming(other))
    #
    # def pvalue(self, other: Self) -> float:
    #     p = 0.5
    #     n = NormalDist(DIMENSION*p, (DIMENSION*p*(1 - p))**.5)
    #     s = n.cdf(self.hamming(other))
    #     return 2*min(s, 1 - s)
    #
    # def sixsigma(self, other: Self) -> bool:
    #     return abs(self.zscore(other)) < 6



@dataclass
class Var(SymbolicBHV):
    name: str

    def nodename(self):
        return f"{self.name}"
@dataclass
class Zero(SymbolicBHV):
    pass
@dataclass
class One(SymbolicBHV):
    pass
SymbolicBHV.ZERO = Zero()
SymbolicBHV.ONE = One()
randid = 0
def next_id():
    global randid
    randid += 1
    return randid
@dataclass
class Rand(SymbolicBHV):
    id: int = field(default_factory=next_id)
@dataclass
class Rand2(SymbolicBHV):
    power: int
@dataclass
class Random(SymbolicBHV):
    active: float
@dataclass
class Majority(SymbolicBHV):
    vs: list[SymbolicBHV]
@dataclass
class Eq:
    l: SymbolicBHV
    r: SymbolicBHV
@dataclass
class Xor(SymbolicBHV):
    l: SymbolicBHV
    r: SymbolicBHV
@dataclass
class And(SymbolicBHV):
    l: SymbolicBHV
    r: SymbolicBHV
@dataclass
class Or(SymbolicBHV):
    l: SymbolicBHV
    r: SymbolicBHV
@dataclass
class Invert(SymbolicBHV):
    v: SymbolicBHV
@dataclass
class Select(SymbolicBHV):
    cond: SymbolicBHV
    when1: SymbolicBHV
    when0: SymbolicBHV

    def graphviz(self):
        noden = self.nodeid()
        if noden in done:
            return
        done.add(noden)
        nodecs = [(self.when1, "1"), (self.when0, "0")]
        print(f"{noden} [label=\"ON {self.cond.nodename()}\"];")
        for c, label in nodecs:
            print(f"{noden} -> {c.nodeid()} [label=\"{label}\"];")
            c.graphviz()
@dataclass
class Mix(SymbolicBHV):
    frac: float
    l: SymbolicBHV
    r: SymbolicBHV
@dataclass
class Active:
    v: SymbolicBHV
@dataclass
class BiasRel:
    rel: SymbolicBHV
    l: SymbolicBHV
    r: SymbolicBHV
