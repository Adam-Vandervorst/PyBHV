from dataclasses import dataclass, field, fields
from .abstract import *
from .shared import stable_hashcode


class Symbolic:
    def nodename(self, **kwargs):
        return f"{type(self).__name__.upper()}"

    def nodeid(self, structural=False, **kwargs):
        return f"{type(self).__name__}{stable_hashcode(self).replace('-', '') if structural else str(id(self))}"

    def children(self, **kwargs):
        return [(getattr(self, f.name), f.name) for f in fields(self) if type(f.type) is type and issubclass(f.type, Symbolic)]

    def graphviz(self, structural=False, done=None, **kwargs):
        noden = self.nodeid(structural, **kwargs)
        if done is None:
            done = set()
        if noden in done:
            return
        done.add(noden)
        kwargs |= dict(done=done, structural=structural)
        nodecs = self.children(**kwargs)
        print(f"{noden} [label=\"{self.nodename(**kwargs)}\"];")
        for c, label in nodecs:
            print(f"{noden} -> {c.nodeid(**kwargs)} [label=\"{label}\"];")
            c.graphviz(**kwargs)

    def show(self, **kwargs):
        """
        Shows the expression tree code-style.
        Only works on referentially transparent DAGs.
        :param kwargs: drawing options
        :return: str
        """
        raise NotImplementedError()

    def instantiate(self, **kwargs):
        raise NotImplementedError()

    def execute(self, random: 'dict[int, AbstractBHV]' = None,
                rand2: 'dict[int, AbstractBHV]' = None,
                rand: 'dict[int, AbstractBHV]' = None,
                randomperms: 'dict[int, MemoizedPermutation]' = None,
                calculated = None, **kwargs):
        if random is None: random = {}
        if rand2 is None: rand2 = {}
        if rand is None: rand = {}
        if randomperms is None: randomperms = {}
        if calculated is None: calculated = {}
        kwargs |= dict(random=random, rand2=rand2, rand=rand, randomperms=randomperms, calculated=calculated)
        if id(self) in calculated:
            return calculated[id(self)]
        else:
            result = self.instantiate(**kwargs)
            calculated[id(self)] = result
            return result


class SymbolicPermutation(Symbolic, MemoizedPermutation):
    _permutations: 'dict[int | tuple[int, ...], Self]' = {}

    @classmethod
    def random(cls) -> 'SymbolicPermutation':
        return PermRandom()

    def __mul__(self, other: 'SymbolicPermutation') -> 'SymbolicPermutation':
        return PermCompose(self, other)

    def __invert__(self) -> 'SymbolicPermutation':
        return PermInvert(self)

    def __call__(self, hv: 'SymbolicBHV') -> 'SymbolicBHV':
        return PermApply(self, hv)

@dataclass
class PermVar(SymbolicPermutation):
    name: str
    def nodename(self, **kwargs):
        return self.name

    def show(self, **kwargs):
        symbolic_var = kwargs.get("symbolic_var", False)
        return f"ParmVar(\"{self.name}\")" if symbolic_var else self.name

    def instantiate(self, **kwargs):
        permvars = kwargs.get("permvars")
        if permvars is None:
            raise RuntimeError(f"No Perm vars supplied but tried to instantiate `{self.name}`")
        elif self.name not in permvars:
            raise RuntimeError(f"Perm var `{self.name}` not in permvars ({set(permvars.keys())})")
        else:
            return permvars[self.name]
randpermid = 0
def next_perm_id():
    global randpermid
    randpermid += 1
    return randpermid
@dataclass
class PermRandom(SymbolicPermutation):
    id: int = field(default_factory=next_perm_id)

    def show(self, **kwargs):
        impl = kwargs.get("impl", "")
        random_id = kwargs.get("random_id", False)
        return f"<{impl}random {self.id}>" if random_id else impl + "random()"

    def instantiate(self, **kwargs):
        randomperms = kwargs.get("randomperms")
        if self.id in randomperms:
            return randomperms[self.id]
        else:
            r = kwargs.get("perm").random()
            randomperms[self.id] = r
            return r
@dataclass
class PermCompose(SymbolicPermutation):
    l: SymbolicPermutation
    r: SymbolicPermutation

    def show(self, **kwargs):
        return f"({self.l.show(**kwargs)} * {self.r.show(**kwargs)})"

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs) * self.r.execute(**kwargs)
@dataclass
class PermInvert(SymbolicPermutation):
    p: SymbolicPermutation

    def show(self, **kwargs):
        return f"(~{self.p.show(**kwargs)})"

    def instantiate(self, **kwargs):
        return ~self.p.execute(**kwargs)


class SymbolicBHV(Symbolic, AbstractBHV):
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

    def permute(self, permutation_id: 'int | tuple[int, ...]') -> Self:
        return Permute(permutation_id, self)

    def swap_halves(self) -> Self:
        return SwapHalves(self)

    def rehash(self) -> Self:
        return ReHash(self)

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

    def active(self) -> int:
        return Active(self)

    def bias_rel(self, other: Self, rel: Self) -> float:
        return BiasRel(rel, self, other)

    def unrelated(self, other: Self, stdvs=6) -> bool:
        return Unrelated(self, other, stdvs)

@dataclass
class PermApply(SymbolicBHV):
    p: SymbolicPermutation
    v: SymbolicBHV

    def show(self, **kwargs):
        return f"{self.p.show(**kwargs)}({self.v.show(**kwargs)})"

    def instantiate(self, **kwargs):
        return self.p.execute(**kwargs)(self.v.execute(**kwargs))
@dataclass
class Var(SymbolicBHV):
    name: str
    def nodename(self, **kwards):
        return self.name

    def show(self, **kwargs):
        symbolic_var = kwargs.get("symbolic_var", False)
        return f"Var(\"{self.name}\")" if symbolic_var else self.name

    def instantiate(self, **kwargs):
        vars = kwargs.get("vars")
        if vars is None:
            raise RuntimeError(f"No Perm vars supplied but tried to instantiate `{self.name}`")
        elif self.name not in vars:
            raise RuntimeError(f"Perm var `{self.name}` not in permvars ({set(vars.keys())})")
        else:
            return vars[self.name]
@dataclass
class Zero(SymbolicBHV):
    def show(self, **kwargs):
        return kwargs.get("impl", "") + "ZERO"

    def instantiate(self, bhv, **kwargs):
        return kwargs.get("bhv").ZERO
@dataclass
class One(SymbolicBHV):
    def show(self, **kwargs):
        return kwargs.get("impl", "") + "ONE"

    def instantiate(self, **kwargs):
        return kwargs.get("bhv").ONE
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

    def show(self, **kwargs):
        impl = kwargs.get("impl", "")
        random_id = kwargs.get("random_id", False)
        return f"<{impl}rand {self.id}>" if random_id else impl + "rand()"

    def instantiate(self, **kwargs):
        rand = kwargs.get("rand")
        if self.id in rand:
            return rand[self.id]
        else:
            r = kwargs.get("bhv").rand()
            rand[self.id] = r
            return r
@dataclass
class Rand2(SymbolicBHV):
    power: int

    def show(self, **kwargs):
        return kwargs.get("impl", "") + f"rand2({self.power})"

    def instantiate(self, **kwargs):
        rand2 = kwargs.get("rand2")
        if self.id in rand2:
            return rand2[self.id]
        else:
            r = kwargs.get("bhv").rand2()
            rand2[self.id] = r
            return r
@dataclass
class Random(SymbolicBHV):
    active: float

    def show(self, **kwargs):
        return kwargs.get("impl", "") + f"random({self.active})"

    def instantiate(self, **kwargs):
        random = kwargs.get("random")
        if self.id in random:
            return random[self.id]
        else:
            r = kwargs.get("bhv").random()
            random[self.id] = r
            return r
@dataclass
class Majority(SymbolicBHV):
    vs: list[SymbolicBHV]

    def children(self, **kwargs):
        return list(zip(self.vs, map(str, range(len(self.vs)))))

    def show(self, **kwargs):
        return kwargs.get("impl", "") + f"majority({[v.show(**kwargs) for v in self.vs]})"

    def instantiate(self, **kwargs):
        return kwargs.get("bhv").majority([v.execute(**kwargs) for v in self.vs])
@dataclass
class Permute(SymbolicBHV):
    id: 'int | tuple[int, ...]'
    v: SymbolicBHV

    def show(self, **kwargs):
        return f"{self.v.show(**kwargs)}.permute({self.id})"

    def instantiate(self, **kwargs):
        return self.v.execute(**kwargs).permute(self.id)
@dataclass
class SwapHalves(SymbolicBHV):
    v: SymbolicBHV

    def show(self, **kwargs):
        return f"{self.v.show(**kwargs)}.swap_halves()"

    def instantiate(self, **kwargs):
        return self.v.execute(**kwargs).swap_halves()
@dataclass
class ReHash(SymbolicBHV):
    v: SymbolicBHV

    def show(self, **kwargs):
        return f"{self.v.show(**kwargs)}.rehash()"

    def instantiate(self, **kwargs):
        return self.v.execute(**kwargs).rehash()
@dataclass
class Eq:
    l: SymbolicBHV
    r: SymbolicBHV

    def show(self, **kwargs):
        return f"({self.l.show(**kwargs)} == {self.r.show(**kwargs)})"

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs) == self.r.execute(**kwargs)
@dataclass
class Xor(SymbolicBHV):
    l: SymbolicBHV
    r: SymbolicBHV

    def show(self, **kwargs):
        return f"({self.l.show(**kwargs)} ^ {self.r.show(**kwargs)})"

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs) ^ self.r.execute(**kwargs)
@dataclass
class And(SymbolicBHV):
    l: SymbolicBHV
    r: SymbolicBHV

    def show(self, **kwargs):
        return f"({self.l.show(**kwargs)} & {self.r.show(**kwargs)})"

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs) & self.r.execute(**kwargs)
@dataclass
class Or(SymbolicBHV):
    l: SymbolicBHV
    r: SymbolicBHV

    def show(self, **kwargs):
        return f"({self.l.show(**kwargs)} | {self.r.show(**kwargs)})"

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs) | self.r.execute(**kwargs)
@dataclass
class Invert(SymbolicBHV):
    v: SymbolicBHV

    def show(self, **kwargs):
        return f"(~{self.v.show(**kwargs)})"

    def instantiate(self, **kwargs):
        return ~self.v.execute(**kwargs)
@dataclass
class Select(SymbolicBHV):
    cond: SymbolicBHV
    when1: SymbolicBHV
    when0: SymbolicBHV
    def nodename(self, compact_select=True, **kwargs):
        return f"ON {self.cond.nodename()}" if compact_select else super().nodename(**kwargs)
    def children(self, compact_select=True, **kwargs):
        return [(self.when1, "1"), (self.when0, "0")] if compact_select else super().children(**kwargs)

    def show(self, **kwargs):
        return f"{self.cond.show(**kwargs)}.select({self.when1.show(**kwargs)}, {self.when0.show(**kwargs)})"

    def instantiate(self, **kwargs):
        return self.cond.execute(**kwargs).select(self.when1.execute(**kwargs), self.when0.execute(**kwargs))
@dataclass
class Active(Symbolic):
    v: SymbolicBHV

    def show(self, **kwargs):
        return f"{self.v.show(**kwargs)}.active()"

    def instantiate(self, **kwargs):
        return self.v.execute(**kwargs).active()
@dataclass
class BiasRel(SymbolicBHV):
    rel: SymbolicBHV
    l: SymbolicBHV
    r: SymbolicBHV

    def show(self, **kwargs):
        return f"{self.l.show(**kwargs)}.bias_rel({self.r.show(**kwargs)}, {self.rel.show(**kwargs)})"

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs).bias_rel(self.r.execute(**kwargs), self.rel.execute(**kwargs))
@dataclass
class Related(Symbolic):
    l: SymbolicBHV
    r: SymbolicBHV
    stdvs: float

    def show(self, **kwargs):
        return f"{self.l.show(**kwargs)}.related({self.r.show(**kwargs)}, {self.stdvs})"

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs).related(self.r.execute(**kwargs), self.stdvs)
@dataclass
class Unrelated(Symbolic):
    l: SymbolicBHV
    r: SymbolicBHV
    stdvs: float

    def show(self, **kwargs):
        return f"{self.l.show(**kwargs)}.unrelated({self.r.show(**kwargs)}, {self.stdvs})"

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs).unrelated(self.r.execute(**kwargs), self.stdvs)
