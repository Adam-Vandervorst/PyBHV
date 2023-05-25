from dataclasses import dataclass, field, fields
from .abstract import *
from .shared import stable_hashcode, bitconfigs, unique_by_id, format_multiple, format_list
from .slice import Slice


class Symbolic:
    name = None

    def named(self, name):
        if name is not None: self.name = name
        return self

    def nodename(self, **kwargs):
        return self.name or f"{type(self).__name__.upper()}"

    def nodeid(self, structural=False, **kwargs):
        return f"{type(self).__name__}{stable_hashcode(self, try_cache=True).replace('-', '') if structural else str(id(self))}"

    def labeled_children(self, **kwargs):
        return [(getattr(self, f.name), f.name) for f in fields(self) if type(f.type) is type and issubclass(f.type, Symbolic)]

    def children(self, **kwargs):
        return [c for c, _ in self.labeled_children(**kwargs)]

    def vars(self) -> list[str]:
        return [v.name for v in self.preorder(lambda x: isinstance(x, Var))]

    def substitute(self, vars):
        if isinstance(self, Var) and self.name in vars:
            return vars[self.name]
        else:
            return self.map(lambda x: x.substitute(vars))

    def substitute_term(self, f):
        res = f(self)
        if res:
            return res
        else:
            return self.map(lambda x: x.substitute_term(f))

    def reconstruct(self, *cs):
        assert not self.labeled_children()
        return self

    def map(self, f):
        return self.reconstruct(*map(f, self.children())).named(self.name)

    def constant(self):
        return self.map(lambda x: None)

    def draw_node(self, **kwargs):
        print(f"{self.nodeid(**kwargs)} [label=\"{self.nodename(**kwargs)}\"];")

    def draw_edges(self, **kwargs):
        for c, label in self.labeled_children(**kwargs):
            print(f"{c.nodeid(**kwargs)} -> {self.nodeid(**kwargs)} [label=\"{label}\"];")

    def draw_children(self, **kwargs):
        for c, label in self.labeled_children(**kwargs):
            c.graphviz(**kwargs)

    def graphviz(self, structural=False, done=None, **kwargs):
        noden = self.nodeid(structural, **kwargs)
        if done is None:
            done = set()
        if noden in done:
            return
        done.add(noden)
        kwargs |= dict(done=done, structural=structural)
        self.draw_node(**kwargs)
        self.draw_edges(**kwargs)
        self.draw_children(**kwargs)

    def show_program(self, name="f", indent="    ", **kwargs):
        kwargs["indent"] = indent
        kwargs["aindent"] = indent
        kwargs["toplevel"] = True
        def extracted(x: Symbolic):
            return x.map(lambda x: x if isinstance(x, Var) else Var(f"_{subexpressions.index(x)}"))
        subexpressions = unique_by_id(self.preorder(lambda x: not isinstance(x, Var))[1:])
        subexpressions.reverse()
        free_vars = unique_by_id(self.preorder(lambda x: isinstance(x, Var)))
        arguments = [fv.show(**kwargs) for fv in free_vars]
        lines = [(x.name or f"_{i}") + " = " + extracted(x).show(**kwargs) for i, x in enumerate(subexpressions)]
        last = f"return {extracted(self).show(**kwargs)}"
        code = format_multiple([*lines, last], start=f"def {name}({format_multiple(arguments, sep=', ')}):", indent=indent, newline_threshold=0)
        return code

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

    def optimal_sharing(self, form=None, **kwargs):
        if form is None: form = {}
        kwargs |= dict(form=form)

        sh = stable_hashcode(self, try_cache=True)
        if sh in form:
            return form[sh]
        else:
            r = self.map(lambda c: c.optimal_sharing(**kwargs))
            shr = stable_hashcode(r, try_cache=True)
            if shr in form:
                raise RuntimeError("Includes check failed")
            form[shr] = r
            return r

    def internal_size(self):
        return 1

    def size(self, counted=None, recount=False, **kwargs):
        if counted is None: counted = {}
        kwargs |= dict(counted=counted)
        if id(self) in counted:
            return counted[id(self)] if recount else 0
        else:
            t = self.internal_size()
            for c in self.children():
                t += c.size(**kwargs)
            counted[id(self)] = t
            return t

    def simplify(self, **kwargs):
        x = self
        while True:
            x_ = x.map(lambda c: c.simplify(**kwargs))
            x_r = x_.reduce(**kwargs)
            if x_r is not None:
                x_ = x_r
            if x == x_:
                return x
            else:
                x = x_

    def reduce(self, **kwargs):
        return None

    def preorder(self, p=lambda x: True):
        return [self]*p(self) + [v for c in self.children() for v in c.preorder(p)]

    def truth_assignments(self, vars):
        configs = bitconfigs(len(vars))

        return [self.execute(vars={v.name: Slice(b) for b, v in zip(config, vars)}, bhv=Slice).b
                for config in configs]


@dataclass
class List(Symbolic):
    xs: list[Symbolic]

    def show(self, **kwargs):
        return format_list((x.show(**kwargs) for x in self.xs),
                           **{k: kwargs[k] for k in ["indent", "aindent", "newline_threshold"] if k in kwargs})

    def labeled_children(self, **kwargs):
        return [(x, str(i)) for i, x in enumerate(self.xs)]

    def reconstruct(self, *cs):
        return List(cs)

    def graphviz(self, structural=False, done=None, **kwargs):
        noden = self.nodeid(structural, **kwargs)
        if done is None:
            done = set()
        if noden in done:
            return
        done.add(noden)
        kwargs |= dict(done=done, structural=structural)
        print(f"subgraph cluster_{self.nodename()} " + "{")
        print(f"label = \"{self.nodename()}\";")
        for c in self.xs:
            c.draw_node(**kwargs)
        print("}")
        for c in self.xs:
            c.draw_edges(**kwargs)
            c.draw_children(**kwargs)

    def instantiate(self, **kwargs):
        return [x.instantiate(**kwargs) for x in self.xs]

    def truth_table(self, vars):
        return [x.truth_assignments(vars) for x in self.xs]


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

    def reconstruct(self, l, r):
        return PermCompose(l, r)

    def show(self, **kwargs):
        brackets = not kwargs.get("toplevel", False)
        return "("*brackets + f"{self.l.show(**kwargs)} * {self.r.show(**kwargs)}" + ")"*brackets

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs) * self.r.execute(**kwargs)
@dataclass
class PermInvert(SymbolicPermutation):
    p: SymbolicPermutation

    def reconstruct(self, p):
        return PermInvert(p)

    def show(self, **kwargs):
        brackets = not kwargs.get("toplevel", False)
        return "("*brackets + f"~{self.p.show(**kwargs)}" + ")"*brackets

    def instantiate(self, **kwargs):
        return ~self.p.execute(**kwargs)


class SymbolicBHV(Symbolic, AbstractBHV):
    @classmethod
    def synth(cls, vs, t):
        assert 2**len(vs) == len(t)
        if vs:
            return vs[0].select(
                    cls.synth(vs[1:], t[:len(t)//2]),
                    cls.synth(vs[1:], t[len(t)//2:]))
        else:
            return cls.ONE if t[0] else cls.ZERO

    @classmethod
    def rand(cls) -> Self:
        return Rand()

    @classmethod
    def rand2(cls, power: int) -> Self:
        assert power >= 0
        return Rand2(power)

    @classmethod
    def random(cls, active: float) -> Self:
        assert 0. <= active <= 1.
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

    def active_fraction(self) -> int:
        return ActiveFraction(self)

    def bias_rel(self, other: Self, rel: Self) -> float:
        return BiasRel(rel, self, other)

    def unrelated(self, other: Self, stdvs=6) -> bool:
        return Unrelated(self, other, stdvs)

    def expected_active_fraction(self, **kwargs):
        raise NotImplementedError()


@dataclass
class PermApply(SymbolicBHV):
    p: SymbolicPermutation
    v: SymbolicBHV

    def reconstruct(self, p, v):
        return PermApply(p, v)

    def show(self, **kwargs):
        return f"{self.p.show(**kwargs)}({self.v.show(**kwargs)})"

    def instantiate(self, **kwargs):
        return self.p.execute(**kwargs)(self.v.execute(**kwargs))

    def expected_active_fraction(self, **kwargs):
        return self.v.expected_active(**kwargs)
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
            raise RuntimeError(f"No vars supplied but tried to instantiate `{self.name}`")
        elif self.name not in vars:
            raise RuntimeError(f"Var `{self.name}` not in vars ({set(vars.keys())})")
        else:
            return vars[self.name]

    def expected_active_fraction(self, **kwargs):
        return kwargs.get("vars").get(self.name)
@dataclass
class Zero(SymbolicBHV):
    def show(self, **kwargs):
        return kwargs.get("impl", "") + "ZERO"

    def instantiate(self, **kwargs):
        return kwargs.get("bhv").ZERO

    def expected_active_fraction(self, **kwargs):
        return 0.
@dataclass
class One(SymbolicBHV):
    def show(self, **kwargs):
        return kwargs.get("impl", "") + "ONE"

    def instantiate(self, **kwargs):
        return kwargs.get("bhv").ONE

    def expected_active_fraction(self, **kwargs):
        return 1.
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

    def expected_active_fraction(self, **kwargs):
        return .5
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
            r = kwargs.get("bhv").rand2(self.power)
            rand2[self.id] = r
            return r

    def expected_active_fraction(self, **kwargs):
        return 1/self.power
@dataclass
class Random(SymbolicBHV):
    frac: float

    def show(self, **kwargs):
        return kwargs.get("impl", "") + f"random({self.frac})"

    def instantiate(self, **kwargs):
        random = kwargs.get("random")
        if self.id in random:
            return random[self.id]
        else:
            r = kwargs.get("bhv").random(self.frac)
            random[self.id] = r
            return r

    def expected_active_fraction(self, **kwargs):
        return self.frac
@dataclass
class Majority(SymbolicBHV):
    vs: list[SymbolicBHV]

    def labeled_children(self, **kwargs):
        return list(zip(self.vs, map(str, range(len(self.vs)))))

    def reconstruct(self, *cs):
        return Majority(cs)

    def show(self, **kwargs):
        args = format_list((v.show(**kwargs) for v in self.vs), **{k: kwargs[k] for k in ["indent", "aindent", "newline_threshold"] if k in kwargs})
        return kwargs.get("impl", "") + f"majority({args})"

    def instantiate(self, **kwargs):
        return kwargs.get("bhv").majority([v.execute(**kwargs) for v in self.vs])

    def expected_active_fraction(self, **kwargs):
        from .poibin import PoiBin
        return 1. - PoiBin([v.expected_active_fraction(**kwargs) for v in self.vs]).cdf(len(self.vs)//2)
@dataclass
class Permute(SymbolicBHV):
    id: 'int | tuple[int, ...]'
    v: SymbolicBHV

    def reconstruct(self, v):
        return Permute(self.id, v)

    def show(self, **kwargs):
        return f"{self.v.show(**kwargs)}.permute({self.id})"

    def instantiate(self, **kwargs):
        return self.v.execute(**kwargs).permute(self.id)

    def expected_active_fraction(self, **kwargs):
        return self.v.expected_active_fraction(**kwargs)
@dataclass
class SwapHalves(SymbolicBHV):
    v: SymbolicBHV

    def reconstruct(self, v):
        return SwapHalves(v)

    def show(self, **kwargs):
        return f"{self.v.show(**kwargs)}.swap_halves()"

    def instantiate(self, **kwargs):
        return self.v.execute(**kwargs).swap_halves()

    def expected_active_fraction(self, **kwargs):
        return self.v.expected_active_fraction(**kwargs)
@dataclass
class ReHash(SymbolicBHV):
    v: SymbolicBHV

    def reconstruct(self, v):
        return ReHash(v)

    def show(self, **kwargs):
        return f"{self.v.show(**kwargs)}.rehash()"

    def instantiate(self, **kwargs):
        return self.v.execute(**kwargs).rehash()

    def expected_active_fraction(self, **kwargs):
        return .5
@dataclass
class Eq(Symbolic):
    l: SymbolicBHV
    r: SymbolicBHV

    def swap(self):
        return Eq(self.r, self.l)

    def reconstruct(self, l, r):
        return Eq(l, r)

    def show(self, toplevel=True, **kwargs):
        brackets = not toplevel
        kwargs["toplevel"] = False
        return "("*brackets + f"{self.l.show(**kwargs)} == {self.r.show(**kwargs)}" + ")"*brackets

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs) == self.r.execute(**kwargs)
@dataclass
class Xor(SymbolicBHV):
    l: SymbolicBHV
    r: SymbolicBHV

    def reconstruct(self, l, r):
        return Xor(l, r)

    def show(self, **kwargs):
        brackets = not kwargs.get("toplevel", False)
        return "("*brackets + f"{self.l.show(**kwargs)} ^ {self.r.show(**kwargs)}" + ")"*brackets

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs) ^ self.r.execute(**kwargs)

    def reduce(self, **kwargs):
        if self.l == self.ONE:
            return ~self.r
        elif self.r == self.ONE:
            return ~self.l
        elif self.l == self.ZERO:
            return self.r
        elif self.r == self.ZERO:
            return self.l
        elif self.l == self.r:
            return self.ZERO
        elif self.l == ~self.r or ~self.l == self.r:
            return self.ONE
        elif isinstance(self.l, Invert) and isinstance(self.r, Invert):
            return Xor(self.l.v, self.r.v)
        elif isinstance(self.l, And) and isinstance(self.r, And):
            if self.l.l == self.r.l: return And(self.l.l, Xor(self.l.r, self.r.r))
            elif self.l.l == self.r.r: return And(self.l.l, Xor(self.l.r, self.r.l))
            elif self.l.r == self.r.l: return And(self.l.r, Xor(self.l.l, self.r.r))
            elif self.l.r == self.r.r: return And(self.l.r, Xor(self.l.l, self.r.l))

    def expected_active_fraction(self, **kwargs):
        afl = self.l.expected_active_fraction(**kwargs)
        afr = self.r.expected_active_fraction(**kwargs)
        return afl*(1. - afr) + (1. - afl)*afr
@dataclass
class And(SymbolicBHV):
    l: SymbolicBHV
    r: SymbolicBHV

    def reconstruct(self, l, r):
        return And(l, r)

    def show(self, **kwargs):
        brackets = not kwargs.get("toplevel", False)
        return "("*brackets + f"{self.l.show(**kwargs)} & {self.r.show(**kwargs)}" + ")"*brackets

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs) & self.r.execute(**kwargs)

    def reduce(self, **kwargs):
        if self.l == self.ZERO or self.r == self.ZERO:
            return self.ZERO
        elif self.l == self.ONE:
            return self.r
        elif self.r == self.ONE:
            return self.l
        elif isinstance(self.l, Invert) and self.l.v == self.r:
            return self.ZERO
        elif isinstance(self.r, Invert) and self.r.v == self.l:
            return self.ZERO
        elif isinstance(self.l, Invert) and isinstance(self.r, Invert):
            return Invert(Or(self.l.v, self.r.v))

    def expected_active_fraction(self, **kwargs):
        afl = self.l.expected_active_fraction(**kwargs)
        afr = self.r.expected_active_fraction(**kwargs)
        return afl*afr
@dataclass
class Or(SymbolicBHV):
    l: SymbolicBHV
    r: SymbolicBHV

    def reconstruct(self, l, r):
        return Or(l, r)

    def show(self, **kwargs):
        brackets = not kwargs.get("toplevel", False)
        return "("*brackets + f"{self.l.show(**kwargs)} | {self.r.show(**kwargs)}" + ")"*brackets

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs) | self.r.execute(**kwargs)

    def reduce(self, **kwargs):
        if self.l == self.ONE or self.r == self.ONE:
            return self.ONE
        elif self.l == self.ZERO:
            return self.r
        elif self.r == self.ZERO:
            return self.l
        elif isinstance(self.l, Invert) and self.l.v == self.r:
            return self.ONE
        elif isinstance(self.r, Invert) and self.r.v == self.l:
            return self.ONE
        elif isinstance(self.l, Invert) and isinstance(self.r, Invert):
            return Invert(And(self.l.v, self.r.v))

    def expected_active_fraction(self, **kwargs):
        afl = self.l.expected_active_fraction(**kwargs)
        afr = self.r.expected_active_fraction(**kwargs)
        return 1. - ((1. - afl)*(1. - afr))
@dataclass
class Invert(SymbolicBHV):
    v: SymbolicBHV

    def reconstruct(self, v):
        return Invert(v)

    def show(self, **kwargs):
        brackets = not kwargs.get("toplevel", False)
        return "("*brackets + f"~{self.v.show(**kwargs)}" + ")"*brackets

    def instantiate(self, **kwargs):
        return ~self.v.execute(**kwargs)

    def reduce(self, **kwargs):
        if self.v == self.ONE:
            return self.ZERO
        elif self.v == self.ZERO:
            return self.ONE
        elif isinstance(self.v, Invert):
            return self.v.v

    def expected_active_fraction(self, **kwargs):
        return 1. - self.v.expected_active_fraction(**kwargs)
@dataclass
class Select(SymbolicBHV):
    cond: SymbolicBHV
    when1: SymbolicBHV
    when0: SymbolicBHV

    def reconstruct(self, c, w1, w0):
        return Select(c, w1, w0)

    def nodename(self, compact_select=False, **kwargs):
        return f"ON {self.cond.nodename()}" if compact_select else super().nodename(**kwargs)

    def labeled_children(self, compact_select=False, **kwargs):
        return [(self.when1, "1"), (self.when0, "0")] if compact_select else super().labeled_children(**kwargs)

    def show(self, **kwargs):
        return f"{self.cond.show(**kwargs)}.select({self.when1.show(**kwargs)}, {self.when0.show(**kwargs)})"

    def instantiate(self, **kwargs):
        return self.cond.execute(**kwargs).select(self.when1.execute(**kwargs), self.when0.execute(**kwargs))

    def internal_size(self):
        return 3

    def reduce(self, **kwargs):
        expand_select_xor = kwargs.get("expand_select_xor", False)
        expand_select_and_or = kwargs.get("expand_select_and_or", False)

        if self.when1 == self.ONE and self.when0 == self.ZERO:
            return self.cond
        elif self.when1 == self.ZERO and self.when0 == self.ONE:
            return ~self.cond
        elif self.when0 == self.when1:
            return self.when0
        elif self.when1 == self.ONE:
            return self.cond | self.when0
        elif self.when1 == self.ZERO:
            return ~self.cond & self.when0
        elif self.when0 == self.ONE:
            return ~self.cond | self.when1
        elif self.when0 == self.ZERO:
            return self.cond & self.when1
        else:
            if self.when1 == ~self.when0:
                return self.cond ^ self.when0
            elif self.when0 == ~self.when1:
                return self.cond ^ self.when0
            elif isinstance(self.when0, Invert) and isinstance(self.when1, Invert):
                return ~Select(self.cond, self.when1.v, self.when0.v)
            else:
                if expand_select_xor:
                    return self.when0 ^ (self.cond & (self.when0 ^ self.when1))
                elif expand_select_and_or:
                    return (self.cond & self.when1) | (~self.cond & self.when0)

    def expected_active_fraction(self, **kwargs):
        afc = self.cond.expected_active_fraction(**kwargs)
        af1 = self.when1.expected_active_fraction(**kwargs)
        af0 = self.when0.expected_active_fraction(**kwargs)
        return afc*af1 + (1. - afc)*af0
@dataclass
class ActiveFraction(Symbolic):
    v: SymbolicBHV

    def reconstruct(self, v):
        return ActiveFraction(v)

    def show(self, **kwargs):
        return f"{self.v.show(**kwargs)}.active_fraction()"

    def instantiate(self, **kwargs):
        return self.v.execute(**kwargs).active_fraction()
@dataclass
class BiasRel(Symbolic):
    rel: SymbolicBHV
    l: SymbolicBHV
    r: SymbolicBHV

    def reconstruct(self, rel, l, r):
        return BiasRel(rel, l, r)

    def show(self, **kwargs):
        return f"{self.l.show(**kwargs)}.bias_rel({self.r.show(**kwargs)}, {self.rel.show(**kwargs)})"

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs).bias_rel(self.r.execute(**kwargs), self.rel.execute(**kwargs))
@dataclass
class Related(Symbolic):
    l: SymbolicBHV
    r: SymbolicBHV
    stdvs: float

    def reconstruct(self, l, r):
        return Related(l, r, self.stdvs)

    def show(self, **kwargs):
        return f"{self.l.show(**kwargs)}.related({self.r.show(**kwargs)}, {self.stdvs})"

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs).related(self.r.execute(**kwargs), self.stdvs)
@dataclass
class Unrelated(Symbolic):
    l: SymbolicBHV
    r: SymbolicBHV
    stdvs: float

    def reconstruct(self, l, r):
        return Unrelated(l, r, self.stdvs)

    def show(self, **kwargs):
        return f"{self.l.show(**kwargs)}.unrelated({self.r.show(**kwargs)}, {self.stdvs})"

    def instantiate(self, **kwargs):
        return self.l.execute(**kwargs).unrelated(self.r.execute(**kwargs), self.stdvs)
