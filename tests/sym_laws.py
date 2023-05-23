from bhv.symbolic import SymbolicBHV as BHV, Var, Eq, Or, And, Xor, Invert, Majority
from bhv.unification import unify
from bhv.shared import stable_hashcode


x = Var("x")
y = Var("y")
z = Var("z")
u = Var("u")
v = Var("v")
w = Var("w")


def associative3(m): return Eq(m(x, u, m(y, u, z)), m(z, u, m(y, u, x)))
def commutative3_1(m): return Eq(m(x, y, z), m(z, y, x))
def commutative3_2(m): return Eq(m(x, y, z), m(y, x, z))
def distributive3(m, w): return Eq(m(x, y, w(u, v, z)), w(m(x, y, u), m(x, y, v), z))


def associative(m): return Eq(m(m(x, y), z), m(x, m(y, z))).named(f"{m.__name__} associative")
def commutative(m): return Eq(m(x, y), m(y, x)).named(f"{m.__name__} commutative")
def idempotent(m): return Eq(m(x, x), x).named(f"{m.__name__} idempotent")
def self_inverse(m, u): return Eq(m(x, x), u).named(f"{m.__name__} self inverse {u}")
def involutive(f): return Eq(f(f(x)), x).named(f"{f.__name__} involutive")
def equals(f, a, b): return Eq(f(a), b).named(f"{f.__name__}({a}) equals {b}")
def right_unit(m, u): return Eq(m(x, u), x).named(f"{m.__name__} right unit {u}")
def right_absorbing(m, z): return Eq(m(x, z), z).named(f"{m.__name__} right absorbing {z}")
def absorptive(m, w): return Eq(m(x, w(x, y)), x).named(f"{m.__name__} absorbs w.r.t. {w.__name__}")
def distributive(m, w): return Eq(m(x, w(y, z)), w(m(x, y), m(x, z))).named(f"{m.__name__} distributes over {w.__name__}")
def demorgan(f, m, w): return Eq(f(m(x, y)), w(f(x), f(y))).named(f"De Morgan {f.__name__} over {m.__name__}, and {m.__name__}")
def drop_left(f, m): return Eq(f(m(x, y)), m(f(x), y)).named(f"Drop {f.__name__} left into {m.__name__}")
def complement(m, f, z): return Eq(m(x, f(x)), z).named(f"Complement by {f.__name__} under {m.__name__} is {z}")
def propagate(f, m): return Eq(f(m(x)), m(f(x)))
def propagate2(f, m): return Eq(f(m(x, y)), m(f(x), f(y)))
def propagate3(f, m): return Eq(f(m(x, y, z)), m(f(x), f(y), f(z)))
def expand_single_inner(f, k, m, w): return Eq(k(x, y), m(w(x, f(y)), w(f(x), y)))
def expand_single_outer(f, k, m, w): return Eq(k(x, y), m(w(x, y), f(m(x, y))))
def determinism(f): return Eq(f(x), f(x))
def extensionality(f, g): return Eq(f(x), g(x))
def extensionality2(f, g): return Eq(f(x, y), g(x, y))
def extensionality3(f, g): return Eq(f(x, y, z), g(x, y, z))
def extensionality4(f, g): return Eq(f(x, y, z, u), g(x, y, z, u))
def extensionality5(f, g): return Eq(f(x, y, z, u, v), g(x, y, z, u, v))
def extensionality6(f, g): return Eq(f(x, y, z, u, v, w), g(x, y, z, u, v, w))
def extensionality7(f, g): return Eq(f(x, y, z, u, v, w, r), g(x, y, z, u, v, w, r))
def extensionalityN(f, g): return Eq(f(xs), g(xs))
def encode_decode(enc, dec): return Eq(x, dec(enc(x)))
def invariant_under(f, p): return Eq(f(x), f(p(x))).named(f"{f.__name__} is invariant under {p.__name__}")
def invariant_under2(f, p): return Eq(f(x, y), f(p(x), p(y))).named(f"{f.__name__} is invariant under {p.__name__}")
def identity(f): return Eq(f(x), x)
def invariant2(m): return Eq(m(x, y), y)
def invariant3(m): return Eq(m(x, y, z), z)
def invariant3_1(m): return Eq(m(x, x, z), x)
def invariant3_2(m): return Eq(m(x, x, z), z)

def lattice(join): return [associative(join), commutative(join), idempotent(join)]
def bounded_lattice(join, top, bot): return lattice(join) + [right_unit(join, bot), right_absorbing(join, top)]
def xor_props(xor_, bot): return [associative(xor_), commutative(xor_), self_inverse(xor_, bot), right_unit(xor_, bot)]
def not_props(not_, bot, top): return [involutive(not_), equals(not_, top, bot), equals(not_, bot, top)]
def or_and_props(or_, and_): return [distributive(or_, and_), distributive(and_, or_), absorptive(or_, and_), absorptive(and_, or_)]
def gf2(add, mul, one, zero):
    return [
        right_unit(add, zero),
        right_unit(mul, one),
        associative(add), commutative(add),
        associative(mul), commutative(mul),
        idempotent(mul),
        self_inverse(add, zero),
        distributive(mul, add)
    ]
def maj3_inv(maj3, inv):
    return [
        commutative3_1(maj3),
        commutative3_2(maj3),
        invariant3_1(maj3),
        invariant3_2(maj3),
        associative3(maj3),
        distributive3(maj3, maj3),
        propagate3(inv, maj3)
    ]
def boolean_algebra(conj, disj, inv, zero, one):
    return [
        right_unit(disj, zero),
        right_unit(conj, one),
        commutative(conj),
        commutative(disj),
        distributive(conj, disj),
        distributive(disj, conj),
        associative(conj),
        associative(disj),
        complement(disj, inv, one),
        complement(conj, inv, zero),
    ]


def bhv_props():
    extra = [
        demorgan(Invert, Or, And),
        demorgan(Invert, And, Or),
        drop_left(Invert, Xor),
        expand_single_inner(Invert, Xor, Or, And),
        expand_single_outer(Invert, Xor, And, Or),
        distributive(And, Xor),
        complement(Xor, Invert, BHV.ONE),
        invariant_under2(Xor, Invert),
        Eq(x ^ (x & y), x & ~y),
        Eq(x ^ (~x & y), x | y),
        Eq((x & y) ^ (x & z), x & (y ^ z)),
    ]

    return (
        bounded_lattice(And, BHV.ZERO, BHV.ONE) +
        bounded_lattice(Or, BHV.ONE, BHV.ZERO) +
        xor_props(Xor, BHV.ZERO) +
        not_props(Invert, BHV.ZERO, BHV.ONE) +
        or_and_props(Or, And) +
        gf2(Xor, And, BHV.ONE, BHV.ZERO) +
        # maj3_inv(majority3, Invert) +
        boolean_algebra(And, Or, Invert, BHV.ZERO, BHV.ONE) +
        extra)


props = bhv_props()
props = [p.swap() if p.r.size() > p.l.size() else p for p in props]
props = list({stable_hashcode(p): p for p in props}.values())
props.sort(key=lambda p: (p.r.size() - p.l.size()) or (len(p.r.vars()) - len(p.l.vars())))

sep = next(filter(lambda p: (p.r.size() - p.l.size()) >= 0, props))
props = props[:props.index(sep)]


def search(e):
    fvars = set(e.vars())
    for prop in props:
        bindings = unify(e, prop.l)
        if bindings and not set(bindings) & fvars:
            print("applying", prop.name, "to", e.show())
            return prop.r.substitute(bindings)
    return None


def rec(x):
    cs = x.children()

    for i in range(len(cs)):
        res_c = search(cs[i])
        if res_c is None:
            continue
        else:
            cs[i] = res_c
            return x.reconstruct(*cs)

    for i in range(len(cs)):
        res_c = rec(cs[i])
        if res_c is None:
            continue
        else:
            cs[i] = res_c
            return x.reconstruct(*cs)

def greedy(initial, iters=100):
    e = initial

    for i in range(iters):
        res_e = search(e)
        if res_e is not None:
            e = res_e
        else:
            rec_e = rec(e)
            if rec_e is not None:
                e = rec_e
            else:
                print("early", i)
                return e

    return e


if __name__ == '__main__':
    x1 = Var("x1")

    teste1 = (~(~(x1))) ^ (~x1 & BHV.rand())
    print(greedy(teste1).show())

    teste2 = ~(x1 ^ BHV.ZERO)
    print(greedy(teste2).show())
