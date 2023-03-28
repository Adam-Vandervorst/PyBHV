from time import monotonic
from itertools import product, groupby

from bhv.abstract import AbstractBHV
from bhv.np import NumPyBoolBHV, NumPyPacked8BHV, NumPyPacked64BHV
from bhv.pytorch import TorchBoolBHV, TorchPackedBHV


def associative(m): return lambda x, y, z: m(m(x, y), z) == m(x, m(y, z))
def commutative(m): return lambda x, y: m(x, y) == m(y, x)
def idempotent(m): return lambda x: m(x, x) == x
def self_inverse(m, u): return lambda x: m(x, x) == u
def self_inverse_op(f): return lambda x: f(f(x)) == x
def equals(f, a, b): return lambda: f(a) == b
def right_unit(m, u): return lambda x: m(x, u) == x
def right_absorbing(m, z): return lambda x: m(x, z) == z
def absorptive(m, w): return lambda x, y: m(x, w(x, y)) == x
def distributive(m, w): return lambda x, y, z: m(x, w(y, z)) == w(m(x, y), m(x, z))
def demorgan(f, m, w): return lambda x, y: f(m(x, y)) == w(f(x), f(y))
def drop_left(f, m): return lambda x, y: f(m(x, y)) == m(f(x), y)
def expand_single_inner(f, k, m, w): return lambda x, y: True  # k(x, y) == m(w(x, f(y)), w(f(x), y))
def expand_single_outer(f, k, m, w): return lambda x, y: True  # k(x, y) == m(w(x, y), f(m(x, y)))


def lattice(join): return [associative(join), commutative(join), idempotent(join)]
def bounded_lattice(join, top, bot): return lattice(join) + [right_unit(join, bot), right_absorbing(join, top)]
def xor_props(xor_, bot): return [associative(xor_), commutative(xor_), self_inverse(xor_, bot), right_unit(xor_, bot)]
def not_props(not_, bot, top): return [self_inverse_op(not_), equals(not_, top, bot), equals(not_, bot, top)]
def or_and_props(or_, and_): return [distributive(or_, and_), distributive(and_, or_), absorptive(or_, and_), absorptive(and_, or_)]


def bhv_props(impl: AbstractBHV):
    extra = [
        demorgan(impl.__invert__, impl.__or__, impl.__and__),
        demorgan(impl.__invert__, impl.__and__, impl.__or__),
        drop_left(impl.__invert__, impl.__xor__),
        expand_single_inner(impl.__invert__, impl.__xor__, impl.__and__, impl.__or__),
        expand_single_outer(impl.__invert__, impl.__xor__, impl.__or__, impl.__and__),
        distributive(impl.__and__, impl.__xor__)
    ]

    return (
        bounded_lattice(impl.__and__, impl.ZERO, impl.ONE) +
        bounded_lattice(impl.__or__, impl.ONE, impl.ZERO) +
        xor_props(impl.__xor__, impl.ZERO) +
        not_props(impl.__invert__, impl.ZERO, impl.ONE) +
        or_and_props(impl.__or__, impl.__and__) +
        extra)


def run_for(impl: AbstractBHV, ts):
    extrema = [impl.ZERO, impl.ONE]
    shared = extrema + [impl.rand() for _ in range(3)]

    argts = {k: list(vs) for k, vs in groupby(ts, lambda f: f.__code__.co_argcount)}

    def rec(args, depth):
        if depth in argts:
            for tn in argts[depth]:
                assert tn(*args), f"property {tn.__qualname__} failed on {args} using implementation {impl.__name__}"
            for x in shared:
                rec(args + [x], depth + 1)

    rec([], 0)


def run():
    all_implementations = [NumPyBoolBHV, NumPyPacked8BHV, NumPyPacked64BHV, TorchBoolBHV, TorchPackedBHV]

    for impl in all_implementations:
        print(f"Testing {impl.__name__}...")
        t0 = monotonic()
        run_for(impl, bhv_props(impl))
        t = monotonic() - t0
        print(f"took ({t*1000:.3} ms)")


if __name__ == '__main__':
    run()
