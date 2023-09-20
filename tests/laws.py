from time import monotonic
from itertools import product, groupby

from bhv.abstract import AbstractBHV, DIMENSION
from bhv.native import NativePackedBHV
from bhv.np import NumPyBoolBHV, NumPyPacked8BHV, NumPyPacked64BHV
# from bhv.pytorch import TorchBoolBHV, TorchPackedBHV
from bhv.vanilla import VanillaBHV


def associative(m): return lambda x, y, z: m(m(x, y), z) == m(x, m(y, z))
def associative3(m): return lambda x, y, z, u: m(x, u, m(y, u, z)) == m(z, u, m(y, u, x))
def commutative(m): return lambda x, y: m(x, y) == m(y, x)
def commutative3(m): return lambda x, y, z: m(x, y, z) == m(y, x, z) == m(z, y, x)
def idempotent(m): return lambda x: m(x, x) == x
def self_inverse(m, u): return lambda x: m(x, x) == u
def self_inverse_op(f): return lambda x: f(f(x)) == x
def equals(f, a, b): return lambda: f(a) == b
def right_unit(m, u): return lambda x: m(x, u) == x
def right_absorbing(m, z): return lambda x: m(x, z) == z
def absorptive(m, w): return lambda x, y: m(x, w(x, y)) == x
def distributive(m, w): return lambda x, y, z: m(x, w(y, z)) == w(m(x, y), m(x, z))
def distributive3(m, w): return lambda x, y, z, u, v: m(x, y, w(u, v, z)) == w(m(x, y, u), m(x, y, v), z)
def demorgan(f, m, w): return lambda x, y: f(m(x, y)) == w(f(x), f(y))
def drop_left(f, m): return lambda x, y: f(m(x, y)) == m(f(x), y)
def propagate(f, m): return lambda x: f(m(x)) == m(f(x))
def propagate2(f, m): return lambda x, y: f(m(x, y)) == m(f(x), f(y))
def propagate3(f, m): return lambda x, y, z: f(m(x, y, z)) == m(f(x), f(y), f(z))
def expand_single_inner(f, k, m, w): return lambda x, y: k(x, y) == m(w(x, f(y)), w(f(x), y))
def expand_single_outer(f, k, m, w): return lambda x, y: k(x, y) == m(w(x, y), f(m(x, y)))
def determinism(f): return lambda x: f(x) == f(x)
def extensionality(f, g): return lambda x: f(x) == g(x)
def extensionality2(f, g): return lambda x, y: f(x, y) == g(x, y)
def extensionality3(f, g): return lambda x, y, z: f(x, y, z) == g(x, y, z)
def extensionality4(f, g): return lambda x, y, z, u: f(x, y, z, u) == g(x, y, z, u)
def extensionality5(f, g): return lambda x, y, z, u, v: f(x, y, z, u, v) == g(x, y, z, u, v)
def extensionality6(f, g): return lambda x, y, z, u, v, w: f(x, y, z, u, v, w) == g(x, y, z, u, v, w)
def extensionality7(f, g): return lambda x, y, z, u, v, w, r: f(x, y, z, u, v, w, r) == g(x, y, z, u, v, w, r)
def extensionality9(f, g): return lambda x, y, z, u, v, w, r, i, j: f(x, y, z, u, v, w, r, i, j) == g(x, y, z, u, v, w, r, i, j)
def extensionalityN(f, g): return lambda xs: f(xs) == g(xs)
def encode_decode(enc, dec): return lambda x: x == dec(enc(x))
def invariant_under(f, p): return lambda x: f(x) == f(p(x))
def invariant_under2(f, p): return lambda x, y: f(x, y) == f(p(x), p(y))
def identity(f): return lambda x: f(x) == x
def invariant2(m): return lambda x, y: m(x, y) == y
def invariant3(m): return lambda x, y, z: m(x, y, z) == z
def invariant3_2(m): return lambda x, y, z: m(x, y, z) == y == z
def complement(m, f, z): return lambda x: m(x, f(x)) == z

def transport(law, l, r): return lambda f, g: law(f, lambda x: r(g(l(x))))
def transport2(law, l, r): return lambda f, g: law(f, lambda x, y: r(g(l(x), l(y))))
def transport3(law, l, r): return lambda f, g: law(f, lambda x, y, z: r(g(l(x), l(y), l(z))))
def assume2(law, p): return lambda f: lambda x, y: True if p(x, y) else law(f)(x, y)
def assume3(law, p): return lambda f: lambda x, y, z: True if p(x, y, z) else law(f)(x, y, z)

def lattice(join): return [associative(join), commutative(join), idempotent(join)]
def bounded_lattice(join, top, bot): return lattice(join) + [right_unit(join, bot), right_absorbing(join, top)]
def xor_props(xor_, bot): return [associative(xor_), commutative(xor_), self_inverse(xor_, bot), right_unit(xor_, bot)]
def not_props(not_, bot, top): return [self_inverse_op(not_), equals(not_, top, bot), equals(not_, bot, top)]
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
        commutative3(maj3),
        assume3(invariant3_2(lambda x, y, z: maj3(z, y, x)), lambda x, y, z: x == y),
        assume3(invariant3(maj3), lambda x, y, z: x == inv(y)),
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
def permute_props(permute):
    π, τ, σ = 42, 13, 39
    Π, Τ, Σ = lambda x: permute(x, π), lambda x: permute(x, τ), lambda x: permute(x, σ)
    Πinv, Τinv, Σinv = lambda x: permute(x, -π), lambda x: permute(x, -τ), lambda x: permute(x, -σ)
    return [
        extensionality(lambda x: Π(Τ(x)), lambda x: permute(x, (τ, π))),
        extensionality(lambda x: Π(Τ(x)), lambda x: permute(x, (0, τ, 0, π, 0))),
        extensionality(lambda x: Π(Σ(Τ(x))), lambda x: permute(x, (τ, σ, π))),
        identity(lambda x: permute(x, 0)),
        identity(lambda x: Πinv(Π(x))),
        identity(lambda x: Τinv(Τ(x))),
        identity(lambda x: Σ(Σinv(x))),
    ]


def bhv_props(impl: AbstractBHV):
    Π, Τ, Σ = lambda x: impl.permute(x, 42), lambda x: impl.permute(x, 13), lambda x: impl.permute(x, -39)
    extra = [
        demorgan(impl.__invert__, impl.__or__, impl.__and__),
        demorgan(impl.__invert__, impl.__and__, impl.__or__),
        drop_left(impl.__invert__, impl.__xor__),
        expand_single_inner(impl.__invert__, impl.__xor__, impl.__or__, impl.__and__),
        expand_single_outer(impl.__invert__, impl.__xor__, impl.__and__, impl.__or__),
        distributive(impl.__and__, impl.__xor__),
        propagate(Π, impl.__invert__),
        propagate2(Π, impl.__xor__),
        propagate2(Π, impl.__and__),
        propagate2(Π, impl.__or__),
        propagate3(Π, impl.majority3),

        complement(impl.__xor__, impl.__invert__, impl.ONE),
        invariant_under2(impl.__xor__, impl.__invert__),
        lambda a, b: a ^ (a & b) == a & ~b,
        lambda a, b: a ^ (~a & b) == a | b
    ]

    return (
        bounded_lattice(impl.__and__, impl.ZERO, impl.ONE) +
        bounded_lattice(impl.__or__, impl.ONE, impl.ZERO) +
        xor_props(impl.__xor__, impl.ZERO) +
        not_props(impl.__invert__, impl.ZERO, impl.ONE) +
        or_and_props(impl.__or__, impl.__and__) +
        gf2(impl.__xor__, impl.__and__, impl.ONE, impl.ZERO) +
        maj3_inv(impl.majority3, impl.__invert__) +
        boolean_algebra(impl.__and__, impl.__or__, impl.__invert__, impl.ZERO, impl.ONE) +
        permute_props(impl.permute) +
        bhv_conv_metrics(Π) +
        extra)


def bhv_conv_ops(dom: AbstractBHV, codom: AbstractBHV, l, r):
    return [
        transport(extensionality, l, r)(dom.__invert__, codom.__invert__),
        transport2(extensionality2, l, r)(dom.__or__, codom.__or__),
        transport2(extensionality2, l, r)(dom.__and__, codom.__and__),
        transport2(extensionality2, l, r)(dom.__xor__, codom.__xor__),
        transport3(extensionality3, l, r)(dom.select, codom.select),
    ]


def bhv_conv_metrics(under):
    return [
        invariant_under(lambda x: x.active(), under),
        invariant_under(lambda x: x.active_fraction(), under),

        invariant_under2(lambda x, y: x.hamming(y), under),
        assume2(invariant_under2(lambda x, y: x.cosine(y), under), lambda x, y: x.active() != 0 and y.active() != 0),
        assume2(invariant_under2(lambda x, y: x.jaccard(y), under), lambda x, y: x.active() != 0 or y.active() != 0),
        invariant_under2(lambda x, y: x.std_apart(y), under),
        invariant_under2(lambda x, y: x.related(y), under),
        invariant_under2(lambda x, y: x.bit_error_rate(y), under),
        invariant_under2(lambda x, y: x.mutual_information(y), under),
    ]


def run_for(impl: AbstractBHV, ts):
    argts = {}
    for t in ts:
        arity = t.__code__.co_argcount
        argts.setdefault(arity, []).append(t)
    max_depth = max(argts.keys())

    extrema = [impl.ZERO, impl.ONE]
    shared = extrema + impl.nrand(3)
    collections = [shared + impl.nrand(1) for d in range(max_depth + 1)]

    def rec(args, depth):
        for tn in argts.get(depth, []):
            assert tn(*args), f"property {tn.__qualname__} failed on {args} using implementation {impl.__name__}"
        if depth <= max_depth:
            for x in collections[depth]:
                rec(args + (x,), depth + 1)

    rec((), 0)


def run():
    # all_implementations = [VanillaBHV, NumPyBoolBHV, NumPyPacked8BHV, NumPyPacked64BHV, TorchBoolBHV, TorchPackedBHV, NativePackedBHV]
    all_implementations = [VanillaBHV, NumPyBoolBHV, NumPyPacked8BHV, NumPyPacked64BHV, NativePackedBHV]

    for impl in all_implementations:
        print(f"Testing {impl.__name__}...")
        t0 = monotonic()
        run_for(impl, bhv_props(impl))
        t = monotonic() - t0
        print(f"took ({t:.3} s)")

    print(f"Testing packing and unpacking NumPyBoolBHV...")
    run_for(NumPyBoolBHV, [encode_decode(NumPyBoolBHV.pack64, NumPyPacked64BHV.unpack)])
    print(f"Testing packing and unpacking NumPyPacked64BHV...")
    run_for(NumPyPacked64BHV, [encode_decode(NumPyPacked64BHV.unpack, NumPyBoolBHV.pack64)])
    print("Testing operators equal between NumPyBoolBHV and NumPyPacked64BHV")
    run_for(NumPyBoolBHV, bhv_conv_ops(NumPyBoolBHV, NumPyPacked64BHV, NumPyBoolBHV.pack64, NumPyPacked64BHV.unpack))
    print("Testing metrics invariant under pack64")
    run_for(NumPyBoolBHV, bhv_conv_metrics(NumPyBoolBHV.pack64))
    # run_for(TorchBoolBHV, bhv_conv_metrics(TorchBoolBHV.pack))
    print("Testing large unbalanced majority")
    rs = [~v for v in NumPyPacked64BHV.nrand2(1001, 3)]
    rs_ = [r.unpack() for r in rs]
    assert NumPyPacked64BHV.majority(rs) == NumPyBoolBHV.majority(rs_).pack64()
    # rs = [~v for v in TorchPackedBHV.nrand2(1001, 3)]
    # rs_ = [r.unpack() for r in rs]
    # assert TorchPackedBHV.majority(rs) == TorchBoolBHV.majority(rs_).pack()
    print("Testing bits and bytes")
    for impl in all_implementations:
        r = impl.rand()
        rb = r.to_bytes()
        rbits = list(r.bits())
        rstr = r.bitstring()
        for impl_ in all_implementations:
            assert impl_.from_bytes(rb).to_bytes() == rb, f"{impl}, {impl_}"
            assert list(impl_.from_bitstream(rbits).bits()) == rbits, f"{impl}, {impl_}"
            assert impl_.from_bytes(rb).bitstring() == rstr, f"{impl}, {impl_}"
            assert impl_.from_bitstring(rstr).bitstring() == rstr, f"{impl}, {impl_}"

    print("Testing word-level roll equivalence")
    rs_np = NumPyPacked64BHV.nrand(10)
    rs_native = [NativePackedBHV.from_bytes(r.to_bytes()) for r in rs_np]

    word_rot_pos_np = [r.roll_words(12) for r in rs_np]
    word_rot_pos_native = [r.roll_words(12) for r in rs_native]
    assert [r == NumPyPacked64BHV.from_bytes(r_.to_bytes()) for r, r_ in zip(word_rot_pos_np, word_rot_pos_native)]
    word_rot_neg_np = [r.roll_words(-12) for r in rs_np]
    word_rot_neg_native = [r.roll_words(-12) for r in rs_native]
    assert [r == NumPyPacked64BHV.from_bytes(r_.to_bytes()) for r, r_ in zip(word_rot_neg_np, word_rot_neg_native)]
    word_bit_rot_pos_np = [r.roll_word_bits(12) for r in rs_np]
    word_bit_rot_pos_native = [r.roll_word_bits(12) for r in rs_native]
    assert [r == NumPyPacked64BHV.from_bytes(r_.to_bytes()) for r, r_ in zip(word_bit_rot_pos_np, word_bit_rot_pos_native)]
    word_bit_rot_neg_np = [r.roll_word_bits(-12) for r in rs_np]
    word_bit_rot_neg_native = [r.roll_word_bits(-12) for r in rs_native]
    assert [r == NumPyPacked64BHV.from_bytes(r_.to_bytes()) for r, r_ in zip(word_bit_rot_neg_np, word_bit_rot_neg_native)]

    print("Testing bit-level roll equivalence")
    rs_np_bool = NumPyBoolBHV.nrand(10)
    rs_np_packed = [NumPyPacked64BHV.from_bytes(r.to_bytes()) for r in rs_np_bool]
    rs_vanilla = [VanillaBHV.from_bytes(r.to_bytes()) for r in rs_np_bool]
    rs_native = [NativePackedBHV.from_bytes(r.to_bytes()) for r in rs_np_bool]

    bit_rot_neg_bool = [r.roll_bits(-112) for r in rs_np_bool]
    bit_rot_neg_packed = [r.roll_bits(-112) for r in rs_np_packed]
    bit_rot_neg_vanilla = [r.roll_bits(-112) for r in rs_vanilla]
    bit_rot_neg_native = [r.roll_bits(-112) for r in rs_native]
    assert [r == NumPyBoolBHV.from_bytes(r_.to_bytes()) for r, r_ in zip(bit_rot_neg_bool, bit_rot_neg_vanilla)]
    assert [r == NumPyBoolBHV.from_bytes(r_.to_bytes()) for r, r_ in zip(bit_rot_neg_bool, bit_rot_neg_packed)]
    assert [r == NumPyBoolBHV.from_bytes(r_.to_bytes()) for r, r_ in zip(bit_rot_neg_bool, bit_rot_neg_native)]


    print("Testing NumPyPacked64BHV majority equivalence")
    run_for(NumPyPacked64BHV, [
        extensionality3(NumPyPacked64BHV._majority3, lambda x, y, z: NumPyPacked64BHV._majority_via_unpacked([x, y, z])),
        extensionality5(NumPyPacked64BHV._majority5, lambda x, y, z, u, v: NumPyPacked64BHV._majority_via_unpacked([x, y, z, u, v])),
        extensionality5(NumPyPacked64BHV._majority5_via_3, NumPyPacked64BHV._majority5),
        # this is slow, obviously
        # extensionality7(NumPyPacked64BHV._majority7_via_3, lambda x, y, z, u, v, w, r: NumPyPacked64BHV._majority_via_unpacked([x, y, z, u, v, w, r])),
        # extensionality7(NumPyPacked64BHV._majority7_via_ite, lambda x, y, z, u, v, w, r: NumPyPacked64BHV._majority_via_unpacked([x, y, z, u, v, w, r])),
        # extensionality7(NumPyPacked64BHV._majority7_via_3, lambda x, y, z, u, v, w, r: NumPyPacked64BHV._majority_via_unpacked([x, y, z, u, v, w, r])),
        # extensionality9(NumPyPacked64BHV._majority9_via_3, lambda x, y, z, u, v, w, r, i, j: NumPyPacked64BHV._majority_via_unpacked([x, y, z, u, v, w, r, i, j])),
    ])

    for s in range(3, 55, 2):
        rs = NumPyPacked64BHV.nrand(s)
        assert NumPyPacked64BHV.majority(rs) == NumPyPacked64BHV._majority_via_unpacked(rs), f"mismatch for size {s}"

    print("Testing NumPyBoolBHV majority threshold equivalence")
    for s in range(3, 55, 2):
        rs = NumPyBoolBHV.nrand(s)
        assert NumPyBoolBHV.majority(rs) == NumPyBoolBHV._direct_majority(rs), f"mismatch for size {s}"



if __name__ == '__main__':
    run()
