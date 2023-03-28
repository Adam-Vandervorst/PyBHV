import unittest
from time import monotonic
from itertools import product

from bhv.abstract import AbstractBHV
from bhv.np import NumPyBoolBHV, NumPyPacked8BHV, NumPyPacked64BHV
from bhv.pytorch import TorchBoolBHV, TorchPackedBHV


ALL = [NumPyBoolBHV, NumPyPacked8BHV, NumPyPacked64BHV, TorchBoolBHV, TorchPackedBHV]


def for_implementations(impls):
    def wrap(law):
        for impl in impls:
            name = f"{law.__name__}_{impl.__name__}"
            globals()[name] = type(name, (TestImpl, law, unittest.TestCase), dict(impl=impl))
    return wrap


class TestImpl:
    impl: AbstractBHV

    @classmethod
    def setUpClass(cls):
        print(f"Testing {cls.__name__}...")
        cls._t0 = monotonic()

        cls.extrema = [cls.impl.ZERO, cls.impl.ONE]
        cls.shared = cls.extrema + [cls.impl.rand() for _ in range(3)]
        cls.xs = cls.shared + [cls.impl.rand() for _ in range(5)]
        cls.ys = cls.shared + [cls.impl.rand() for _ in range(2)]
        cls.zs = cls.shared + [cls.impl.rand() for _ in range(2)]

    @classmethod
    def tearDownClass(cls):
        t = monotonic() - cls._t0
        print(f"Done testing {cls.__name__} ({t*1000:.3} ms)")


class Comm:
    def test_associative(self):
        for x, y, z in product(self.xs, self.ys, self.zs):
            self.assertEqual(self.op(self.op(x, y), z), self.op(x, self.op(y, z)))


class Assoc:
    def test_commutative(self):
        for x, y in product(self.xs, self.ys):
            self.assertEqual(self.op(x, y), self.op(y, x))


class JoinLattice(Comm, Assoc):
    def test_idempotent(self):
        for x in self.xs:
            self.assertEqual(self.op(x, x), x)


class SelfInv:
    def test_self_inverse(self):
        for x in self.xs:
            self.assertEqual(self.op(x, x), self.bot)


class SelfInvOp:
    def test_self_inverse_op(self):
        for x in self.xs:
            self.assertEqual(self.op(self.op(x)), x)


class Inverse:
    def test_extrema(self):
        self.assertEqual(self.op(self.top), self.bot)
        self.assertEqual(self.op(self.bot), self.top)


class BottomIdentity:
    def test_bottom_identity(self):
        for x in self.xs:
            self.assertEqual(self.op(x, self.bot), x)


class BoundedJoinLattice(JoinLattice, BottomIdentity):
    def test_top_absorbing(self):
        for x in self.xs:
            self.assertEqual(self.op(x, self.top), self.top)


class Absorptive:
    def test_absorptive(self):
        for x, y in product(self.xs, self.ys):
            self.assertEqual(self.op1(x, self.op2(x, y)), x)
            self.assertEqual(self.op2(x, self.op1(x, y)), x)


class Distributive:
    def test_distributive(self):
        for x, y, z in product(self.xs, self.ys, self.zs):
            self.assertEqual(self.op1(x, self.op2(y, z)), self.op2(self.op1(x, y), self.op1(x, z)))
            self.assertEqual(self.op2(x, self.op1(y, z)), self.op1(self.op2(x, y), self.op2(x, z)))


class DeMorgan:
    def test_demorgan(self):
        for x, y in product(self.xs, self.ys):
            self.assertEqual(self.op(self.op1(x, y)), self.op2(self.op(x), self.op(y)))
            self.assertEqual(self.op(self.op2(x, y)), self.op1(self.op(x), self.op(y)))


class DropSingle:
    def test_drop_single(self):
        for x, y in product(self.xs, self.ys):
            self.assertEqual(self.op(self.op1(x, y)), self.op1(self.op(x), y))
            self.assertEqual(self.op(self.op1(x, y)), self.op1(x, self.op(y)))


class ExpandSingle:
    def test_expand_single_inner(self):
        for x, y in product(self.xs, self.ys):
            self.assertEqual(self.op3(x, y), self.op1(self.op2(x, self.op(y)), self.op2(self.op(x), y)))

    def test_expand_single_outer(self):
        for x, y in product(self.xs, self.ys):
            self.assertEqual(self.op3(x, y), self.op2(self.op1(x, y), self.op(self.op2(x, y))))


@for_implementations(ALL)
class TestAND(BoundedJoinLattice):
    op = lambda _, x, y: x & y
    top = property(lambda self: self.impl.ZERO)
    bot = property(lambda self: self.impl.ONE)


@for_implementations(ALL)
class TestOR(BoundedJoinLattice):
    op = lambda _, x, y: x | y
    top = property(lambda self: self.impl.ONE)
    bot = property(lambda self: self.impl.ZERO)


@for_implementations(ALL)
class TestXOR(Comm, Assoc, SelfInv, BottomIdentity):
    op = lambda _, x, y: x ^ y
    bot = property(lambda self: self.impl.ZERO)


@for_implementations(ALL)
class TestNOT(SelfInvOp, Inverse):
    op = lambda _, x: ~x
    bot = property(lambda self: self.impl.ZERO)
    top = property(lambda self: self.impl.ONE)


@for_implementations(ALL)
class TestORAND(Absorptive, Distributive):
    op1 = lambda _, x, y: x | y
    op2 = lambda _, x, y: x & y


@for_implementations(ALL)
class TestORANDNOT(DeMorgan):
    op = lambda _, x: ~x
    op1 = lambda _, x, y: x | y
    op2 = lambda _, x, y: x & y


@for_implementations(ALL)
class TestXORNOT(DropSingle):
    op = lambda _, x: ~x
    op1 = lambda _, x, y: x ^ y


@for_implementations(ALL)
class TestXORORANDNOT(ExpandSingle):
    op = lambda _, x: ~x
    op1 = lambda _, x, y: x | y
    op2 = lambda _, x, y: x & y
    op3 = lambda _, x, y: x ^ y


if __name__ == '__main__':
    unittest.main()
