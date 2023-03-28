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


class BottomIdentity:
    def test_bottom_identity(self):
        for x in self.xs:
            self.assertEqual(self.op(x, self.bot), x)


class BoundedJoinLattice(JoinLattice, BottomIdentity):
    def test_top_absorbing(self):
        for x in self.xs:
            self.assertEqual(self.op(x, self.top), self.top)


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
class TestXOR(CommAssoc, SelfInv, BottomIdentity):
    op = lambda _, x, y: x ^ y
    bot = property(lambda self: self.impl.ZERO)


if __name__ == '__main__':
    unittest.main()

