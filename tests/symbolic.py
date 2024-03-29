import unittest

from fractions import Fraction
from bhv.symbolic import Var, SymbolicBHV, PermVar, Majority, SymbolicPermutation, Representative
from bhv.shared import stable_hashcode


class TestReduction(unittest.TestCase):
    def test_optimal_sharing(self):
        x = Var("x")
        y = Var("y")

        se = (x ^ y)

        e = se | ~se
        e_ = (x ^ y) | ~(x ^ y)

        self.assertEqual(stable_hashcode(e), stable_hashcode(se | ~se))
        self.assertNotEqual(stable_hashcode(e), e_)
        self.assertEqual(stable_hashcode(e), stable_hashcode(e_.optimal_sharing()))

    def test_reduce(self):
        x = Var("x")
        y = Var("y")
        p = PermVar("p")
        q = PermVar("q")

        e = ~(~((x & y) ^ (x & (y & ~y))))

        self.assertEqual(e.simplify(), (x & y))

        # perm merges with its inverse
        self.assertEqual(x, p((~p)(x)).reduce())
        self.assertEqual(x, (~p)(p(x)).reduce())
        self.assertEqual(x, (~p)((~~p)(x)).reduce())

        self.assertIsNone(p(p(x)).reduce())
        self.assertIsNone(p(q(x)).reduce())
        self.assertIsNone(p(~q(x)).reduce())
        self.assertIsNone((~p)(q(x)).reduce())
        self.assertIsNone(p((~~p)(x)).reduce())  # reduce only rewrites outer operation

        # double invert = perm
        self.assertEqual(p, (~~p).reduce())

        # identity perm
        p_id = SymbolicPermutation.IDENTITY
        self.assertEqual(p(x), p_id(p(x)).reduce())

        # perm on 0 (resp. 1) is 0 (resp. 1)
        zero = SymbolicBHV.ZERO
        one = SymbolicBHV.ONE
        self.assertEqual(zero, p(zero).reduce())
        self.assertEqual(one, p(one).reduce())

    def test_simplify(self):
        x = Var("x")
        p = PermVar("p")

        # inverse of invert perm is perm
        self.assertEqual(p(p(x)), p((~~p)(x)).simplify())
        self.assertEqual(p(x), (~~p)(x).simplify())
        self.assertEqual(x, (~~~p)(p(x)).simplify())
        self.assertEqual(p(p(x)), (~~~~p)(p(x)).simplify())

        # identity perm
        p_id = SymbolicPermutation.IDENTITY
        self.assertEqual(p(x), p(p_id(x)).simplify())

    def test_distribute(self):
        a = Var("a")
        b = Var("b")
        x = Var("x")
        y = Var("y")
        z = Var("z")
        p = PermVar("p")
        q = PermVar("q")

        zero = SymbolicBHV.ZERO

        # perm
        ex = p((x ^ y) ^ z)
        ex_ = p(x ^ y) ^ p(z)
        self.assertEqual(ex_, ex.distribute())

        em = p(Majority([x, y, z]))
        em_ = Majority([p(x), p(y), p(z)])
        self.assertEqual(em_, em.distribute())

        emx = p(Majority([x, y, (x ^ z)]))
        emx_ = Majority([p(x), p(y), p(x ^ z)])
        self.assertEqual(emx_, emx.distribute())

        ev = p(x)
        self.assertIsNone(ev.distribute())

        ep = p(q(x))
        self.assertIsNone(ep.distribute())

        ez = p(zero)
        self.assertIsNone(ez.distribute())

        # Xor
        em = Majority([x, y]) ^ z
        em_ = Majority([x ^ z, y ^ z])
        self.assertEqual(em_, em.distribute())

        em2 = z ^ Majority([x, y])
        em2_ = Majority([z ^ x, z ^ y])
        self.assertEqual(em2_, em2.distribute())

        emm = Majority([a, b]) ^ Majority([x, y])
        emm_ = Majority([a ^ Majority([x, y]), b ^ Majority([x, y])])
        self.assertEqual(emm_, emm.distribute())

        f = p(x) ^ p(y)
        self.assertIsNone(f.distribute())

        ev = x ^ y
        self.assertIsNone(ev.distribute())

        ez = zero ^ x
        self.assertIsNone(ez.distribute())

    def test_shred(self):
        a = Var("a")
        b = Var("b")
        x = Var("x")
        y = Var("y")
        z = Var("z")
        p = PermVar("p")
        q = PermVar("q")

        ep = p(q(x ^ (y ^ z)))
        ep_ = (p(q(x))) ^ (p(q(y)) ^ p(q(z)))
        self.assertEqual(ep_, ep.shred())

        epq = Majority([p(x ^ z), q(y ^ z)])
        epq_ = Majority([p(x) ^ p(z), q(y) ^ q(z)])
        self.assertEqual(epq_, epq.shred())

        ex = x ^ (y ^ Majority([a, b]))
        ex_ = Majority([(x ^ (y ^ a)), (x ^ (y ^ b))])
        self.assertEqual(ex_, ex.shred())

        ezero = p(SymbolicBHV.ZERO)
        self.assertEqual(ezero, ezero.shred())


class TestRepresentative(unittest.TestCase):
    def test_expected_error(self):
        a = Var("a")
        b = Var("b")
        x = Var("x")
        y = Var("y")
        z = Var("z")

        def R(*args): return Representative([*args])

        # representative of vector and itself
        self.assertEqual({}, R().expected_errors())
        self.assertEqual({'a': 0}, R(a).expected_errors())
        self.assertEqual({'a': 0}, R(a, a).expected_errors())
        self.assertEqual({'a': 0}, R(a, a, a).expected_errors())

        # test zero and one
        self.assertEqual({'b': Fraction(1, 4), 'a': Fraction(1, 4)}, R(b, a).expected_errors({'b': 0}))
        self.assertEqual({'a': Fraction(1, 4)}, R(SymbolicBHV.ZERO, a).expected_errors())
        self.assertEqual({'a': Fraction(1, 4)}, R(SymbolicBHV.ONE, a).expected_errors())

        # representative of independent random hypervectors
        self.assertEqual({'x': Fraction(1, 4), 'y': Fraction(1, 4)}, R(x, y).expected_errors())
        self.assertEqual({v.name: Fraction(1, 3) for v in [x, y, z]}, R(x, y, z).expected_errors())
        self.assertEqual({v.name: Fraction(3, 8) for v in [a, x, y, z]}, R(a, x, y, z).expected_errors())
        self.assertEqual({v.name: Fraction(2, 5) for v in [a, b, x, y, z]}, R(a, b, x, y, z).expected_errors())

        # expected error of a non occurring variable
        self.assertEqual(Fraction(1, 2), R().expected_error('a', dict()))  # assume that R() == RAND
        self.assertEqual(Fraction(1, 2), R(x, y, z).expected_error('a', dict()))

        self.assertEqual({'x': Fraction(7, 18), 'y': Fraction(7, 18), 'a': Fraction(4, 18)},
                         R(R(x, a, y), R(x, a, y), a).expected_errors())

        self.assertEqual({'a': Fraction(1, 6), 'b': Fraction(1, 3)}, R(a, a, b).expected_errors())

        # print(R(Majority([x, y, z]), a).expected_errors())


if __name__ == '__main__':
    unittest.main()
