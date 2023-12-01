import unittest

from bhv.symbolic import Var, SymbolicBHV, PermVar, Majority, Zero
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

        # perm invert merges
        self.assertEqual(x, p((~p)(x)).reduce())
        self.assertEqual(x, (~p)(p(x)).reduce())
        self.assertEqual(x, (~p)((~~p)(x)).reduce())

        self.assertIsNone(p(p(x)).reduce())
        self.assertIsNone(p(q(x)).reduce())
        self.assertIsNone(p((~~p)(x)).reduce())

    def test_simplify(self):
        x = Var("x")
        p = PermVar("p")

        # inverse of invert perm is perm
        self.assertEqual(p(p(x)), p((~~p)(x)).simplify())
        self.assertEqual(p(x), (~~p)(x).simplify())
        self.assertEqual(x, (~~~p)(p(x)).simplify())
        self.assertEqual(p(p(x)), (~~~~p)(p(x)).simplify())

    def test_distribute(self):
        # perm
        a = Var("a")
        b = Var("b")
        x = Var("x")
        y = Var("y")
        z = Var("z")
        p = PermVar("p")
        q = PermVar("q")

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

        ez = p(Zero())
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

        ez = Zero() ^ x
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

        ezero = p(Zero())
        self.assertEqual(ezero, ezero.shred())


if __name__ == '__main__':
    unittest.main()
