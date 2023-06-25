import unittest

from bhv.symbolic import Var, SymbolicBHV
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

        e = ~(~((x & y) ^ (x & (y & ~y))))

        self.assertEqual(e.simplify(), (x & y))


if __name__ == '__main__':
    unittest.main()
