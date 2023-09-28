from bhv.slice import Slice
from bhv.np import NumPyPacked64BHV as BHV

import unittest


class TestLevels(unittest.TestCase):
    @staticmethod
    def vs(on, off):
        return [Slice.ONE for _ in range(on)] + [Slice.ZERO for _ in range(off)]

    def test_basic(self):
        vs60 = self.vs(6, 0)
        vs51 = self.vs(5, 1)
        vs42 = self.vs(4, 2)
        vs24 = self.vs(2, 4)
        vs15 = self.vs(1, 5)
        vs06 = self.vs(0, 6)

        vs43 = self.vs(4, 3)
        vs33 = self.vs(3, 3)

        self.assertEqual(Slice.threshold(vs42, 3), Slice.ONE)
        self.assertEqual(Slice.threshold(vs42, 5), Slice.ZERO)

        self.assertEqual(Slice.window(vs15, 0, 4), Slice.ONE)
        self.assertEqual(Slice.window(vs15, 2, 4), Slice.ZERO)
        self.assertEqual(Slice.window(vs42, 2, 4), Slice.ONE)
        self.assertEqual(Slice.window(vs24, 2, 4), Slice.ONE)
        self.assertEqual(Slice.window(vs51, 2, 4), Slice.ZERO)
        self.assertEqual(Slice.window(vs51, 2, 6), Slice.ONE)

        self.assertEqual(Slice.agreement(vs60, 1.), Slice.ONE)
        self.assertEqual(Slice.agreement(vs06, 1.), Slice.ONE)
        self.assertEqual(Slice.agreement(vs51, 1.), Slice.ZERO)
        self.assertEqual(Slice.agreement(vs15, 1.), Slice.ZERO)
        self.assertEqual(Slice.agreement(vs15, 3/5), Slice.ONE)
        self.assertEqual(Slice.agreement(vs24, 3/5), Slice.ONE)
        self.assertEqual(Slice.agreement(vs43, 3/5), Slice.ZERO)
        self.assertEqual(Slice.agreement(vs06, 1/2), Slice.ONE)
        self.assertEqual(Slice.agreement(vs43, 1/2), Slice.ONE)
        self.assertEqual(Slice.agreement(vs33, 1/2), Slice.ONE)

    def test_threshold_equivalence(self):
        rs = BHV.nrand(15)

        for i in range(16):
            self.assertEqual(BHV.threshold(rs, i), BHV._logic_threshold(rs, i))

    def test_window_equivalence(self):
        rs = BHV.nrand(17)

        for t in range(16):
            for b in range(t + 1):
                self.assertEqual(BHV.window(rs, b, t), BHV._ite_window(rs, b, t))

        self.assertEqual(BHV.window(rs, 3, 10), BHV._ite_window(rs, 3, 10))

    def test_real_window(self):
        self.assertEqual(BHV.window([BHV.ONE, BHV.ONE, BHV.ZERO, BHV.ZERO], 3, 3), BHV.ZERO)
        self.assertEqual(BHV.window([BHV.ONE, BHV.ONE, BHV.ONE, BHV.ZERO], 3, 3), BHV.ONE)
        self.assertEqual(BHV.window([BHV.ONE, BHV.ONE, BHV.ONE, BHV.ONE], 3, 3), BHV.ZERO)


        self.assertEqual(BHV.window([BHV.ONE, BHV.ZERO, BHV.ZERO, BHV.ZERO], 2, 3), BHV.ZERO)
        self.assertEqual(BHV.window([BHV.ONE, BHV.ONE, BHV.ZERO, BHV.ZERO], 2, 3), BHV.ONE)
        self.assertEqual(BHV.window([BHV.ONE, BHV.ONE, BHV.ONE, BHV.ZERO], 2, 3), BHV.ONE)
        self.assertEqual(BHV.window([BHV.ONE, BHV.ONE, BHV.ONE, BHV.ONE], 2, 3), BHV.ZERO)


if __name__ == '__main__':
    unittest.main()



