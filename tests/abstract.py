import unittest

from bhv.vanilla import VanillaBHV as BHV, DIMENSION


class BaseBHVMethods(unittest.TestCase):
    def test_closest_furthest(self):
        a, b, c, d = BHV.nrand(4)

        self.assertEqual(0, a.closest([a, b, c, d]))

        self.assertEqual(1, a.furthest([a, b]))
        self.assertEqual(1, a.furthest([BHV.majority([a, b, c]), BHV.majority([a, b, c, d])]))

    def test_top_bottom_k(self):
        a, b, c, d = BHV.nrand(4)

        self.assertEqual([0, 1], a.top([a, BHV.majority([a, b, c]), BHV.majority([a, b, c, d])], 2))
        self.assertEqual([1, 2], a.bottom_k([a, BHV.majority([a, b, c]), BHV.majority([a, b, c, d])], 2))

    def test_within_outside(self):
        a, b, c, d = BHV.nrand(4)

        self.assertEqual([0], a.within([a, b], 0))
        self.assertEqual([0, 1], a.within([a, b], DIMENSION))
        self.assertEqual([0], a.within([a ^ BHV.level(1)], DIMENSION))
        self.assertEqual([], a.within([a ^ BHV.level(1)], DIMENSION - 1))
        self.assertEqual([], a.within([a ^ BHV.level(.5)], DIMENSION/2 - 1))
        self.assertEqual([0], a.within([a ^ BHV.level(.5)], DIMENSION/2))
        self.assertEqual([0, 1], a.within([a, a ^ BHV.level(.001), a ^ BHV.level(.1), b], 100))

        self.assertEqual([], a.outside([a], 0))
        self.assertEqual([], a.outside([b], DIMENSION))
        self.assertEqual([], a.outside([a ^ BHV.level(1)], DIMENSION))
        self.assertEqual([0], a.outside([a ^ BHV.level(1)], DIMENSION - 1))
        self.assertEqual([0], a.outside([a ^ BHV.level(.5)], DIMENSION / 2 - 1))
        self.assertEqual([], a.outside([a ^ BHV.level(.5)], DIMENSION / 2))
        self.assertEqual([2, 3], a.outside([a, a ^ BHV.level(.001), a ^ BHV.level(.1), b], 100))

    def test_within_outside_std(self):
        a, b, c, d = BHV.nrand(4)

        self.assertEqual([0, 1], a.within([a, a ^ BHV.level(.0001), a ^ BHV.level(.01), b], 2))
        self.assertEqual([2, 3], a.outside([a, a ^ BHV.level(.0001), a ^ BHV.level(.1), b], 2))


if __name__ == '__main__':
    unittest.main()
