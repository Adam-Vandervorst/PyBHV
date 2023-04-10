import unittest

from bhv.np import NumPyPacked64BHV as BHV
from bhv.embedding import Random, InterpolateBetween


class TestRandomEmbedding(unittest.TestCase):
    def test_random(self):
        a, b, c = "abc"
        embedding = Random(BHV)
        hva = embedding.forward(a)
        hvb = embedding.forward(b)
        self.assertTrue(hva.unrelated(hvb))
        hva_ = embedding.forward(a)
        self.assertEqual(hva, hva_)

        hvq = BHV.rand()
        self.assertIsNone(embedding.back(hvq))

        self.assertEqual(b, embedding.back(hvb))


class TestInterpolateLineEmbedding(unittest.TestCase):
    def test_internal(self):
        embedding = InterpolateBetween(BHV)
        a, b, c = .1, .5, .68
        self.assertAlmostEqual(a, embedding.back(embedding.forward(a)), 2)


if __name__ == '__main__':
    unittest.main()
