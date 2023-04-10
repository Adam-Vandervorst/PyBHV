import unittest

from bhv.np import NumPyPacked64BHV as BHV
from bhv.embedding import Random


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


if __name__ == '__main__':
    unittest.main()
