import unittest

# from bhv.np import NumPyPacked64BHV as BHV, DIMENSION
# from bhv.pytorch import TorchPackedBHV as BHV, DIMENSION
from bhv.vanilla import VanillaBHV as BHV, DIMENSION
# from bhv.native import NativePackedBHV as BHV, DIMENSION
from bhv.symbolic import Var, SymbolicBHV


class TestFeistal(unittest.TestCase):
    def test_halves(self):
        h1 = BHV.EVEN
        h2 = BHV.ODD

        self.assertEqual(h1, ~h2)
        self.assertEqual(h1.active(), DIMENSION/2)
        self.assertEqual(h2.active(), DIMENSION/2)
        self.assertEqual(h1.hamming(h2), DIMENSION)

        self.assertEqual(h1.swap_even_odd(), h2)
        self.assertEqual(h2.swap_even_odd(), h1)

        r = BHV.rand()

        self.assertEqual(r.swap_even_odd().swap_even_odd(), r)
        self.assertEqual(r.swap_even_odd().swap_even_odd().swap_even_odd(), r.swap_even_odd())

    def test_keys(self):
        x = BHV.rand()

        ks = BHV.nrand(10)

        x_enc_k = [x.feistal(k) for k in ks]

        x_dec_k = [x_enc.feistal(k, inv=True) for x_enc, k in zip(x_enc_k, ks)]

        for k, x_enc, x_dec in zip(ks, x_enc_k, x_dec_k):
            self.assertEqual(x, x_dec, msg="inverse")
            self.assertNotEqual(x_enc, x_dec, msg="variant")
            self.assertNotEqual(x_enc, ~x_dec, msg="not opposite")
            self.assertNotEqual(x_enc, k, msg="not key")
            self.assertNotEqual(x_enc, BHV.EVEN, msg="not even")
            self.assertNotEqual(x_enc, BHV.ODD, msg="not odd")

        for x_enc_i in x_enc_k:
            eq = False
            for x_enc_j in x_enc_k:
                if x_enc_i == x_enc_j:
                    if eq: self.fail("different keys produce the same result")
                    eq = True
                else:
                    self.assertFalse(x_enc_i.related(x_enc_j))


class TestRehash(unittest.TestCase):
    def test_deterministic(self):
        v = BHV.rand()
        h1 = v.rehash()
        h2 = v.rehash()

        self.assertEqual(h1, h2)

    def test_random_different(self):
        vs = BHV.nrand(10)
        hs = [v.rehash() for v in vs]

        for v in hs:
            eq = False
            for v_ in hs:
                if v == v_:
                    if eq: self.fail("different keys produce the same result")
                    eq = True
                else:
                    self.assertFalse(v.related(v_))

    def test_close_different(self):
        seed = BHV.rand()
        vs = [seed.flip_pow(8) for _ in range(10)]

        for v in vs:
            eq = False
            for v_ in vs:
                if v == v_:
                    if eq: self.fail("flip_pow produced the same vector")
                    eq = True
                else:
                    self.assertTrue(v.related(v_))

        hs = [v.rehash() for v in vs]

        for v in hs:
            eq = False
            for v_ in hs:
                if v == v_:
                    if eq: self.fail("different keys produce the same result")
                    eq = True
                else:
                    self.assertFalse(v.related(v_))


if __name__ == '__main__':
    unittest.main()
