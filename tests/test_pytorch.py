import unittest

import torch

from bhv.pytorch import TorchPackedHV, TorchBoolHV


class TestTorchBoolBHV(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(True)


class TestTorchBHVConversion(unittest.TestCase):
    def test_random(self):
        rp = TorchPackedHV.rand()
        self.assertTrue(torch.equal(rp.data, rp.unpack().pack().data))
        ru = TorchBoolHV.rand()
        self.assertTrue(torch.equal(ru.data, ru.pack().unpack().data))

    def test_extrema(self):
        self.assertTrue(torch.equal(TorchPackedHV.ZERO.unpack().data, TorchBoolHV.ZERO.data))
        self.assertTrue(torch.equal(TorchPackedHV.ZERO.data, TorchBoolHV.ZERO.pack().data))
        self.assertTrue(torch.equal(TorchPackedHV.ONE.unpack().data, TorchBoolHV.ONE.data))
        self.assertTrue(torch.equal(TorchPackedHV.ONE.data, TorchBoolHV.ONE.pack().data))


if __name__ == '__main__':
    unittest.main()
